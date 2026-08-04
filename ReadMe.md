# Kinetic Ordinary Differential Equations Solver (KODES)

A CUDA C++ library for solving large ensembles of ordinary differential equations (ODEs) on the
GPU, aimed at chemical kinetics in CFD.

For CFD calculations with chemical reactions, due to the significant difference in the time scales
of chemical and hydrodynamic processes, it is customary to separate chemical reactions into a
separate step and solve them as ODEs in each cell of the computational grid. Systems from
neighboring cells do not affect each other, which allows direct parallelization of the entire
calculation process.

The classical approach of parallelizing the solution of a *single* ODE system (e.g. SUNDIALS,
which Cantera is built on) is limited here: its efficiency scales with vector size, but the number
of components (species) in a chemical system is fixed by the kinetic mechanism, not by the CFD
case, so there is little to parallelize within one system.

KODES takes the opposite approach: each individual ODE system (each mesh cell's chemistry) is
integrated by a single CUDA thread using a small, unparallelized stiff solver, while thousands of
systems are integrated **simultaneously**, one per thread, in a single kernel launch. Throughput
comes from the number of systems (mesh cells), not from parallelizing inside one system — which
matches how CFD scales (more cells, not bigger mechanisms).

## Core data layout

A "system" is one reactor (one mesh cell): a state vector of length `sizeOfSystem` plus
`numOfParameters` external parameters held constant over the step (e.g. pressure). A batch is
`numOfSystems` such systems integrated together. Host-side storage is *component-major* (one
pointer per state component, each pointing at that component's value across every system) so it
can alias existing per-species arrays without copying; device-side storage is a flat
`sizeOfSystem * numOfSystems` buffer indexed as `component*numOfSystems + system` (via the
`INDEXVEC`/`INDEXMAT` macros), so that all threads in a warp read/write consecutive memory for the
same component. Converting between the two layouts is the `Operator`'s job (see below).

## Classes

### Basic types and utilities

- **`basic_types.cuh`** — the library's fundamental typedefs: `scalar` (`double`) and `label`
  (`int`), used everywhere instead of the built-in types. Also defines `SMALL`/`GREAT` sentinel
  values, the `GRID_DIM`/`T_ID`/`INDEXVEC`/`INDEXMAT` device-indexing macros used throughout the
  integrators, and `stepState` — the small struct (forward/backward direction, trial and achieved
  step size, first/last/reject flags) threaded through an integrator's `solve()` call.
- **`basic_linalg.cuh`/`.cu`** — device-side building blocks used by the integrators:
  `LUDecompose`/`LUBacksubstitute` (in-place LU factorization and back-substitution),
  `hessenbergReduce`/`hessenbergShiftedFactorise`/`hessenbergSolve` (see *One reduction per
  Jacobian instead of one LU per stage* below), plus small helpers (`copyVec`, `sumVec`, `sqr`,
  `clamp`, `swap`, `normalizeError`).
- **`kodes::Config`** (`kodes_config.cuh`/`.cu`) — a small RapidJSON-backed JSON config-file reader
  (`getDouble`/`getInt`/`getString`/`getBool`/`hasKey` with defaults), independent of the ODE
  machinery. Requires the `external/rapidjson` submodule.
- **`kodes::mpiSelectDevice`** (`kodes_mpi.cuh`/`.cu`) — optional MPI device-binding helper for
  running across an arbitrary number of ranks and an arbitrary number of GPUs (single node or a
  heterogeneous multi-node cluster). Each rank is assumed to already own its own local slice of
  systems (e.g. via the host CFD code's own domain decomposition) — this only binds the calling
  rank's host thread to a CUDA device (round-robin over the GPUs visible on its node), it does not
  scatter/gather any data. Call it once per rank, right after `MPI_Init`, before creating any
  device-side `kodes` object; `kodes` never calls `MPI_Init`/`MPI_Finalize` itself. A separate
  translation unit from the rest of the library, so targets that don't need MPI never link it — see
  `examples/mpi_device_select`. For an end-to-end run combining this with a real solve, see
  `examples/integrators/GRIMECH_mpi` — the MPI counterpart to `GRIMECH/seulex3`, where each rank
  binds its own GPU and solves its own independent copy of the 257-system GRIMech 3.0 problem.

### `ODESystem` — the equations being integrated

- **`kodes::ODESystem`** (`ODESystem.cuh`) — abstract base every mechanism/test system implements:
  `derivatives(x, param, y, dydx)` and `jacobian(x, param, y, dfdx, dfdy)`, both `__device__`, plus
  `nEqns()`. Concrete systems also follow a (not formally enforced) `createGPU`/`destroyGPU` factory
  convention: a `__global__` placement-new kernel constructs the object directly in device memory,
  since the integrator kernels need a device-resident `ODESystem*`.
- **`kodes::GRIMESHSystem`** (`ODESystem/GRIMESHSystem.cuh`/`.cu`) — GRI-Mech 3.0 (53 species, 325
  reactions), generated by pyJac (`src/ODESystem/grimech/out`). `derivatives`/`jacobian` forward
  into the generated `dydt`/`eval_jacob` functions against a per-thread `mechanism_memory` scratch
  block (concentrations, rates, Jacobian workspace — allocated separately via
  `initialize_gpu_memory`/`free_gpu_memory`, distinct from the state buffers below). Compiled for
  constant pressure (`CONP`); state is `[T, Y_0..Y_{NSP-2}]` with the mechanism's designated last
  species (N2) recovered implicitly from mass conservation, pressure passed as the system parameter.
- **`kodes::H2O2System`** (`ODESystem/H2O2System.cuh`/`.cu`) — the same pattern applied to a
  smaller pyJac-generated H2/O2 mechanism (`src/ODESystem/h2o2/out`), used as a lighter-weight test
  case than GRI-Mech.
- **`kodes::HIRESSystem`** (`ODESystem/HIRESSystem.cuh`/`.cu`) — the HIRES stiff-ODE benchmark
  problem (8 equations), used to validate the integrator against a standard test case unrelated to
  chemistry.

### `Resources` — state storage, host and device

- **`kodes::Resources`** (`Resources/Resources.cuh`) — common base: `numOfSystems`,
  `sizeOfSystem`, `numOfParameters` and their accessors.
- **`kodes::HostResources`** (`Resources/HostResources.cuh`/`.cu`) — host-side state as an array of
  `sizeOfSystem` pointers (`vectors`) plus `numOfParameters` pointers (`parameters`); each pointer
  is caller-owned storage for that component across all systems (`setVector`/`setParameter` assign
  them directly — no ownership transfer, so callers can point straight at their own arrays).
- **`kodes::DeviceResources`** (`Resources/DeviceResources.cuh`/`.cu`) — the device-side
  counterpart: two flat `cudaMalloc`'d buffers (`vectors`, `parameters`), sized
  `sizeOfSystem*numOfSystems` and `numOfParameters*numOfSystems`. Built via a `create`/`destroy`
  pair (the object itself lives in device memory, constructed by a placement-new kernel).
- **`kodes::SeulexDeviceResources`** (`Resources/SeulexDeviceResources.cuh`/`.cu`) — extends
  `DeviceResources` with the extra device scratch the Seulex integrator needs per system: the
  polynomial extrapolation table, Jacobian (`dfdy`) and LU work matrix (`a`), pivot indices, and the
  various temporaries used between the outer step and the inner `seul` sub-stepping. `create` takes
  a host-allocated `SeulexDeviceResources` "stub" that stages the device pointers before copying the
  whole struct to the device, and that same stub is later read by `Operator` to know where the
  device buffers live.
- **`kodes::Operator`** (`Resources/Operator.cuh`) — copies state between a `HostResources` and a
  `DeviceResources` (or subclass): `cpyHostToDevice()`/`cpyDeviceToHost()` transfer every component
  and parameter, translating between the host's per-component pointers and the device's flat
  layout. Full-state copies each call; no partial/incremental transfer.

### `Integrators` — advancing the ODEs

- **`kodes::Integrator`** (`Integrators/Integrator.cuh`) — abstract base template
  (`Integrator<ODESystem, SolverDeviceResources>`) that fixes the CUDA launch configuration
  (threads/blocks/shared memory) from `numOfSystems` at construction time, and declares
  `solve(stepState)` for subclasses to implement as a kernel launch.
- **`kodes::Seulex`** (`Integrators/Seulex.cuh`/`.cu`) — a GPU port of the semi-implicit
  Bulirsch-Stoer extrapolation method (the same algorithm as OpenFOAM's own `seulex` ODE solver),
  one CUDA thread per system. `solve(stepState)` launches a single kernel that integrates every
  system in the batch from local time 0 to the step's target end-time, using
  a shift-invariant Hessenberg factorization for the implicit linear solves (see below) and
  polynomial extrapolation (`extrapolate`) to control step size and order. The step-control
  coefficients are compile-time `__constant__`s in this header, tolerances come from
  `IntegratorControls`. `SeulexProfile` collects a per-system cycle breakdown of the cost centres,
  printed for the system selected with `setProfileSystem`.

#### One reduction per Jacobian instead of one LU per stage

Stage `k` of the extrapolation splits the step `dtTot` into `nSeq_[k]` sub steps and solves with

    A(gamma) = gamma*I - J,   gamma = nSeq_[k]/dtTot

The Jacobian `J` is held fixed across all stages of a step, and across every following step until
`theta` says it has gone stale, so the stage matrices differ from each other only by a **multiple
of the identity**. Factorizing each from scratch costs `O(n^3)` per stage and used to dominate the
run time, while a stage only solves `nSeq_[k]` right hand sides — far fewer than the `~n/3`
back-substitutions an LU is worth.

An LU factorization cannot be updated cheaply under a full-rank diagonal shift (Sherman-Morrison
and Woodbury only cover low-rank changes), but an **orthogonal similarity is shift invariant**:

    J = Q H Q^T   =>   gamma*I - J = Q (gamma*I - H) Q^T

So `J` is reduced once to upper Hessenberg form by Householder reflections (`hessenbergReduce`,
`(10/3)n^3`, charged to the Jacobian evaluation that produced it), and after that

- `hessenbergShiftedFactorise` builds `gamma*I - H` and eliminates its single subdiagonal in
  `O(n^2)` — about `n/3` times cheaper than the LU it replaces, and the pivoting collapses to one
  bit per column since the interchange can only ever be between neighbouring rows;
- `hessenbergSolve` applies `Q^T`, the triangular solves and `Q`, all `O(n^2)`. `Q` is never
  formed: the reflections are read straight out of the space below the subdiagonal of the reduced
  matrix, where the zeros they create would otherwise sit.

The reduced Jacobian lives in `dfdy` (overwritten in place by `hessenbergReduce`), the reflection
coefficients in `hessTau`, and the per-stage factors in the existing `a` work matrix — the only
new storage is `hessTau`, one vector per system.

The trade is `O(n^3)` per stage against `O(n^3)` per Jacobian plus a solve that costs about 2.5
back-substitutions instead of 1 (the two orthogonal transforms). It therefore pays off in
proportion to how many stage matrices one Jacobian serves; the profile print reports that ratio
directly as *stage factorisations per reduction*. Counting memory traffic in units of `n^2`, a
reduction is worth `~2.5n` and an LU `~(2/3)n`, so for GRI-Mech (`n = 53`) break-even sits at
roughly four to five stage factorizations per reduction — about what a single step already
produces, and every further step the Jacobian survives is pure gain.
