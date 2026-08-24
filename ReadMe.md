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

A "system" is one reactor (one mesh cell): a state vector of length `systemSize` plus
`parameterSize` external parameters held constant over the step (e.g. pressure). An ensemble of
`ensembleSize` systems is shipped to the device in batches of `batchSize`. Host-side storage is
*component-major* (one pointer per state component, each pointing at that component's value across
every system) so it can alias existing per-species arrays without copying. Converting between the
host and device layouts is the `Operator`'s job (see below).

On the device there are two distinct address spaces, and keeping them apart is what lets an
arbitrarily large ensemble run on a fixed amount of VRAM:

- **state space** — the batch itself: `vectors` (`systemSize * batchSize`), `parameters` and the
  per-system step bookkeeping. One slot per *system*, stride `batchSize`, addressed with
  `INDEXSTATE(system, component, batchSize)`. Only a few dozen bytes per system, so a batch can be
  made large enough to fill the free VRAM, which keeps the number of `cudaMemcpy` rounds low.
- **scratch space** — the per-thread temporaries an implicit method needs (Jacobian and LU work
  matrix, both `systemSize^2`, the extrapolation table, the pivot indices, ...). One slot per
  *resident thread*, stride `GRID_DIM` (the number of threads actually launched), addressed with
  `INDEXVEC`/`INDEXMAT`. Allocating more of these than the device can keep resident at once is
  pure waste, and for a large mechanism they are what actually exhausts the VRAM.

A thread walks its share of the batch in a grid-stride loop: `DeviceResources::loadSystem(system)`
pulls one system into the thread's scratch slot, the integrator works entirely in scratch space
(which is also the layout pyJac's generated `dydt`/`eval_jacob` expect — its `INDEX` macro is
identical to `INDEXVEC`), and `storeSystem(system)` writes the result back. Both directions stay
coalesced: consecutive threads touch consecutive systems.

The grid size — and hence the number of scratch slots — is determined at run time from the
occupancy of the solve kernel for the mechanism at hand, see `LaunchConfig` below.

## Classes

### Basic types and utilities

- **`basic_types.cuh`** — the library's fundamental typedefs: `scalar` (`double`) and `label`
  (`int`), used everywhere instead of the built-in types. Also defines `SMALL`/`GREAT` sentinel
  values and the device-indexing macros used throughout the integrators: `GRID_DIM`/`T_ID` plus
  `INDEXVEC`/`INDEXMAT` for scratch space and `INDEXSTATE` for state space (see *Core data layout*).
- **`kodes::LaunchConfig`** (`Integrator/LaunchConfig.cuh`) — how a solve is mapped onto the
  device: `threads`×`blocks` (`== scratchSize`, the systems integrated *simultaneously*, and thus
  the number of scratch slots to allocate), `batchSize` (systems per host↔device transfer) and the
  dynamic shared memory per block. `maxConcurrentThreads()` answers the first half by asking
  `cudaOccupancyMaxActiveBlocksPerMultiprocessor` how many blocks of the *actual solve kernel* fit
  on an SM — that depends on the kernel's register and shared-memory footprint and therefore on the
  mechanism, so it can only be known at run time. `kodes::planLaunch<ODESystem, IntegrationMethod,
  Resources>(...)` (declared in `Integrator.cuh`) combines it with `cudaMemGetInfo` and the
  resource class's `scratchBytesPerThread`/`stateBytesPerSystem` into a complete plan: shrink the
  grid until the temporaries fit, then spend the rest of the memory budget on the batch. Pass any
  scratch owned outside the resources object (for a pyJac mechanism, `required_mechanism_size()`)
  as `extraScratchBytesPerThread`, and pad that allocation to `scratchSize`, not to `batchSize`.
- **`basic_linalg.cuh`/`.cu`** — device-side building blocks used by the integrators:
  `LUDecompose`/`LUBacksubstitute` (in-place LU factorization and back-substitution, used to solve
  the linear systems in each implicit step), plus small helpers (`copyVec`, `sumVec`, `sqr`,
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
  counterpart, and the place where the two address spaces meet. State space: `vectors` and
  `parameters`, sized `systemSize*batchSize` and `parameterSize*batchSize`. Scratch space: `y_` and
  `param_`, the working copy of the system a thread currently integrates, sized with `scratchSize`.
  `loadSystem(system)`/`storeSystem(system)` move one system between the two. Built via a
  `create`/`destroy` pair (the object itself lives in device memory, constructed by a placement-new
  kernel). The static `stateBytesPerSystem`/`scratchBytesPerThread` report the per-system and
  per-thread cost to `planLaunch`; every subclass extends the latter.
- **`kodes::SeulexDeviceResources`** (`Resources/IntegratorDeviceResources/Seulex/…`) — extends
  `DeviceResources` with the extra scratch the Seulex integrator needs per thread: the polynomial
  extrapolation table, Jacobian (`dfdy`) and LU work matrix (`a`), pivot indices, and the various
  temporaries used between the outer step and the inner `seul` sub-stepping. All of it is sized
  with `scratchSize`, so the `systemSize^2` blocks are allocated once per resident thread rather
  than once per system of the batch. `create` takes a host-allocated `SeulexDeviceResources` "stub"
  that stages the device pointers before copying the whole struct to the device, and that same stub
  is later read by `Operator` to know where the device buffers live.
- **`kodes::Operator`** (`Resources/Operator.cuh`) — copies state between a `HostResources` and a
  `DeviceResources` (or subclass): `cpyHostToDevice(batchIndex)`/`cpyDeviceToHost(batchIndex)`
  transfer every component and parameter of one batch, translating between the host's per-component
  pointers and the device's flat state-space layout. `getRealBatchSize(batchIndex)` gives the size
  of that batch — the last one is normally shorter than `batchSize`.

### `Integrators` — advancing the ODEs

- **`kodes::Integrator`** (`Integrator/Integrator.cuh`) — template
  (`Integrator<ODESystem, IntegrationMethod, IntegratorDeviceResources>`) driving the solve. It is
  handed a `LaunchConfig` at construction and owns the `adaptive_solve` kernel: the grid-stride
  loop over the batch, the outer step-count loop that walks one system from local time 0 to the
  target end-time, and — for methods that declare `useAdaptiveStep` — the step-size controller
  `adaptiveStep`. `solve(deltaT, realBatchSize)` launches it for one batch, `setDeltaT` seeds the
  step state of the whole batch.
- **`kodes::Seulex`** (`Integrator/IntegrationMethods/Seulex/…`) — a GPU port of the semi-implicit
  Bulirsch-Stoer extrapolation method (the same algorithm as OpenFOAM's own `seulex` ODE solver).
  `step()` advances the system the calling thread currently holds in its scratch slot, using
  `LUDecompose`/`LUBacksubstitute` for the implicit linear solves and polynomial extrapolation
  (`extrapolate`) to control step size and order. Absolute/relative tolerances and the step-control
  coefficients are compile-time `__constant__`s in this header, not runtime-configurable.

## Typical run

See `examples/integrators/GRIMECH/seulex5.cu`. The order matters: plan first (it needs the free
VRAM before anything has been allocated), then size every allocation from the plan.

```cpp
// 1) ask the device how many systems it can run at once and how big a batch fits
kodes::LaunchConfig config = kodes::planLaunch
<
    kodes::pyJacSystem,
    kodes::Seulex<kodes::pyJacSystem>,
    kodes::SeulexDeviceResources
>
(
    ensembleSize, host_res.systemSize(), host_res.parameterSize(),
    required_mechanism_size()          // pyJac's own per-thread scratch
);

// 2) per-thread scratch is padded to the resident threads, not to the batch
initialize_gpu_memory(config.scratchSize, &h_mem, &d_mem);

kodes::SeulexDeviceResources  stub(config.batchSize, config.scratchSize, systemSize, parameterSize);
auto* res = kodes::SeulexDeviceResources::create
(
    config.batchSize, config.scratchSize, systemSize, parameterSize, &stub
);

// 3) solve batch by batch
kodes::Integrator<kodes::pyJacSystem, kodes::Seulex<kodes::pyJacSystem>, kodes::SeulexDeviceResources>
    solver(ode, res, config, controls);

solver.setDeltaT(xEnd);

for (label i = 0; i < config.numOfBatches(ensembleSize); i++)
{
    op.cpyHostToDevice(i);
    solver.solve(xEnd, op.getRealBatchSize(i));
    op.cpyDeviceToHost(i);
}
```
