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

- **state space** — the batch itself: `vectors` (`systemSize * batchSize`) and `parameters`. One
  slot per *system*, stride `batchSize`, addressed with `INDEXSTATE(system, component, batchSize)`.
  A handful of scalars per system, so a batch can be made large enough to fill the free VRAM, which
  keeps the number of `cudaMemcpy` rounds low.
- **scratch space** — the working copy of the system a thread is integrating, its step state
  (`StepState`) and the temporaries an implicit method needs (Jacobian and LU work matrix, both
  `systemSize^2`, the extrapolation table, the pivot indices, ...). One slot per *resident thread*,
  stride `GRID_DIM` (the number of threads actually launched), addressed with `INDEXVEC`/`INDEXMAT`.
  Allocating more of these than the device can keep resident at once is pure waste, and for a large
  mechanism they are what actually exhausts the VRAM. Nothing here outlives the system that is
  currently in the slot — a later batch would overwrite it anyway — so none of it is kept per
  system.

A thread walks its share of the batch in a grid-stride loop over the *balanced* order (see
`Balancer` below, identity order when none is set): `DeviceResources::loadSystem(system)`
pulls one system into the thread's scratch slot (`currentVector`/`currentParameters`), the
integrator works entirely in scratch space (which is also the layout pyJac's generated
`dydt`/`eval_jacob` expect — its `INDEX` macro is identical to `INDEXVEC`), and
`storeSystem(system)` writes the result back. Both directions stay coalesced: consecutive threads
touch consecutive systems.

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
  dynamic shared memory per block. It is *requested* by one of two constructors and *resolved* by
  `planLaunch`:

  ```cpp
  kodes::LaunchConfig("best")            // everything the device offers (default)
  kodes::LaunchConfig("half")            // one half of it, to share the GPU with another process
  kodes::LaunchConfig("best", 128)       // ... with 128 threads per block
  kodes::LaunchConfig(8192, 1000000)     // concurrent systems and batch size set by hand
  ```

  The named shares live in the `kodes::deviceShares` table — add a line there to add a name. A
  share scales both the concurrency and the memory budget; sizes set by hand are only checked
  against the free VRAM. `KODES_MEMORY_HEADROOM` caps how much of the free VRAM any plan may claim.
- **`kodes::planLaunch<ODESystem, IntegrationMethod, Resources>(...)`** (declared in
  `Integrator.cuh`) — resolves a request against the device. `maxConcurrentThreads()` asks
  `cudaOccupancyMaxActiveBlocksPerMultiprocessor` how many blocks of the *actual solve kernel* fit
  on an SM — that depends on the kernel's register and shared-memory footprint and therefore on the
  mechanism, so it can only be known at run time. That, `cudaMemGetInfo` and the resource class's
  `scratchBytesPerThread`/`stateBytesPerSystem` go into `makePlan()`, which holds all the sizing
  arithmetic and no CUDA call: cap the grid at what the budget affords, then spend the rest on the
  batch. Memory owned outside the resources object is declared through the two extras — scratch per
  thread (for a pyJac mechanism, `required_mechanism_size()`; pad that allocation to `scratchSize`,
  not to `batchSize`) and state per system (`Balancer::bytesPerSystem()` when the batch is sorted).
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
  `parameters`, sized `systemSize*batchSize` and `parameterSize*batchSize`. Scratch space:
  `currentVector()` and `currentParameter(i)`, the working copy of the system a thread is
  integrating right now, sized with `scratchSize`.
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

### `Balancer` — who runs next to whom

A warp runs at the speed of its stiffest member: if one lane needs 400 steps and the other 31 need
12, the whole warp pays 400. Sorting the batch by a scalar property before integrating it puts
similar cells next to each other, and since a thread picks up positions `T_ID`, `T_ID + GRID_DIM`,
… the 32 lanes of a warp always get 32 *neighbours* of that order.

- **`kodes::Balancer`** (`Balancer/Balancer.cuh`/`.cu`) — abstract base holding the two arrays the
  ordering needs, both `batchSize` long: `keys` (`scalar`) and `order` (`label`, the traversal
  order, i.e. `order[i]` is the system to integrate at position `i`). Subclasses implement the one
  abstract function, `__device__ scalar key(resources, system)`. `balance()` runs the whole pass:
  a kernel fills `keys` through that virtual call, the keys come back to the host, `quickSortByKey`
  sorts them carrying `order` along, and the resulting order is uploaded. The batch itself is never
  moved — only the order in which the kernel walks it — so `Operator` is untouched.
- **`kodes::quickSortByKey`** (`Balancer/Balancer.cuh`) — in-place quicksort of the keys applying
  every move to the index array as well. Iterative (explicit stack, always recursing into the
  smaller half so the stack stays under `log2(size)` entries), median-of-three pivot, Hoare
  partitioning so that runs of equal keys — a cold field is mostly one temperature — stay O(n log n),
  and a final insertion pass for short ranges. Host side: it costs one round trip of the keys per
  batch (~80 ms per million systems). If it ever shows up in a profile, `thrust::sort_by_key` is a
  drop-in replacement inside `balance()`.
- **`kodes::TemperatureBalancer`** (`Balancer/TemperatureBalancer.cuh`/`.cu`) — the simplest useful
  key: component 0 of the state vector, the temperature. Built with the same `create`/`destroy`
  host-stub pair as the device resources.

`Integrator::setBalancer(balancer, hostStub)` points the resources at the balancer's order array
and rebalances at the start of every `solve()`, since a new batch brings new cells. Without it the
traversal is the copy order.

### `Integrators` — advancing the ODEs

- **`kodes::Integrator`** (`Integrator/Integrator.cuh`) — template
  (`Integrator<ODESystem, IntegrationMethod, IntegratorDeviceResources>`) driving the solve. It is
  handed a `LaunchConfig` at construction and owns the `adaptive_solve` kernel: the grid-stride
  loop over the batch, the outer step-count loop that walks one system from local time 0 to the
  target end-time, and — for methods that declare `useAdaptiveStep` — the step-size controller
  `adaptiveStep`. `solve(deltaT, realBatchSize)` launches it for one batch, `setDeltaT` seeds the
  step state of every thread slot. `resetStep()` reseeds a slot for its next system from the trial
  step the previous one ended with, so each system starts warm from its predecessor in that slot.
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
    required_mechanism_size(),            // pyJac's own per-thread scratch
    kodes::Balancer::bytesPerSystem(),    // the sorted order, per system
    kodes::LaunchConfig("best")           // or "half", or explicit sizes
);

// 2) per-thread scratch is padded to the resident threads, not to the batch
initialize_gpu_memory(config.scratchSize, &h_mem, &d_mem);

kodes::SeulexDeviceResources  stub(config.batchSize, config.scratchSize, systemSize, parameterSize);
auto* res = kodes::SeulexDeviceResources::create
(
    config.batchSize, config.scratchSize, systemSize, parameterSize, &stub
);

// 3) solve batch by batch, each one sorted by temperature first
kodes::Integrator<kodes::pyJacSystem, kodes::Seulex<kodes::pyJacSystem>, kodes::SeulexDeviceResources>
    solver(ode, res, config, controls);

kodes::TemperatureBalancer balancerStub(config.batchSize);
auto* balancer = kodes::TemperatureBalancer::create(config.batchSize, &balancerStub);
solver.setBalancer(balancer, &balancerStub);

solver.setDeltaT(tEnd);

for (label i = 0; i < config.numOfBatches(ensembleSize); i++)
{
    op.cpyHostToDevice(i);
    solver.solve(tEnd, op.getRealBatchSize(i));
    op.cpyDeviceToHost(i);
}
```
