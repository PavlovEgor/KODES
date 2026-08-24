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
  not to `batchSize`) and state per system. A balancer declares both: `bytesPerSystem()` for the
  keys and the order, and `scratchBytesPerThread()` for the slot a key that evaluates the right
  hand side writes into.
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

- **`kodes::Balancer`** (`Balancer/Balancer.cuh`/`.cu`) — abstract base holding the arrays the
  ordering needs: `keys` (`scalar`, `numOfKeys*batchSize`), `bucket` (`label`, which bucket each
  system fell in) and `order` (`label`, the traversal order, i.e. `order[i]` is the system to
  integrate at position `i`), plus the `KODES_BALANCER_BUCKETS`-long histogram. Subclasses
  implement the one abstract function, `__device__ void key(resources, ode, system, key)`, which
  fills `key[0 … numOfKeys-1]`. The batch itself is never moved — only the order in which the
  kernel walks it — so `Operator` is untouched.
- **`balance()`** — a bucket sort, four kernels on the default stream, no host round trip and no
  synchronisation before the solve that follows:
  1. `fillKeys` fills `keys` through the virtual call and reduces the range of each of them. A warp
     folds its own lanes with `__shfl_down_sync` first, so a range costs one pair of atomics per
     warp; and since there is no `atomicMin` for `double`, the keys are compared as the unsigned
     integers of the same order (`orderedBits`).
  2. `fillBuckets` puts each system in one of `KODES_BALANCER_BUCKETS` buckets and counts how many
     land in each. Every key is cut into the same number of equal bins of its own range, and the
     bins are mixed into one bucket index most significant first, so the buckets run in
     lexicographic order: key 1 only ever reorders systems that already share a bin of key 0.
  3. `scanBuckets` turns the histogram into the offset each bucket starts at — one block walking it
     in chunks, carrying the running total in shared memory. A few thousand entries do not repay a
     second kernel for the block offsets.
  4. `scatterOrder` gives every system the next free slot of its bucket, with one `atomicAdd`.

  The order is therefore exact *between* buckets and arbitrary *inside* one: the batch comes out
  sorted to within one bucket width, and a bucket is ~60 systems, two warps, for a batch of a
  million. That is the whole point of the balancing, so a comparison sort would only buy a
  distinction the warps cannot feel. It also means the pass is O(n) rather than O(n log n) and that
  the keys never leave the device — the previous host quicksort cost a round trip of ~80 ms per
  million systems. What it does *not* give is a reproducible permutation: which system of a bucket
  lands in which of its slots depends on the order the atomics happen to fire in. The results do
  not depend on it, since the systems are integrated independently.

  A key that is not a number — a system that has already blown up — is kept out of the range and
  binned first, where it cannot drag a warp of healthy cells along.

The bucket budget is shared out between the keys — `binsPerKey` gives 16384 bins to one key, 128
to each of two, 24 to each of three — so every key added buys a distinction and pays for it in
resolution on the ones already there. Order them by how much they matter; at most
`KODES_MAX_KEYS` of them. Three are provided:

- **`kodes::TemperatureBalancer`** — one key, component 0 of the state vector. The cheapest: it is
  already in the state, so the pass reads one scalar per system and does nothing else.
- **`kodes::RHSNormBalancer`** — one key, `log10` of the RMS *relative* rate of change of the
  state, `relativeRHSNorm()`. Where temperature is a proxy for stiffness this is a measurement of
  it: an inverse time scale, and about as direct a statement of how small a step the system will
  need as one right-hand-side evaluation can give. Relative because `dT/dt` is orders of magnitude
  above every `dY_i/dt`, so an absolute norm would report nothing but the temperature; `log10`
  because the value runs over a dozen decades and the bins are equal in the key. It costs one
  `derivatives()` call per system per batch — for a generated mechanism, one explicit step's worth
  of work against a solve of hundreds of implicit ones.
- **`kodes::StiffnessBalancer`** — two keys: temperature, then the norm inside each band of it.
  Temperature alone leaves a band holding fresh mixture next to burnt gas, which do not need the
  same number of steps; the norm alone puts a cold cell and a hot equilibrated one in the same bin
  though their Jacobians have nothing in common. This is what more than one key is for, and what
  `examples/integrators/GRIMECH/seulex5.cu` uses.

All three are built with the same `create`/`destroy` host-stub pair as the device resources, and
all take the same `(batchSize, scratchSize, systemSize)`. Each declares its own
`bytesPerSystem()`/`scratchBytesPerThread(systemSize)` for `planLaunch` — a key that evaluates the
right hand side needs a `systemSize` scratch slot per resident thread to write it into.

Unlike the resources, the stub can also be asked for with `createStub`/`destroyStub` instead of
being declared by the caller: `key()` is a *device-only virtual*, so the vtable of a host side
object can only be emitted by a compiler that invents a host stub for one — nvcc does, `nvc++
-cuda` does not and stops with an undefined `key` in the vtable. A caller compiled by anything
other than nvcc (an OpenFOAM chemistry model, say) must therefore hold the stub as a pointer and
let the `.cu` construct it. `Balancer/BalancerFactory.cuh` holds those four steps once, and is
included only by the `.cu` of a subclass — never by a caller, since it is the file that launches
the kernels. Adding a balancer is a subclass with a `key()`, plus four one-line forwards to it.

`Integrator::setBalancer(balancer, hostStub)` points the resources at the balancer's order array
and rebalances at the start of every `solve()`, since a new batch brings new cells. Without it the
traversal is the copy order. The integrator hands the balancer its `ODESystem*` on every pass, so
an `ODESystem` template argument that does not derive from `kodes::ODESystem` can only be used
with a balancer whose keys stay out of the right hand side.

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
    // pyJac's own per-thread scratch, plus the slot the norm key writes into
    required_mechanism_size() + kodes::StiffnessBalancer::scratchBytesPerThread(NSP),
    kodes::StiffnessBalancer::bytesPerSystem(),   // the keys and the order, per system
    kodes::LaunchConfig("best")           // or "half", or explicit sizes
);

// 2) per-thread scratch is padded to the resident threads, not to the batch
initialize_gpu_memory(config.scratchSize, &h_mem, &d_mem);

kodes::SeulexDeviceResources  stub(config.batchSize, config.scratchSize, systemSize, parameterSize);
auto* res = kodes::SeulexDeviceResources::create
(
    config.batchSize, config.scratchSize, systemSize, parameterSize, &stub
);

// 3) solve batch by batch, each one sorted by temperature and stiffness first
kodes::Integrator<kodes::pyJacSystem, kodes::Seulex<kodes::pyJacSystem>, kodes::SeulexDeviceResources>
    solver(ode, res, config, controls);

kodes::StiffnessBalancer balancerStub(config.batchSize, config.scratchSize, systemSize);
auto* balancer = kodes::StiffnessBalancer::create
(
    config.batchSize, config.scratchSize, systemSize, &balancerStub
);
solver.setBalancer(balancer, &balancerStub);

solver.setDeltaT(tEnd);

for (label i = 0; i < config.numOfBatches(ensembleSize); i++)
{
    op.cpyHostToDevice(i);
    solver.solve(tEnd, op.getRealBatchSize(i));
    op.cpyDeviceToHost(i);
}
```
