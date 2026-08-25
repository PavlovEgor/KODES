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

Which integration method runs, which balancer orders the batch and every tolerance are read from a
JSON settings file when the program starts — see *Typical run* below. How a class whose virtual
functions only exist on the GPU can be selected by name at run time is a pattern used everywhere in
the library, and **[`DeviceObjects_explained.md`](DeviceObjects_explained.md)** explains it in full.

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
- **`kodes::planLaunch(...)`** (declared in `Integrator.cuh`) — resolves a request against the
  device. `maxConcurrentThreads()` asks `cudaOccupancyMaxActiveBlocksPerMultiprocessor` how many
  blocks of the *actual solve kernel* fit on an SM — that depends on the kernel's register and
  shared-memory footprint, so it can only be known at run time. That, `cudaMemGetInfo` and what the
  chosen classes cost go into `makePlan()`, which holds all the sizing arithmetic and no CUDA call:
  cap the grid at what the budget affords, then spend the rest on the batch. The costs come from the
  method's and the balancer's *names* — every type table entry answers
  `scratchBytesPerThread`/`stateBytesPerSystem`, so the plan can be made before either object
  exists, which it has to be, since it is the plan that fixes their sizes. Memory owned outside both
  (for a pyJac mechanism, `required_mechanism_size()`; pad that allocation to `scratchSize`, not to
  `batchSize`) goes in the extra.
- **`basic_linalg.cuh`/`.cu`** — device-side building blocks used by the integrators:
  `LUDecompose`/`LUBacksubstitute` (in-place LU factorization and back-substitution, used to solve
  the linear systems in each implicit step), plus small helpers (`copyVec`, `sumVec`, `sqr`,
  `clamp`, `swap`, `normalizeError`).
- **`kodes::Config`** (`kodes_config.cuh`/`.cu`) — a small RapidJSON-backed JSON config-file reader
  (`getDouble`/`getInt`/`getString`/`getBool`/`hasKey` with defaults), independent of the ODE
  machinery. Keys are dotted paths, so `"controls.absTol"` reaches into the nested object. Requires
  the `external/rapidjson` submodule.
- **`kodes::Settings`** (`Settings/Settings.cuh`/`.cu`) — the whole of a run in one JSON file: the
  name of the method, the name of the balancer, the `LaunchConfig` to resolve and the
  `IntegratorControls`. Every entry has a default, and both names are looked up in their table in
  the constructor, so a typo fails before anything has been allocated on the device. See
  `examples/integrators/GRIMECH/seulex5.json`. It is a source list of its own in
  `cmake/kodes.cmake` — the only part of the library needing rapidjson — because a caller with its
  own settings to read (the OpenFOAM chemistry model reads an OpenFOAM dictionary) passes the same
  names and numbers by hand and never links it.

### Device objects — one pattern for every class chosen at run time

Which method integrates, which balancer orders the batch and which mechanism is integrated are all
decisions made when the program starts, and all three are made the same way. The mechanics are
worth reading once — **`DeviceObjects_explained.md`** covers them in full, with the reasons — but
in short:

- **`Factory/DeviceObject.cuh`** — how such an object is built. It lives in device memory, because
  a virtual call from inside a kernel needs a device vtable, and its buffers are allocated by the
  host into a *host stub* of the same type: allocate on the stub, byte-copy it to the device,
  placement-new on top of the copy. That order is why the constructor of such a class must set
  value members only — it runs after the copy and would overwrite the addresses it brought. A
  concrete class keeps a three-point contract (that constructor, a non-virtual
  `allocate()`/`deallocate()` pair, a placement `operator new`) and gets its four host statics —
  `create`/`destroy`/`createStub`/`destroyStub` — from one line of `KODES_DEFINE_DEVICE_OBJECT` in
  its `.cu`. The host half of the pattern is static polymorphism (the factory is a template, so
  `stub->allocate()` resolves to the concrete one); only the device half uses a vtable.
- **`Factory/TypeTable.cuh`** — how one of them is chosen by name. A `TypeEntry` is a concrete class
  reduced to plain function pointers, so the dispatch needs no host vtable at all: a caller nvcc did
  not compile can still select and own a class whose virtuals only exist on the device. That is the
  reason for `createStub` — `key()` and `step()` are device-only virtuals, and the host vtable of a
  class with one can only be emitted by a compiler that invents a host stub for it, which nvcc does
  and `nvc++ -cuda` does not. Keeping construction inside the library's own `.cu` leaves callers
  holding nothing but a pointer. `Handle<Base>` owns the device object and its stub together and
  returns both to the class that made them.

The tables are `Balancer/balancerTable.cu` and `Integrator/IntegrationMethods/methodTable.cu`; a
method's entry names the `DeviceResources` subclass holding its scratch, since the two can only be
chosen together.
- **`kodes::mpiSelectDevice`** (`kodes_mpi.cuh`/`.cu`) — optional MPI device-binding helper for
  running across an arbitrary number of ranks and an arbitrary number of GPUs (single node or a
  heterogeneous multi-node cluster). Each rank is assumed to already own its own local slice of
  systems (e.g. via the host CFD code's own domain decomposition) — this only binds the calling
  rank's host thread to a CUDA device (round-robin over the GPUs visible on its node), it does not
  scatter/gather any data. Call it once per rank, right after `MPI_Init`, before creating any
  device-side `kodes` object; `kodes` never calls `MPI_Init`/`MPI_Finalize` itself. A separate
  translation unit from the rest of the library, so targets that don't need MPI never link it — see
  `examples/mpi_device_select`.

### `ODESystem` — the equations being integrated

- **`kodes::ODESystem`** (`ODESystem.cuh`) — abstract base every mechanism implements:
  `derivatives(x, param, y, dydx)` and `jacobian(x, param, y, dfdx, dfdy)`, both `__device__`. A
  device object like the rest, so a `__global__` placement-new kernel constructs it directly in
  device memory — the solve kernel needs a device-resident `ODESystem*`. Its constructor takes the
  mechanism's scratch rather than the four sizes, so it writes its own `create`/`destroy` instead of
  taking them from `KODES_DEFINE_DEVICE_OBJECT`.
- **`kodes::pyJacSystem`** (`ODESystem/pyJacSystem.cuh`/`.cu`) — any mechanism generated by pyJac.
  `derivatives`/`jacobian` forward into the generated `dydt`/`eval_jacob` against a per-thread
  `mechanism_memory` scratch block (concentrations, rates, Jacobian workspace — allocated separately
  via `initialize_gpu_memory`/`free_gpu_memory`, distinct from the state buffers below). Compiled
  for constant pressure (`CONP`); state is `[T, Y_0..Y_{NSP-2}]` with the mechanism's designated
  last species recovered implicitly from mass conservation, pressure passed as the system parameter.

  Which mechanism it is is a compile-time choice, unlike everything else in the library: pyJac
  generates C for one mechanism, and `NSP` is a macro of its `mechanism.cuh`. Two are in the tree —
  GRI-Mech 3.0 (53 species, 325 reactions, `src/ODESystem/grimech/out`) and a smaller H2/O2
  mechanism (`src/ODESystem/h2o2/out`) — and `kodes_pyjac_mechanism(grimech ...)` in
  `cmake/kodes.cmake` picks one by directory name.

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
  `loadSystem(system)`/`storeSystem(system)` move one system between the two. A device object like
  every other, so its buffers are allocated by its own `allocate()` into a host stub that `Operator`
  then reads to find them. The static `stateBytesPerSystem`/`scratchBytesPerThread` report the
  per-system and per-thread cost to `planLaunch`; every subclass extends the latter.
- **`kodes::SeulexDeviceResources`** (`Resources/IntegratorDeviceResources/Seulex/…`) — extends
  `DeviceResources` with the extra scratch the Seulex integrator needs per thread: the polynomial
  extrapolation table, Jacobian (`dfdy`) and LU work matrix (`a`), pivot indices, and the various
  temporaries used between the outer step and the inner `seul` sub-stepping. All of it is sized
  with `scratchSize`, so the `systemSize^2` blocks are allocated once per resident thread rather
  than once per system of the batch. Its `allocate()` also fills the `__constant__` step sequence
  and extrapolation coefficients, since it is that same tableau — `iMaxx_` — which sizes the
  order-indexed part of the scratch.
- **`kodes::AdaptiveDeviceResources`**/**`kodes::EulerDeviceResources`** — the same for a method
  that takes trial steps: the two vectors `IntegrationMethod::adaptiveStep` needs to hold a trial
  state and the derivative it started from, plus the error vector Euler writes.
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

All three are *device objects* in the sense of the section below, so all three are built the same
way and picked by name — `"temperature"`, `"rhsNorm"`, `"stiffness"` — out of the table in
`Balancer/balancerTable.cu`, with `"none"` for no balancing at all. Each declares its own
`stateBytesPerSystem`/`scratchBytesPerThread` for `planLaunch`: a key that evaluates the right hand
side needs a `systemSize` scratch slot per resident thread to write it into.

Adding a balancer is a subclass with a `key()`, one `KODES_DEFINE_DEVICE_OBJECT` line in its `.cu`
and one line in the table.

`Integrator::setBalancer(balancer, hostStub)` points the resources at the balancer's order array
and rebalances at the start of every `solve()`, since a new batch brings new cells. Nulls — which
is what the `"none"` name gives — leave the traversal in the copy order.

### `Integrators` — advancing the ODEs

- **`kodes::IntegrationMethod`** (`Integrator/IntegrationMethods/IntegrationMethod.cuh`) — abstract
  base for one numerical method, a device object like the balancer: `__device__ virtual scalar
  step(ode, resources, controls)` advances the system the calling thread holds in its scratch slot,
  and which subclass runs is a name in `methodTable`. `usesAdaptiveStep()` says whether that step is
  a *trial* step, in which case the base's own `adaptiveStep()` — the step-size controller — retries
  it smaller until the error is inside the tolerance and then grows the next one. A method owns no
  storage: the `DeviceResources` subclass named in the same table entry holds all of it, which is
  what makes it safe for `step()` to cast the base pointer down to it.
- **`kodes::Seulex`** (`Integrator/IntegrationMethods/Seulex/…`) — a GPU port of the semi-implicit
  Bulirsch-Stoer extrapolation method (the same algorithm as OpenFOAM's own `seulex` ODE solver).
  `step()` uses `LUDecompose`/`LUBacksubstitute` for the implicit linear solves and polynomial
  extrapolation (`extrapolate`) to control step size and order, so it controls its own step and
  `usesAdaptiveStep` is false. The step-control coefficients are `__constant__`s in
  `SeulexConstants`; the tolerances come from `IntegratorControls`, i.e. from the settings file.
- **`kodes::Euler`** (`Integrator/IntegrationMethods/Euler/…`) — one explicit step and its error, to
  be accepted or rejected by `adaptiveStep`. The simplest thing the machinery can be checked
  against, not something to point at a stiff mechanism.
- **`kodes::Integrator`** (`Integrator/Integrator.cuh`) — drives the solve. It owns the
  `adaptive_solve` kernel: the grid-stride loop over the batch and the step-count loop that walks
  one system from local time 0 to the target end time. `solve(deltaT, realBatchSize)` launches it
  for one batch, `setDeltaT` seeds the step state of every thread slot, and `resetStep()` reseeds a
  slot for its next system from the trial step the previous one ended with, so each system starts
  warm from its predecessor in that slot. It is not a template: the ODE system, the method, the
  resources and the balancer are all device objects it dispatches on, so there is one instantiation
  of the kernel however the run was configured.

## Typical run

See `examples/integrators/GRIMECH/seulex5.cu` and the `seulex5.json` beside it. The order matters:
plan first (it needs the free VRAM before anything has been allocated), then size every allocation
from the plan.

```cpp
// 1) the method and the balancer are names, read from a file
kodes::Settings settings("seulex5.json");

const char* method   = settings.method().c_str();     // "seulex"
const char* balancer = settings.balancer().c_str();   // "stiffness"

// 2) ask the device how many systems it can run at once and how big a batch
//    fits. The two names carry what their classes will cost; pyJac's own
//    per-thread scratch is owned by neither, so it goes in the extra.
kodes::LaunchConfig config = kodes::planLaunch
(
    ensembleSize, host_res.systemSize(), host_res.parameterSize(),
    method, balancer,
    required_mechanism_size(),
    settings.launchRequest()              // "best", "half", or explicit sizes
);

// 3) per-thread scratch is padded to the resident threads, not to the batch
initialize_gpu_memory(config.scratchSize, &h_mem, &d_mem);

// 4) build what the names selected. Each handle owns a device object and the
//    host stub holding its buffers, and returns both when it goes out of scope.
auto resources = kodes::newResources(method, config.batchSize, config.scratchSize,
                                     systemSize, parameterSize);
auto integrationMethod = kodes::newMethod(method, config.batchSize, config.scratchSize,
                                          systemSize, parameterSize);
auto balancing = kodes::newBalancer(balancer, config.batchSize, config.scratchSize,
                                    systemSize, parameterSize);

// 5) solve batch by batch, each one ordered before it is integrated
kodes::Operator op(&host_res, resources.host());

kodes::Integrator solver
(
    ode, resources.device(), integrationMethod.device(), config, settings.controls()
);

solver.setBalancer(balancing.device(), balancing.host());

solver.setDeltaT(settings.initialTimeStep());

for (label i = 0; i < config.numOfBatches(ensembleSize); i++)
{
    op.cpyHostToDevice(i);
    solver.solve(settings.endTime(), op.getRealBatchSize(i));
    op.cpyDeviceToHost(i);
}
```

A caller with its own settings to read — the OpenFOAM chemistry model — skips step 1 and passes the
same names and numbers from wherever it got them.

## Building

```sh
git submodule update --init external/rapidjson      # only for kodes::Settings

cd examples/integrators/GRIMECH && cmake -B build && cmake --build build
./build/seulex5                                     # reads build/seulex5.json
```

`cmake/kodes.cmake` holds the library's source and include lists, so an example's `CMakeLists.txt`
is a handful of lines. Every method and every balancer is compiled in, whichever one a run names:
the choice is made when the program starts, so all of them have to be in the binary.
`CUDA_SEPARABLE_COMPILATION` is not optional — a device-only virtual is called from a kernel in
another translation unit.
