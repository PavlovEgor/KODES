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
matches how CFD scales (more cells, not bigger mechanisms). Every thread is on its own: its own step
size, its own Jacobian, its own factorization, nothing to wait for.

That is the 1.x line, and on purpose the simplest form of the idea. 0.20 to 0.23 integrated a whole
block in lockstep, with one Jacobian and one LU factorization for all its threads, and sorted the
batch with a `Balancer` so that a block got cells alike; that line lives on the `0.23.0` branch.
1.0.0 keeps everything else 0.23 brought — the device object pattern, the type tables, the ODE
system as a table row, the plan, the prebuilt per-mechanism libraries — and returns the solve to one
system per thread.

Which integration method runs and every tolerance are read from a JSON settings file when the
program starts — see *Typical run* below. How a class whose virtual functions only exist on the GPU
can be selected by name at run time is a pattern used everywhere in the library, and
**[`DeviceObjects_explained.md`](DeviceObjects_explained.md)** explains it in full.

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
  `systemSize^2`, the extrapolation table, the pivot indices and row scales, ...). One slot per
  *resident thread*, stride `GRID_DIM` (the number of threads actually launched), addressed with
  `INDEXVEC`/`INDEXMAT`. Allocating more of these than the device can keep resident at once is pure
  waste, and for a large mechanism they are what actually exhausts the VRAM. Nothing here outlives
  the system that is currently in the slot except the trial step it ended with, which seeds the
  next system the slot picks up.

A thread walks its share of the batch in a grid-stride loop, in the order it was copied in:
`DeviceResources::loadSystem(system)` pulls one system into the thread's scratch slot
(`currentVector`/`currentParameters`), the integrator works entirely in scratch space (which is
also the layout pyJac's generated `dydt`/`eval_jacob` expect — its `INDEX` macro is identical to
`INDEXVEC`), and `storeSystem(system)` writes the result back. Both directions stay coalesced:
consecutive threads touch consecutive systems. A system at or below `Treact` is skipped.

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

  The named shares live in the `kodes::kDeviceShares` table — add a line there to add a name. A
  share scales both the concurrency and the memory budget; sizes set by hand are only checked
  against the free VRAM. `KODES_MEMORY_HEADROOM` caps how much of the free VRAM any plan may claim.

  `threadsPerBlock` is an occupancy knob and nothing else: every thread integrates on its own, so
  the block size changes how the grid is cut up, not what any system computes.
- **`kodes::planLaunch(...)`** (declared in `Integrator.cuh`) — resolves a request against the
  device. `maxConcurrentThreads()` asks `cudaOccupancyMaxActiveBlocksPerMultiprocessor` how many
  blocks of the *actual solve kernel* fit on an SM — that depends on the kernel's register and
  shared-memory footprint, so it can only be known at run time. That, `cudaMemGetInfo` and what the
  chosen classes cost go into `makePlan()`, which holds all the sizing arithmetic and no CUDA call:
  cap the grid at what the budget affords, then spend the rest on the batch. The costs come from the
  method's and the ODE system's *names* — every type table entry answers
  `scratchBytesPerThread`/`scratchBytesPerBlock`/`stateBytesPerSystem`, so the plan can be made
  before either object exists, which it has to be, since it is the plan that fixes their sizes. The
  system's entry carries what the mechanism's own workspace costs per thread —
  `required_mechanism_size()` for pyJac. The extra is what is left for memory owned outside both,
  and nothing in the library goes there any more. Every per block cost is zero in this version, so
  `blockScratchInShared` always comes out false; the slot stays in the tables and in the plan.

  The order inside `planLaunch` matters: a device call stack the caller asks for is sized first of
  all, because the driver reserves it out of the same VRAM the plan is about to divide up — and by
  default none is asked for: 0 leaves the driver's own value alone (see
  `KODES_DEVICE_STACK_BYTES`).
- **`basic_linalg.cuh`/`.cu`** — device-side building blocks used by the integrators:
  `LUDecompose`/`LUBacksubstitute` (in-place LU factorization with scaled partial pivoting, Crout's
  form as in OpenFOAM, and back-substitution, used to solve the linear systems in each implicit
  step), plus small helpers (`copyVec`, `sumVec`, `sqr`, `clamp`, `swap`, `normalizeError`). The
  row scales of the pivoting are `systemSize` scalars of the thread's scratch slot, passed in as
  `vv`. In 0.19 they were a local `scalar vv[128]`, which a mechanism of more than 128 species ran
  past in silence, into the thread's stack.
- **`kodes::Config`** (`Settings/Config.cuh`/`.cu`) — a small RapidJSON-backed JSON config-file reader
  (`getDouble`/`getInt`/`getString`/`getBool`/`hasKey` with defaults), independent of the ODE
  machinery. Keys are dotted paths, so `"controls.absTol"` reaches into the nested object. Requires
  the `external/rapidjson` submodule.
- **`kodes::Settings`** (`Settings/Settings.cuh`/`.cu`) — the whole of a run in one JSON file: the
  name of the method and of the ODE system, the `LaunchConfig` to resolve and the
  `IntegratorControls`. Every entry has a default, and both names are looked up in their table in
  the constructor, so a typo fails before anything has been allocated on the device. See
  `examples/reactors/grimech.json`. It is a source list of its own in
  `cmake/kodes.cmake` — the only part of the library needing rapidjson — because a caller with its
  own settings to read (the OpenFOAM chemistry model reads an OpenFOAM dictionary) passes the same
  names and numbers by hand and never links it.

### Device objects — one pattern for every class chosen at run time

Which method integrates and which mechanism is integrated are decisions made when the program
starts, and both are made the same way. The mechanics are
worth reading once — **`DeviceObjects_explained.md`** covers them in full, with the reasons — but
in short:

- **`Factory/device_object.cuh`** — how such an object is built. It lives in device memory, because
  a virtual call from inside a kernel needs a device vtable, and its buffers are allocated by the
  host into a *host stub* of the same type: allocate on the stub, byte-copy it to the device,
  placement-new on top of the copy. That order is why the constructor of such a class must set
  value members only — it runs after the copy and would overwrite the addresses it brought. A
  concrete class keeps a three-point contract (that constructor, a non-virtual
  `allocate()`/`deallocate()` pair, a placement `operator new`) and gets its four host statics —
  `create`/`destroy`/`createStub`/`destroyStub` — from one line of `KODES_DEFINE_DEVICE_OBJECT` in
  its `.cu`. The host half of the pattern is static polymorphism (the factory is a template, so
  `stub->allocate()` resolves to the concrete one); only the device half uses a vtable.
- **`Factory/type_table.cuh`** — how one of them is chosen by name. A `TypeEntry` is a concrete class
  reduced to plain function pointers, so the dispatch needs no host vtable at all: a caller nvcc did
  not compile can still select and own a class whose virtuals only exist on the device. That is the
  reason for `createStub` — `step()` and `derivatives()` are device-only virtuals, and the host vtable of a
  class with one can only be emitted by a compiler that invents a host stub for it, which nvcc does
  and `nvc++ -cuda` does not. Keeping construction inside the library's own `.cu` leaves callers
  holding nothing but a pointer. `Handle<Base>` owns the device object and its stub together and
  returns both to the class that made them.

The tables are `Integrator/IntegrationMethods/method_table.cu` and `ODESystem/system_table.cu`; a method's entry names the `DeviceResources` subclass holding its
scratch, since the two can only be chosen together.
- **`kodes::mpiSelectDevice`** (`mpi_select_device.cuh`/`.cu`) — optional MPI device-binding helper for
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
  `derivatives(x, param, y, dydx)` and `jacobian(x, param, y, dfdy)`, both `__device__`. The
  Jacobian is `df/dy` alone: the systems this library integrates are autonomous, so the `df/dx` it
  used to take alongside it was a vector every caller allocated per thread, passed in, and never
  read - pyJac never wrote it either. A device object on the same terms as the others: the
  five-size constructor, an `allocate()`/`deallocate()` pair, the four host statics from
  `KODES_DEFINE_DEVICE_OBJECT`, and a row in a type table. It used to be the exception — its
  constructor took the mechanism's scratch instead of the sizes, so it hand-wrote its own
  construction kernels and answered to nobody for what it cost. The scratch is padded to
  `scratchSize`, which *is* one of the five sizes, so there was nothing in the way.
- **`kodes::PyJacSystem`** (`ODESystem/PyJacSystem/`) — any mechanism generated by pyJac.
  `derivatives`/`jacobian` forward into the generated `dydt`/`eval_jacob` against a per-thread
  `mechanism_memory` scratch block (concentrations, rates, Jacobian workspace — distinct from the
  state buffers below). That block is what its `allocate()` allocates, by calling
  `initialize_gpu_memory` on the host stub before the stub is copied to the device; its
  `scratchBytesPerThread` is `required_mechanism_size()`, so the plan sizes the run for it without
  being told. Compiled for constant pressure (`CONP`); state is `[T, Y_0..Y_{NSP-2}]` with the
  mechanism's designated last species recovered implicitly from mass conservation, pressure passed
  as the system parameter.
- **`ODESystem/system_table.cuh`/`.cu`** — the second type table. It has one row, and will until
  there is a second kind of system, because pyJac fixes the mechanism when the library is compiled.
  The row still earns its place: it is what lets `planLaunch` ask the cost before there is an
  object, and `Handle<ODESystem>` own the object and its stub the way it owns the others. It
  also keeps pyJac's generated headers out of a caller's translation units — `PyJacSystem.cu` and
  `system_table.cu` are the only two files in the library that include them.

  Which mechanism it is is a compile-time choice, unlike everything else in the library: pyJac
  generates C for one mechanism, and `NSP` is a macro of its `mechanism.cuh`. It is also the only
  such choice, and it is made entirely outside `src/` — no file there names a mechanism. The
  mechanisms are data, and live under `data/mechanisms/<name>/`: the generated code in `out/`,
  beside the input files it came from. Two are in the tree, GRI-Mech 3.0 (53 species, 325
  reactions) and a smaller H2/O2 mechanism (9 species, 28 reactions), and
  `kodes_pyjac_mechanism(grimech ...)` in `cmake/kodes.cmake` selects one by directory name.
  Dropping a new pyJac output in beside them is enough to make it selectable.

### `Resources` — state storage, host and device

This is the one family with no table and no concrete classes of its own, and both are deliberate.

A method and the resources holding its scratch can only be chosen *together*, because `Seulex::step()`
casts the base pointer down to `SeulexDeviceResources*` and `IntegrationMethod::adaptiveStep()` to
`AdaptiveDeviceResources*`. So the row that selects a subclass is a *field* of the method's row in
`method_table.cu` rather than a table of its own — one with a name of its own would make
`{"method": "euler", "resources": "seulexResources"}` a legal settings file that fails inside the
kernel.

The tree says the same thing: `SeulexDeviceResources` lives in `IntegrationMethods/Seulex/` beside
the method it exists for, and `EulerDeviceResources` in `IntegrationMethods/Euler/`. What stays
here is only what any method can use — the abstract part of the family (`Resources`,
`DeviceResources`, `AdaptiveDeviceResources`), the `StepState` mixin, and the two host-side classes
that are not device objects at all (`HostResources`, `Operator`). That also keeps the dependency
one-way: `IntegrationMethods` includes from `Resources`, never the reverse. It used to be the
reverse as well, since `SeulexDeviceResources` needs the Seulex tableau in `seulex_constants.cuh`.

For the same reason `DeviceResources` and `AdaptiveDeviceResources` carry no
`KODES_DECLARE_DEVICE_OBJECT`: neither is ever a row, so only the leaves declare the four statics —
the same division as in `ODESystem`, whose base declares nothing but the placement
`operator new` either.

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
  integrating right now, sized with `scratchSize`, and the `StepState` arrays - the trial step, the
  local time, the direction and the first/last/reject flags of that system.
  `loadSystem(system)`/`storeSystem(system)` move one system between the two. A device object like
  every other, so its buffers are allocated by its own `allocate()` into a host stub that `Operator`
  then reads to find them. The statics `stateBytesPerSystem`/`scratchBytesPerThread`/
  `scratchBytesPerBlock` report the per-system, per-thread and per-block cost to `planLaunch` (the
  last is zero); a subclass extends whichever it adds to.
- **`kodes::SeulexDeviceResources`** (`Integrator/IntegrationMethods/Seulex/`) — extends
  `DeviceResources` with the extra scratch the Seulex integrator needs, all of it per thread and
  sized with `scratchSize`: the polynomial extrapolation table, the Jacobian `dfdy` and the LU work
  matrix `a`, the pivot indices and row scales of the factorization, the order-indexed step
  estimates, the target order and the contraction rate carried from one step to the next, and the
  vectors the outer step and the inner `seul` sub-stepping run through. Its `allocate()` also fills
  the `__constant__` step sequence and extrapolation coefficients.
- **`kodes::AdaptiveDeviceResources`** (`Resources/`) / **`kodes::EulerDeviceResources`**
  (`Integrator/IntegrationMethods/Euler/`) — the same for a method that takes trial steps: the two
  vectors `IntegrationMethod::adaptiveStep` needs to hold a trial state and the derivative it
  started from, plus the error vector Euler writes. The intermediate stays in `Resources/` because
  any trial-step method uses it; only the leaf belongs to one method.
- **`kodes::Operator`** (`Resources/Operator.cuh`) — copies state between a `HostResources` and a
  `DeviceResources` (or subclass): `cpyHostToDevice(batchIndex)`/`cpyDeviceToHost(batchIndex)`
  transfer every component and parameter of one batch, translating between the host's per-component
  pointers and the device's flat state-space layout. `getRealBatchSize(batchIndex)` gives the size
  of that batch — the last one is normally shorter than `batchSize`.

### `Integrators` — advancing the ODEs

- **`kodes::IntegrationMethod`** (`Integrator/IntegrationMethods/IntegrationMethod.cuh`) — abstract
  base for one numerical method, a device object like the ODE system: `__device__ virtual scalar
  step(ode, resources, controls)` advances the system the calling thread holds in its scratch slot,
  and which subclass runs is a name in `method_table`. `usesAdaptiveStep()` says whether that step
  is a *trial* step, in which case the base's own `adaptiveStep()` — the step-size controller —
  retries it smaller until the error is inside the tolerance, and then grows the next one. A method
  owns no storage: the `DeviceResources` subclass named in the same table entry holds all of it,
  which is what makes it safe for `step()` to cast the base pointer down to it.
- **`kodes::Seulex`** (`Integrator/IntegrationMethods/Seulex/`) — a GPU port of the
  semi-implicit Bulirsch-Stoer extrapolation method (the same algorithm as OpenFOAM's own `seulex`
  ODE solver). `step()` uses `LUDecompose`/`LUBacksubstitute` for the implicit linear solves and
  polynomial extrapolation (`extrapolate`) to control step size and order, so it controls its own
  step and `usesAdaptiveStep` is false. The step-control coefficients are `__constant__`s in
  `seulex_constants.cuh`; the tolerances come from `IntegratorControls`, i.e. from the settings
  file.
- **`kodes::Euler`** (`Integrator/IntegrationMethods/Euler/`) — one explicit step and its error,
  to be accepted or rejected by `adaptiveStep`. The simplest thing the machinery can be checked
  against, not something to point at a stiff mechanism.
- **`kodes::Integrator`** (`Integrator/Integrator.cuh`) — drives the solve. It owns the
  `solveKernel` kernel: the grid-stride loop over the batch and the step-count loop that walks one
  system from local time 0 to the target end time. `solve(deltaT, realBatchSize)` launches it for
  one batch and `setDeltaT` seeds the trial step of every scratch slot; a slot carries the step it
  ended one system with into the next, so a system starts warm from its predecessor. It is not a
  template: the ODE system, the method and the resources are all device objects it dispatches on, so
  there is one instantiation of the kernel however the run was configured.

## Typical run

See `examples/reactors/reactors.cu` and the `grimech.json` beside it. The order matters:
plan first (it needs the free VRAM before anything has been allocated), then size every allocation
from the plan.

```cpp
// 1) the method and the system are names, read from a file
kodes::Settings settings("grimech.json");

const char* method   = settings.method().c_str();     // "seulex"
const char* odeSystem = settings.system().c_str();    // "pyJac"

// 2) ask the device how many systems it can run at once and how big a batch
//    fits. The two names carry what their classes will cost, pyJac's own
//    per-thread workspace included - there is nothing left to declare by hand.
kodes::LaunchConfig config = kodes::planLaunch
(
    ensembleSize, host_res.systemSize(), host_res.parameterSize(),
    method, odeSystem,
    settings.launchRequest()              // "best", "half", or explicit sizes
);

// 3) build what the names selected. Each handle owns a device object and the
//    host stub holding its buffers, and returns both when it goes out of scope.
//
//    pyJac's scratch is one of those buffers. It is padded to the resident
//    threads, not to the batch, and newSystem() is handed config.scratchSize
//    for it.
auto resources = kodes::newResources(method, config.batchSize, config.scratchSize,
                                     config.blocks, systemSize, parameterSize);
auto integrationMethod = kodes::newMethod(method, config.batchSize, config.scratchSize,
                                          config.blocks, systemSize, parameterSize);
auto ode = kodes::newSystem(odeSystem, config.batchSize, config.scratchSize,
                            config.blocks, systemSize, parameterSize);

// 4) solve batch by batch
kodes::Operator op(&host_res, resources.host());

kodes::Integrator solver
(
    ode.device(), resources.device(), integrationMethod.device(), config, settings.controls()
);

solver.setDeltaT(settings.initialTimeStep());

for (label i = 0; i < config.numOfBatches(ensembleSize); i++)
{
    op.cpyHostToDevice(i);
    solver.solve(settings.endTime(), op.getRealBatchSize(i));
    op.cpyDeviceToHost(i);
}
```

A caller with its own settings to read skips step 1 and passes the same names and numbers from
wherever it got them. A caller with no CUDA compiler writes none of this: it opens a prebuilt library,
which does all four steps behind one call — see *As a prebuilt library* below.

## Building

```sh
git submodule update --init external/rapidjson      # only for kodes::Settings

cd examples/reactors && cmake -B build && cmake --build build

./build/reactors_grimech            # GRI-Mech 3.0, reads build/grimech.json
./build/reactors_h2o2               # H2/O2,        reads build/h2o2.json
```

Every method is compiled in, whichever one a run names: the choice is made when the program
starts, so all of them have to be in the binary. `CUDA_SEPARABLE_COMPILATION` is not
optional — a device-only virtual is called from a kernel in another translation unit.

## Building against KODES

To compile KODES into a CUDA program of your own, take the source list and the include list from
**`cmake/kodes.cmake`**:

```cmake
include(${CMAKE_CURRENT_SOURCE_DIR}/<path to KODES>/cmake/kodes.cmake)

kodes_pyjac_mechanism(grimech MECH_SOURCES MECH_INCLUDE_DIRS)

add_executable(myApp myApp.cu ${KODES_SOURCES} ${MECH_SOURCES})
target_include_directories(myApp PRIVATE ${KODES_INCLUDE_DIRS} ${MECH_INCLUDE_DIRS})
set_target_properties(myApp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
```

`${KODES_SETTINGS_SOURCES}` adds `kodes::Settings` (and needs the rapidjson submodule);
`${KODES_MPI_SOURCES}` adds `kodes::mpiSelectDevice`. Neither is in the core list, since a consumer
reading its settings from somewhere else on one GPU wants neither.

A program with no CUDA compiler of its own — the OpenFOAM chemistry model, since 0.23.0 — does not
compile KODES at all: it opens a prebuilt library, which is the next section.

## As a prebuilt library, one per mechanism

```sh
cmake -B build                      # every mechanism of data/mechanisms/
cmake --build build -j
cmake --install build               # into $FOAM_USER_LIBBIN when OpenFOAM is sourced, else lib/
```

builds `libkodes_grimech.so` and `libkodes_h2o2.so`: the whole library and one mechanism each,
behind one device link. A program uses one by opening it with `dlopen()` and including nothing of
KODES but **`src/Library/library_api.h`** — plain C++, no CUDA type in it, so the program needs no
CUDA compiler, no KODES headers and no `-rdc`. That is what lets an OpenFOAM chemistry model pick its
mechanism in a case dictionary instead of in a build. `examples/library/` is the reactors example
done this way, a CMake project with no CUDA in it:

```sh
cd examples/library && cmake -B build && cmake --build build
LD_LIBRARY_PATH=../../build ./build/library h2o2
```

**Why one library per mechanism, and not one `libkodes` for all of them.** The device code is
compiled `-rdc=true`, and the solve kernel calls the method and the mechanism through
device-only virtuals, which are resolved at *device* link time. CUDA does not link device code
across shared libraries: a `.so` is device-linked when it is built, and whatever device code it did
not see then is out of its reach for good. So the mechanism — `dydt()`, `eval_jacob()`, everything
pyJac generated — has to be inside the same device link as the kernel that calls it, and a library
without one in it would leave the consumer the device link to do, with nvcc, and pyJac's sources to
compile, with nvcc again. Built per mechanism, the device link is finished and nothing about CUDA is
left on the consumer's side.

**What crosses into it.** Two `extern "C"` symbols, looked up with `dlsym()` by the names
`library_api.h` gives them: `kodesLibraryApiVersion()`, to be compared with `kLibraryApiVersion`
before anything else, and `kodesLibrary()`, which hands out the `kodes::Library` of the mechanism.
That one answers what the mechanism is — its name, the KODES version it was built from, its species
in pyJac's order, so a consumer maps them onto its own without reading `mechanism.cuh` — and makes a
`kodes::Solver` from a `kodes::SolverSettings`: the method, the device share or the sizes by hand,
the tolerances. Its `balancer` field and `Solver::balancer()` are the interface of 0.23, kept as
they were so that a 0.23 caller works unchanged: the field is not read, and `balancer()` answers
`"none"`. The solver plans, builds and integrates exactly as *Typical run*
above; `solve(deltaT, state, parameters)` takes the ensemble in the `HostResources` layout and
integrates it in place, batch by batch. The types in the interface are `double` and `int`, not
`scalar` and `label`, so a caller with types of its own of those names converts at the call.

**How it is built.** `kodes_library(<name>)` in `cmake/kodes.cmake`, which the top-level
`CMakeLists.txt` calls for every entry of `KODES_MECHANISMS`:

- an entry is a directory of `data/mechanisms/`, or the path of any directory holding pyJac's `out/`
  — a consumer's own mechanism, built without being copied in here:
  `-DKODES_MECHANISMS="/path/to/pyJacSource/C1_C3_HT_NOX_159_2459;grimech"`;
- the species list is read out of the "Species Indexes" comment of `mechanism.cuh` when the library
  is configured, since pyJac writes it nowhere else, and checked against `NSP` when it is compiled;
- the CUDA runtime is linked in statically, so the machine running it needs the driver and nothing
  else of CUDA; and every symbol but the two entry points is hidden, so two of these in one process
  cannot see each other;
- `CMAKE_CUDA_ARCHITECTURES` defaults to `120` as in the examples; `-DKODES_LIBRARY_DIR` changes
  where `cmake --install` puts the libraries.

## Layout

One shape for every family chosen at run time: the abstract base and the type table sit at the root
of the family's directory, and each concrete class gets a directory of its own.

```
src/
  basic_types.cuh         scalar, label, the indexing macros
  basic_linalg.{cuh,cu}   LU factorisation and the small device helpers
  mpi_select_device.{cuh,cu}
  Factory/                device_object.cuh, type_table.cuh — the pattern above
  ODESystem/              ODESystem.cuh, system_table — no mechanism is named
                          anywhere in src/
    PyJacSystem/
  Resources/              only what any method can use: Resources,
                          DeviceResources, AdaptiveDeviceResources, StepState,
                          and the host side HostResources, Operator. No table
                          and no concrete classes — see the Resources section
                          above for why
  Integrator/             Integrator, LaunchConfig, IntegratorControls
    IntegrationMethods/   IntegrationMethod, method_table
      Seulex/             Seulex, SeulexDeviceResources, seulex_constants —
                          one directory per row of method_table, because a
                          method and its storage are one choice
      Euler/              Euler, EulerDeviceResources
  Settings/               Config (JSON reader), Settings
  Library/                library_api.h — the CUDA-free interface of a prebuilt
                          libkodes_<mechanism>.so — and library_api.cu behind it

CMakeLists.txt            the prebuilt libraries, one per mechanism
cmake/
  kodes.cmake             the source lists, kodes_pyjac_mechanism, kodes_library

data/
  mechanisms/
    grimech/              GRI-Mech 3.0: the pyJac output in out/, beside the
    h2o2/                 .dat/.inp files it was generated from

examples/
  reactors/               an ensemble of reactors — one program, one target per
                          mechanism, everything else from grimech.json / h2o2.json
  library/                the reactors again, through a prebuilt library and
                          with no CUDA in the program
  config/                 everything kodes::Config can read
  mpi_device_select/      binding one rank per GPU
```

Includes stay flat within a family — `#include "IntegrationMethod.cuh"` — because every family
root is on the include path. A concrete class is reached through its directory instead:
`#include "Seulex/Seulex.cuh"`. That is deliberate: it keeps `KODES_INCLUDE_DIRS` at
one entry per family rather than one per class, so adding a class never touches the include list,
and a consumer's build cannot break because a new `-I` was forgotten. It also makes the include
line say where the class lives.

A file starting with a capital letter holds the class it is named after; a `snake_case` one holds
free functions, a table or constants. That and the rest of the conventions are in
**[`CodingStyle.md`](CodingStyle.md)**.
