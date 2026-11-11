# Tests

Sets of chemical states, one per flame of kodes4foam's tutorials and mechanism, for
three uses:

- **benchmark against OpenFOAM**: the same states, the same time step, the same
  tolerances as OpenFOAM's own chemistry integrates them with in the tutorial, so a
  difference in wall clock is a difference in the integrator;
- **benchmark against Cantera** for `bdf`: Cantera's reactors integrate with CVODE, a BDF
  code, so it is the method compared like for like;
- **validation against Cantera**: the state each method ends with, against Cantera's.

Every state is the same problem: an adiabatic reactor at constant pressure, integrated
over the case's `deltaT` - what OpenFOAM's chemistry does in a cell over one flow step,
what pyJac's right hand side describes, and Cantera's `IdealGasConstPressureReactor`.

## Layout

```
tests/
  cases/<case>/               one per tutorial of kodes4foam
    case.json                 where it comes from, deltaT, tolerances, the streams
    <mechanism>.csv           the states, one file per mechanism the case runs with
  benchmark/                  the KODES runner: a case through a prebuilt libkodes_<mechanism>.so
  tools/
    states.py                 the state file format, and the species of a mechanism.cuh
    make_initial_states.py    the simplest sets, from case.json
    states_from_openfoam.py   a set from a time directory of an OpenFOAM run
    check_states.py           format, bounds, sum of Y, species against the mechanism
```

| Case | Flame | Mechanisms | deltaT | relTol |
|---|---|---|---|---|
| 2D_planar_flame | premixed H2/air front, phi = 0.5 | h2o2, grimech, C1_C3 | 2e-6 | 1e-3 |
| 2D_jet_flame | non-premixed H2 jet into air | h2o2, grimech, C1_C3 | 5e-6 | 1e-3 |
| 3D_nonPremixed_burner | coaxial H2/air injector, PaSR | h2o2, grimech, C1_C3 | 1e-5 | 1e-3 |
| 3D_premixed_burner | premixed H2/air, phi = 0.6, PaSR | h2o2, grimech, C1_C3 | 1e-5 | 1e-3 |
| nonUniformHeating2D | homogeneous H2/air autoignition at 1500 K | h2o2, grimech, C1_C3 | 1e-6 | 1e-2 |
| counterFlowFlame2D | CH4/air counterflow diffusion flame | grimech, C1_C3 | 1e-6 | 1e-1 |
| 2D_flat_burner | rich CH4/air flat burner flame, phi = 1.15, developed | grimech | 1.2e-5 | 1e-3 |

absTol is 1e-12 everywhere. Each `case.json` says which file of the tutorial its numbers
are from: `deltaT` is the tutorial's `maxDeltaT`, the step a developed run settles at,
except for `counterFlowFlame2D`, whose `maxDeltaT` is 1 and not a step it ever takes.

## The mechanisms

kodes4foam's, in `tutorials/mechanisms/<mechanism>/` (the pyJac sources under
`src/kodesChemistryModel/pyJacSource/`): `h2o2` with N2, 10 species; `grimech`, 53;
`C1_C3_HT_NOX_159_2459`, 159. Not KODES's own `data/mechanisms/h2o2`, which has no N2 and
so cannot hold air.

## State files

```
# optional comment lines
label,T,p,H2,O2,N2,H2O
unburnt,320.0,101325.0,0.01446221408,0.2295446063,0.7559931796,0.0
```

T in K, p in Pa, then mass fractions **by specie name**. A specie of the mechanism without
a column is zero; a column naming a specie the mechanism does not have is an error. By
name, because the same file has to reach pyJac's order, OpenFOAM's species table and
Cantera's - and pyJac's order is not even the mechanism's, it moves the eliminated specie
last. `tools/states.py` reads and writes the format and reads a `mechanism.cuh`'s species.

## The states there are now

`2D_flat_burner` is a developed flame already: 2048 of the 86 688 cells of its run at
t = 0.031 s (the 0.05 mm mesh, standard chemistry, the relaxational oscillation going),
by ranks, so most of them are the cold mixture, air and co-flow around the flame. Its
`deltaT` is the step that run took there. It has no streams, and `make_initial_states.py`
leaves it alone.

The others are the simplest ones: the streams of each tutorial's initial and inlet conditions - unburnt
mixture, ignition kernel, fuel, air - and mixtures of pairs of them at 25, 50 and 75 % by
mass (`fuel0.25+air` is a quarter fuel). Temperature is mixed linearly too, which ignores
that the two cp differ: these states are below, at and above ignition, and that is all
they are for. Between 1 and 12 per case.

```bash
tests/tools/make_initial_states.py                 # rewrite them all from case.json
tests/tools/check_states.py --mechanisms ../../tutorials/mechanisms
```

## Replacing them with a developed flame

Run the tutorial with OpenFOAM's chemistry until the flame is developed, then take the
states of one time directory:

```bash
cd kodes4foam/tutorials
./Allrun grimech                                   # 2D_planar_flame, standard chemistry
../external/KODES/tests/tools/states_from_openfoam.py \
    2D_planar_flame 0.0005 mechanisms/grimech/mechanism.cuh \
    ../external/KODES/tests/cases/2D_planar_flame/grimech.csv
../external/KODES/tests/tools/check_states.py --mechanisms mechanisms
```

It reads ASCII and binary fields of a reconstructed case (`reconstructPar` first, for a
parallel run), and a specie without a field of its own from `Ydefault`. A mesh is too many
states to keep in git, so it keeps `--max-states` cells (2048 by default):

- `--sample ranks`, the default: evenly spaced in the order of temperature, so the same
  distribution as the mesh - what a benchmark against the whole run should see;
- `--sample temperature`: evenly over the range of temperature - what a validation should
  see, without cold cells being most of it.

## Benchmark

`benchmark/` integrates a case with KODES: the states repeated to an ensemble, over the
case's `deltaT` with its tolerances, one warm-up and then `--repeats` timed `solve()`s,
each from the same states and the same first trial step. It is a plain C++ program that
opens a prebuilt `libkodes_<mechanism>.so` through `library_api.h`, as kodes4foam does, so
a `solve()` is a whole step of the chemistry, host to device and back included, and the
species of the file reach the state by name.

```bash
cmake -B build && cmake --build build -j                  # the libraries, at the top
cd tests/benchmark && cmake -B build && cmake --build build
LD_LIBRARY_PATH=../../build ./build/benchmark                # 2D_flat_burner, grimech, seulex
LD_LIBRARY_PATH=../../build ./build/benchmark --balancer none  # the same, unbalanced
LD_LIBRARY_PATH=../../build ./build/benchmark --method bdf --ensemble 86688 \
    --output bdf.csv                                         # and the final states
./build/benchmark --help
```

By default the 2048 states of `2D_flat_burner` become 131072 systems, the copies shuffled
over the ensemble with a fixed seed. The file is sorted by temperature, so taken in its
own order (`--order as-is`) every warp would get neighbours in temperature - a batch
already balanced, which is not what a mesh hands over, and against which `--balancer` would
have nothing left to do. `--output` writes the final state
of each state of the file in the same format, for a validation to read.

## Not here yet

Still to come, each reading the files above:

- a Cantera reference: the same states in `IdealGasConstPressureReactor`, for the final
  states and CVODE's wall clock;
- the OpenFOAM side: the tutorial's own chemistry, timed through `kodesCase.py`.

The tools need Python 3 and nothing else.
