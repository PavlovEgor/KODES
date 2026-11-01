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

The simplest ones: the streams of each tutorial's initial and inlet conditions - unburnt
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

## Not here yet

The tools only make and check states. Still to come, each reading the files above:

- a KODES runner: the states repeated to an ensemble, integrated with a chosen method,
  the final states and the wall clock written out;
- a Cantera reference: the same states in `IdealGasConstPressureReactor`, for the final
  states and CVODE's wall clock;
- the OpenFOAM side: the tutorial's own chemistry, timed through `kodesCase.py`.

The tools need Python 3 and nothing else.
