#!/usr/bin/env python3
"""
Эталонный расчёт того же состояния через Cantera (gri30), для сравнения с
выводом pyjac_probe (см. pyjac_probe.cu в этой же папке).

Cantera всегда возвращает величины в своей внутренней системе СИ-кмоль
(kmol/m3, Pa, kg, J), независимо от единиц, в которых был написан исходный
.cti/.yaml файл механизма - это и делает её удобным независимым эталоном для
проверки, не отвалились ли единицы измерения при генерации кода pyJac.

Запуск:
    python3 cantera_reference.py [state.txt] > cantera_output.txt

Требует: pip install cantera
"""
import sys
import numpy as np
import cantera as ct

# Порядок видов pyJac для GRIMech3.0 (см. mechanism.cuh -> "Species Indexes").
PYJAC_SPECIES_ORDER = [
    "H2", "H", "O", "O2", "OH", "H2O", "HO2", "H2O2", "C", "CH", "CH2", "CH2(S)",
    "CH3", "CH4", "CO", "CO2", "HCO", "CH2O", "CH2OH", "CH3O", "CH3OH", "C2H",
    "C2H2", "C2H3", "C2H4", "C2H5", "C2H6", "HCCO", "CH2CO", "HCCOH", "N", "NH",
    "NH2", "NH3", "NNH", "NO", "NO2", "N2O", "HNO", "CN", "HCN", "H2CN", "HCNN",
    "HCNO", "HOCN", "HNCO", "NCO", "AR", "C3H7", "C3H8", "CH2CHO", "CH3CHO", "N2",
]


def read_state(path):
    rows = []
    with open(path) as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if line:
                rows.append([float(x) for x in line.split()])
            if len(rows) == 2:
                break
    (T, P), X = rows
    return T, P, np.array(X, dtype=float)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "state.txt"
    T, P, X = read_state(path)

    gas = ct.Solution("gri30.yaml")

    if list(gas.species_names) != PYJAC_SPECIES_ORDER:
        print("!!! ВНИМАНИЕ: порядок видов в Cantera gri30 отличается от "
              "PYJAC_SPECIES_ORDER !!!")
        print("cantera:", list(gas.species_names))
        print("pyjac  :", PYJAC_SPECIES_ORDER)
        print("Перекладываю мольные доли по именам видов, чтобы сравнение "
              "осталось корректным.\n")
        by_name = dict(zip(PYJAC_SPECIES_ORDER, X))
        X = np.array([by_name[name] for name in gas.species_names])

    X = X / X.sum()
    gas.TPX = T, P, X

    rho = gas.density

    print("=== Cantera reference state ===")
    print(f"T0  = {T:.10f} K")
    print(f"P0  = {P:.10f} Pa")
    print(f"rho (Cantera, ideal gas EOS) = {rho:.15e} kg/m3\n")

    print("mole fractions (ненулевые):")
    for name, x in zip(gas.species_names, gas.X):
        if x != 0.0:
            print(f"  X[{name}] = {x:.10f}")

    print("\nmass fractions (ненулевые):")
    for name, y in zip(gas.species_names, gas.Y):
        if y != 0.0:
            print(f"  Y[{name}] = {y:.10f}")

    conc = gas.concentrations           # kmol/m3
    wdot = gas.net_production_rates     # kmol/m3/s

    print("\nconcentrations (kmol/m3) - сравнивать с conc[] из pyjac_probe:")
    for name, c in zip(gas.species_names, conc):
        print(f"  conc[{name}] = {c:.15e}")

    print("\nnet production rates wdot (kmol/m3/s) - сравнивать с "
          "spec_rates[] из pyjac_probe:")
    for name, w in zip(gas.species_names, wdot):
        print(f"  wdot[{name}] = {w:.15e}")

    # Точно те же формулы, что в src/ODESystem/grimech/out/dydt.cu (ветка CONV):
    #   dY_k/dt = spec_rates_k * MW_k / rho
    #   dT/dt   = -(1/(rho*cv)) * sum_k(u_k_molar * spec_rates_k)
    MW = gas.molecular_weights                    # kg/kmol
    u_molar = gas.partial_molar_int_energies       # J/kmol
    cv_mass = gas.cv_mass                          # J/(kg*K)

    dY = wdot * MW / rho
    dT = -np.sum(u_molar * wdot) / (rho * cv_mass)

    print("\n=== Cantera dy/dt по формулам CONV из dydt.cu "
          "(сравнивать с dy/dt из pyjac_probe) ===")
    print(f"  dT/dt = {dT:.15e}")
    for name, dy in zip(gas.species_names, dY):
        print(f"  dY[{name}]/dt = {dy:.15e}")

    print("\nforward/reverse rate constants, первые 10 реакций (сравнивать "
          "с fwd_rates/rev_rates из pyjac_probe ТОЛЬКО если вы проверили, "
          "что pyJac не переставлял порядок реакций относительно gri30):")
    kf = gas.forward_rate_constants
    kr = gas.reverse_rate_constants
    for i in range(min(10, gas.n_reactions)):
        print(f"  rxn {i:3d} ({gas.reaction(i).equation}): "
              f"kf = {kf[i]:.6e}   kr = {kr[i]:.6e}")


if __name__ == "__main__":
    main()
