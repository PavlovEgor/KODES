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


def reorder(values, from_order, to_order):
    by_name = dict(zip(from_order, values))
    return np.array([by_name[name] for name in to_order])


def compute_dTY(gas, T, P, Y_pyjac_full):
    """dT/dt и dY/dt (оба в порядке pyjac) по тем же формулам CONP, что и в
    src/ODESystem/grimech/out/dydt.cu:
        dY_k/dt = spec_rates_k * MW_k / rho
        dT/dt   = -(1/(rho*cp)) * sum_k(h_k_molar * spec_rates_k)
    Y_pyjac_full - полный (все NSP видов, включая последний неявный) вектор
    массовых долей в порядке pyjac.
    """
    Y_cantera = reorder(Y_pyjac_full, PYJAC_SPECIES_ORDER, gas.species_names)
    gas.TPY = T, P, Y_cantera

    wdot_cantera = gas.net_production_rates        # kmol/m3/s
    rho = gas.density
    MW_cantera = gas.molecular_weights              # kg/kmol
    h_molar_cantera = gas.partial_molar_enthalpies  # J/kmol
    cp_mass = gas.cp_mass                            # J/(kg*K)

    dY_cantera = wdot_cantera * MW_cantera / rho
    dT = -np.sum(h_molar_cantera * wdot_cantera) / (rho * cp_mass)

    dY_pyjac = reorder(dY_cantera, gas.species_names, PYJAC_SPECIES_ORDER)
    return dT, dY_pyjac


def numeric_jacobian(gas, T0, P, Y_pyjac_full, dT_step=1e-3, dY_step=1e-7):
    """Численный (центральные разности) якобиан той же compute_dTY() по
    независимым переменным состояния (T, Y_0..Y_{NSP-2} в порядке pyjac) -
    последний вид (Y_{NSP-1}) зависимый, Y_{NSP-1} = 1 - sum(остальных),
    точно как в pyjac_probe.cu/eval_jacob().

    Возвращает J[row, col] = d(d(var_row)/dt) / d(var_col), где var_0 = T,
    var_k (k=1..NSP-1) = Y[pyjac species k-1] - то есть в РОВНО той же
    раскладке, что jac[col*NSP+row] в pyjac_probe.cu.
    """
    n = len(PYJAC_SPECIES_ORDER)  # NSP
    Y_explicit0 = np.array(Y_pyjac_full[:n - 1], dtype=float)

    def f(T, Y_explicit):
        Yfull = np.empty(n)
        Yfull[:n - 1] = Y_explicit
        Yfull[n - 1] = 1.0 - Y_explicit.sum()
        return compute_dTY(gas, T, P, Yfull)

    J = np.zeros((n, n))
    for col in range(n):
        if col == 0:
            dT_p, dY_p = f(T0 + dT_step, Y_explicit0)
            dT_m, dY_m = f(T0 - dT_step, Y_explicit0)
            h2 = 2.0 * dT_step
        else:
            i = col - 1
            Yp = Y_explicit0.copy(); Yp[i] += dY_step
            Ym = Y_explicit0.copy(); Ym[i] -= dY_step
            dT_p, dY_p = f(T0, Yp)
            dT_m, dY_m = f(T0, Ym)
            h2 = 2.0 * dY_step
        J[0, col] = (dT_p - dT_m) / h2
        J[1:, col] = (dY_p[:n - 1] - dY_m[:n - 1]) / h2
    return J


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

    # Y_pyjac_full - полный вектор массовых долей в порядке pyjac (см. compute_dTY).
    Y_pyjac_full = reorder(gas.Y, gas.species_names, PYJAC_SPECIES_ORDER)
    dT, dY_pyjac = compute_dTY(gas, T, P, Y_pyjac_full)
    # compute_dTY() выше уже поменял gas.TPY - возвращаем состояние к исходному,
    # чтобы дальнейшие вызовы (forward/reverse rate constants и т.п.) считались
    # для того же состояния, что было задано изначально через gas.TPX.
    gas.TPX = T, P, X

    print("\n=== Cantera dy/dt по формулам CONP из dydt.cu "
          "(сравнивать с dy/dt из pyjac_probe) ===")
    print(f"  dT/dt = {dT:.15e}")
    for name, dy in zip(PYJAC_SPECIES_ORDER[:-1], dY_pyjac[:-1]):
        print(f"  dY[{name}]/dt = {dy:.15e}")

    print("\nforward/reverse rate constants, первые 10 реакций (сравнивать "
          "с fwd_rates/rev_rates из pyjac_probe ТОЛЬКО если вы проверили, "
          "что pyJac не переставлял порядок реакций относительно gri30):")
    kf = gas.forward_rate_constants
    kr = gas.reverse_rate_constants
    for i in range(min(10, gas.n_reactions)):
        print(f"  rxn {i:3d} ({gas.reaction(i).equation}): "
              f"kf = {kf[i]:.6e}   kr = {kr[i]:.6e}")

    # Численный якобиан (центральные разности) той же compute_dTY() - эталон
    # для сравнения с jac[] из eval_jacob() в pyjac_probe.cu. Метки и раскладка
    # (jac[col*NSP+row], col = переменная T/Y[...], row = уравнение d(...)/dt)
    # намеренно совпадают со строками, которые печатает pyjac_probe, чтобы
    # можно было сравнивать построчно по метке "d(...)/d(...)", не по номеру
    # строки в файле.
    n = len(PYJAC_SPECIES_ORDER)
    J = numeric_jacobian(gas, T, P, Y_pyjac_full)
    gas.TPX = T, P, X  # снова возвращаем состояние газа к исходному

    print("\n=== Cantera: численный якобиан (центральные разности) той же "
          "compute_dTY(), сравнивать с jac[] из eval_jacob() в pyjac_probe ===")
    print("J[row,col] = d(d(var_row)/dt)/d(var_col); печатается в той же "
          "column-major раскладке jac[col*NSP+row], что и в pyjac_probe.cu\n")
    for col in range(n):
        colName = "T" if col == 0 else PYJAC_SPECIES_ORDER[col - 1]
        for row in range(n):
            rowName = "T" if row == 0 else PYJAC_SPECIES_ORDER[row - 1]
            idx = col * n + row
            val = J[row, col]
            if row == 0 and col == 0:
                print(f"  jac[{idx}] d(dT/dt)/d(T) = {val:.15e}")
            elif row == 0:
                print(f"  jac[{idx}] d(dT/dt)/d(Y[{colName}]) = {val:.15e}")
            elif col == 0:
                print(f"  jac[{idx}] d(dY[{rowName}]/dt)/d(T) = {val:.15e}")
            else:
                print(f"  jac[{idx}] d(dY[{rowName}]/dt)/d(Y[{colName}]) = {val:.15e}")


if __name__ == "__main__":
    main()
