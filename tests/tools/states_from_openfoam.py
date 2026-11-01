#!/usr/bin/env python3
"""Take the states of a test case from a time directory of its OpenFOAM run -
the developed flame the simplest states stand in for until then.

    tests/tools/states_from_openfoam.py <OpenFOAM case> <time> <mechanism.cuh> <out.csv>
        [--max-states N] [--sample ranks|temperature]

    # e.g. from kodes4foam/tutorials, after ./Allrun grimech in 2D_planar_flame:
    tests/tools/states_from_openfoam.py 2D_planar_flame 0.0005 \\
        mechanisms/grimech/mechanism.cuh ../external/KODES/tests/cases/2D_planar_flame/grimech.csv

Reads T, p and one field per specie of the mechanism from <case>/<time>; a
specie with no file of its own takes the time directory's Ydefault, as
OpenFOAM would. ASCII or binary fields (LSB, 32 bit labels, 64 bit scalars),
uncompressed, of a reconstructed case: run reconstructPar first.

A mesh of 40 000 cells is too many states to keep in git, so --max-states
(default 2048) picks some of them:

  ranks        cells evenly spaced in the order of their temperature: the same
               distribution as the mesh, which is what a benchmark against the
               whole run should see (the default)
  temperature  the same number of cells from each of max-states temperature
               bins: the whole range evenly, cold cells no longer the bulk -
               what a validation should see
"""
import argparse
import re
import struct
from pathlib import Path

from states import mechanism_species, write_states


def read_field(path, n_cells=None):
    """The internal field of a volScalarField, ASCII or binary, as a list of
    floats"""
    data = Path(path).read_bytes()

    # the header and the keywords are ASCII whatever the format
    m = re.search(rb"internalField\s+uniform\s+([^;\s]+)\s*;", data)
    if m:
        if n_cells is None:
            raise ValueError(f"{path}: uniform, and no other field says how many cells there are")
        return [float(m.group(1))]*n_cells

    m = re.search(rb"internalField\s+nonuniform\s+List<scalar>\s+(\d+)\s*\(", data)
    if not m:
        raise ValueError(f"{path}: no internalField")

    n = int(m.group(1))

    if re.search(rb"format\s+binary", data):
        arch = re.search(rb'arch\s+"([^"]*)"', data)
        if arch and arch.group(1) != b"LSB;label=32;scalar=64":
            raise ValueError(f"{path}: arch {arch.group(1).decode()}, only LSB;label=32;scalar=64 is read")

        values = list(struct.unpack_from(f"<{n}d", data, m.end()))

        if data[m.end() + 8*n:m.end() + 8*n + 1] != b")":
            raise ValueError(f"{path}: the binary list does not end after {n} values")

        return values

    body = data[m.end():data.index(b")", m.end())]
    values = [float(v) for v in body.split()]

    if len(values) != n:
        raise ValueError(f"{path}: {len(values)} values, the list says {n}")

    return values


def mesh_cells(case):
    """The number of cells, from the note blockMesh and the other mesh
    utilities leave in polyMesh/owner; None if there is none"""
    owner = Path(case)/"constant"/"polyMesh"/"owner"
    if not owner.exists():
        return None
    m = re.search(rb"nCells:\s*(\d+)", owner.read_bytes()[:4096])
    return int(m.group(1)) if m else None


def pick(T, max_states, how):
    """Indices of the cells to keep"""
    order = sorted(range(len(T)), key=T.__getitem__)

    if len(order) <= max_states:
        return order

    if how == "ranks":
        return [order[round(i*(len(order) - 1)/(max_states - 1))] for i in range(max_states)]

    # temperature: walk up the bins, one cell nearest each bin's centre
    lo, hi = T[order[0]], T[order[-1]]
    picked, j = [], 0
    for i in range(max_states):
        target = lo + (i + 0.5)*(hi - lo)/max_states
        while j + 1 < len(order) and abs(T[order[j + 1]] - target) <= abs(T[order[j]] - target):
            j += 1
        if not picked or picked[-1] != order[j]:
            picked.append(order[j])
    return picked


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("case", type=Path)
    parser.add_argument("time")
    parser.add_argument("mechanism", type=Path, help="the mechanism.cuh the case ran with")
    parser.add_argument("out", type=Path)
    parser.add_argument("--max-states", type=int, default=2048)
    parser.add_argument("--sample", choices=["ranks", "temperature"], default="ranks")
    args = parser.parse_args()

    time_dir = args.case/args.time
    species = mechanism_species(args.mechanism)

    T = read_field(time_dir/"T", mesh_cells(args.case))
    p = read_field(time_dir/"p", len(T))

    Y = {}
    missing = []
    for sp in species:
        if (time_dir/sp).exists():
            Y[sp] = read_field(time_dir/sp, len(T))
        else:
            missing.append(sp)

    if missing:
        default = read_field(time_dir/"Ydefault", len(T)) if (time_dir/"Ydefault").exists() else [0.0]*len(T)
        for sp in missing:
            Y[sp] = default
        print(f"{len(missing)} species from Ydefault: {' '.join(missing)}")

    cells = pick(T, args.max_states, args.sample)

    states = [
        (f"cell{c}", T[c], p[c], {sp: Y[sp][c] for sp in species if Y[sp][c] != 0})
        for c in cells
    ]

    write_states(
        args.out, states,
        comment=f"{args.case.resolve().name} at t = {args.time}, {len(cells)} of {len(T)} cells by {args.sample}"
    )

    print(f"{args.out}: {len(states)} of {len(T)} cells, T {min(T):.0f}..{max(T):.0f} K")


if __name__ == "__main__":
    main()
