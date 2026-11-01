"""The state file format every tool here reads and writes, and every runner -
KODES, Cantera, OpenFOAM - is to read.

<mechanism>.csv, one state per row:

    label,T,p,<specie>,<specie>,...

T in K, p in Pa, then mass fractions by specie name. A specie of the
mechanism that has no column is zero; a column naming a specie the mechanism
does not have is an error, not something to drop in silence. Lines starting
with # are comments.

By name and not in pyJac's order: the same file has to reach OpenFOAM's
species table and Cantera's as well, and pyJac's order is not even fixed for a
mechanism - it moves the eliminated specie to the end.
"""
import csv
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent.parent
CASES_DIR = TESTS_DIR/"cases"


def write_states(path, states, comment=None):
    """states: (label, T, p, {specie: Y}) tuples"""
    species = list(dict.fromkeys(sp for _, _, _, Y in states for sp in Y))

    with open(path, "w", newline="") as f:
        if comment:
            for line in comment.splitlines():
                f.write(f"# {line}\n")

        w = csv.writer(f, lineterminator="\n")
        w.writerow(["label", "T", "p", *species])

        for label, T, p, Y in states:
            w.writerow([label, repr(float(T)), repr(float(p)), *(repr(float(Y.get(sp, 0))) for sp in species)])


def read_states(path):
    """(label, T, p, {specie: Y}) tuples"""
    with open(path, newline="") as f:
        rows = csv.reader(line for line in f if not line.startswith("#"))
        header = next(rows)

        if header[:3] != ["label", "T", "p"]:
            raise ValueError(f"{path}: the header must start with label,T,p")

        species = header[3:]

        return [
            (row[0], float(row[1]), float(row[2]), dict(zip(species, map(float, row[3:]))))
            for row in rows
        ]


def mechanism_species(mechanism_cuh):
    """The species of a pyJac mechanism.cuh, in pyJac's order, from the
    'Species Indexes' comment it opens with"""
    names = []
    inside = False

    for line in Path(mechanism_cuh).read_text().splitlines():
        if "Species Indexes" in line:
            inside = True
            continue

        if inside:
            if line.strip().startswith("*/"):
                break

            index, name = line.split()
            assert int(index) == len(names), f"{mechanism_cuh}: species out of order at {line!r}"
            names.append(name)

    if not names:
        raise ValueError(f"{mechanism_cuh}: no 'Species Indexes' comment")

    return names
