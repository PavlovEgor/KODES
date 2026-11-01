#!/usr/bin/env python3
"""Check the state files of the test cases: the format, T and p positive, mass
fractions in [0, 1] that sum to 1, and - given where the mechanisms are -
that every specie column is a specie of the mechanism.

    tests/tools/check_states.py
    tests/tools/check_states.py --mechanisms ../../tutorials/mechanisms

--mechanisms is a directory of <mechanism>/mechanism.cuh, kodes4foam's
tutorials/mechanisms for example. Exits 1 on the first file that fails.
"""
import argparse
import json
import sys
from pathlib import Path

from states import CASES_DIR, read_states, mechanism_species

SUM_TOL = 1e-6


def check(path, species):
    problems = []
    states = read_states(path)

    if not states:
        problems.append("no states")

    for label, T, p, Y in states:
        if not T > 0 or not p > 0:
            problems.append(f"{label}: T = {T}, p = {p}")

        bad = {sp: y for sp, y in Y.items() if not 0 <= y <= 1}
        if bad:
            problems.append(f"{label}: mass fractions outside [0, 1]: {bad}")

        if abs(sum(Y.values()) - 1) > SUM_TOL:
            problems.append(f"{label}: mass fractions sum to {sum(Y.values()):.10g}")

        if species is not None:
            unknown = [sp for sp in Y if sp not in species]
            if unknown:
                problems.append(f"{label}: not species of the mechanism: {unknown}")

    return len(states), problems


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mechanisms", type=Path, help="directory of <mechanism>/mechanism.cuh")
    args = parser.parse_args()

    failed = False

    for case_dir in sorted(p for p in CASES_DIR.iterdir() if (p/"case.json").exists()):
        case = json.loads((case_dir/"case.json").read_text())

        for mechanism in case["mechanisms"]:
            path = case_dir/f"{mechanism}.csv"
            name = path.relative_to(CASES_DIR.parent)

            if not path.exists():
                print(f"{name}: missing")
                failed = True
                continue

            species = None
            if args.mechanisms:
                species = mechanism_species(args.mechanisms/mechanism/"mechanism.cuh")

            n, problems = check(path, species)
            print(f"{name}: {n} states, " + ("ok" if not problems else f"{len(problems)} problem(s)"))

            for problem in problems:
                print(f"    {problem}")

            failed |= bool(problems)

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
