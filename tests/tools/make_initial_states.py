#!/usr/bin/env python3
"""Write the simplest state sets of the test cases: the streams of each
case.json - the initial and inlet states of its tutorial - and mixtures of
pairs of them, one <mechanism>.csv per mechanism the case runs with.

    tests/tools/make_initial_states.py              every case
    tests/tools/make_initial_states.py 2D_jet_flame one

A mixture at fraction a is a*first + (1-a)*second in mass fractions and in
temperature. Mixing the temperature linearly ignores that the two cp differ,
and the point of these states is only to have something below, at and above
ignition until there are states of a developed flame - see ../README.md.

The same states go to every mechanism: they name only major species, which
every mechanism of a case has.
"""
import argparse
import json
import sys
from pathlib import Path

from states import CASES_DIR, write_states


def states_of(case):
    streams = case["streams"]
    states = [(name, s["T"], s["p"], s["Y"]) for name, s in streams.items()]

    for first, second in case["mixtures"]:
        s1, s2 = streams[first], streams[second]

        if s1["p"] != s2["p"]:
            sys.exit(f"{first} and {second} are at different pressures")

        for a in case["mixingFractions"]:
            species = list(dict.fromkeys([*s1["Y"], *s2["Y"]]))
            Y = {sp: a*s1["Y"].get(sp, 0) + (1 - a)*s2["Y"].get(sp, 0) for sp in species}
            T = a*s1["T"] + (1 - a)*s2["T"]
            states.append((f"{first}{a:g}+{second}", T, s1["p"], Y))

    return states


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("cases", nargs="*", help="case directories under tests/cases (default: all)")
    args = parser.parse_args()

    names = args.cases or sorted(p.name for p in CASES_DIR.iterdir() if (p/"case.json").exists())

    for name in names:
        case = json.loads((CASES_DIR/name/"case.json").read_text())

        # A case whose states are a developed flame has no streams to make any
        # from, and its files are not to be overwritten
        if "streams" not in case:
            print(f"{name}: states of a developed flame, left as they are")
            continue

        states = states_of(case)

        for mechanism in case["mechanisms"]:
            path = CASES_DIR/name/f"{mechanism}.csv"
            write_states(path, states)
            print(f"{path.relative_to(CASES_DIR.parent)}: {len(states)} states")


if __name__ == "__main__":
    main()
