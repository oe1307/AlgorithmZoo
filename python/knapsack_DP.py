import re
from argparse import ArgumentParser

import pandas as pd
import numpy as np


def argparser():
    parser = ArgumentParser()
    parser.add_argument(
        "-r",
        "--problem",
        type=str,
        required=True,
    )
    return parser


def read_problem(problem):
    database = list()
    compiler = re.compile("capacity (?P<capacity>[0-9]+)*")
    for line in open(problem):
        if (m := compiler.match(line)) is not None:
            capacity = int(m.groupdict()["capacity"])
        else:
            try:
                database.append(list(map(int, line.split(" "))))
            except Exception:
                pass
    return capacity, database


def main(capacity, database):
    dp = np.zeros((len(database) + 1, capacity + 1))
    for i, (w, v) in enumerate(database):
        for cap in range(capacity):
            if cap + 1 >= w:
                dp[i + 1][cap + 1] = max(dp[i][cap + 1], dp[i][cap + 1 - w] + v)
            else:
                dp[i + 1][cap + 1] = dp[i][cap + 1]
    print(pd.DataFrame(dp))


if __name__ == "__main__":
    parser = argparser()
    args = parser.parse_args()
    capacity, database = read_problem(args.problem)
    main(capacity, database)
