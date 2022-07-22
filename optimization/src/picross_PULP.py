from argparse import ArgumentParser

from pulp import LpProblem, LpBinary, LpVariable, lpSum, PULP_CBC_CMD, LpStatus


def argparser():
    parser = ArgumentParser()
    parser.add_argument(
        "-r",
        "--problem",
        type=str,
        required=True,
    )
    return parser


def read_problem(path):
    picross = {"row": list(), "column": list()}
    for line in open(path):
        line = line.rstrip("\n")
        if "row" in line:
            mode = "row"
        elif "column" in line:
            mode = "column"
        else:
            try:
                line = list(map(int, line.split(" ")))
                picross[mode].append(line)
            except Exception:
                pass
    assert len(picross["row"]) == len(picross["column"])
    assert len(picross["row"]) != 0
    return picross


def main(picross):
    row = picross["row"]
    column = picross["column"]

    problem = LpProblem("Picross")

    x = [
        [LpVariable(f"x({i})({j})", cat=LpBinary) for j in range(len(column))]
        for i in range(len(row))
    ]

    y = [
        [
            [LpVariable(f"y({i})({j})({k})", cat=LpBinary) for k in range(len(row[i]))]
            for j in range(len(column))
        ]
        for i in range(len(row))
    ]
    z = [
        [
            [
                LpVariable(f"z({i})({j})({k})", cat=LpBinary)
                for k in range(len(column[j]))
            ]
            for j in range(len(column))
        ]
        for i in range(len(row))
    ]

    for i in range(len(row)):
        if row[i] == [0]:
            problem += lpSum(x[i][j] for j in range(len(column))) == 0
        else:
            for k in range(len(row[i])):
                problem += lpSum(y[i][j][k] for j in range(len(column))) == 1

    for j in range(len(column)):
        if column[j] == [0]:
            problem += lpSum(x[i][j] for i in range(len(row))) == 0
        else:
            for k in range(len(column[j])):
                problem += lpSum(z[i][j][k] for i in range(len(row))) == 1

    for i in range(len(row)):
        for j in range(len(column)):
            for k in range(len(row[i]) - 1):
                problem += y[i][j][k] <= lpSum(
                    y[i][m][k + 1] for m in range(j + row[i][k] + 1, len(row))
                )
            problem += (
                lpSum(y[i][j][-1] for j in range(len(row) - row[i][-1] + 1, len(row)))
                == 0
            )

    for i in range(len(row)):
        for j in range(len(column)):
            for k in range(len(column[j]) - 1):
                problem += z[i][j][k] <= lpSum(
                    z[m][j][k + 1] for m in range(i + column[j][k] + 1, len(column))
                )
            problem += (
                lpSum(
                    z[i][j][-1]
                    for i in range(len(column) - column[j][-1] + 1, len(column))
                )
                == 0
            )

    for i in range(len(row)):
        for j in range(len(column)):
            y_set = list()
            for k in range(len(row[i])):
                y_set += [y[i][m][k] for m in range(max(0, j - row[i][k] + 1), j + 1)]
            problem += x[i][j] <= lpSum(y_set)

    for i in range(len(row)):
        for j in range(len(column)):
            y_set = list()
            for k in range(len(column[j])):
                y_set += [
                    z[m][j][k] for m in range(max(0, i - column[j][k] + 1), i + 1)
                ]
            problem += x[i][j] <= lpSum(y_set)

    for i in range(len(row)):
        for j in range(len(column)):
            for k in range(len(row[i])):
                problem += x[i][j] >= lpSum(
                    y[i][m][k] for m in range(j - row[i][k] + 1, j + 1)
                )

    for i in range(len(row)):
        for j in range(len(column)):
            for k in range(len(column[j])):
                problem += x[i][j] >= lpSum(
                    z[m][j][k] for m in range(i - column[j][k] + 1, i + 1)
                )

    solver = PULP_CBC_CMD()
    result = problem.solve(solver=solver)
    status = LpStatus[result]

    if status == "Optimal":
        for i in range(len(row)):
            box = ""
            for j in range(len(column)):
                if x[i][j].value() == 1:
                    box += "■ "
                else:
                    box += "□ "
            print(box)
    else:
        print("解なし")


if __name__ == "__main__":
    parser = argparser()
    args = parser.parse_args()
    picross = read_problem(args.problem)
    main(picross)
