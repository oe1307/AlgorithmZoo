from argparse import ArgumentParser

from base import Variable, Problem
from solver import get_solver
from utils import setup_logger


def argparser():
    parser = ArgumentParser()
    parser.add_argument(
        "-s",
        "--solver",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--log_level",
        type=int,
        default=20,
        help="10:DEBUG,20:INFO,30:WARNING,40:ERROR,50:CRITICAL",
    )
    return parser


def main(solver):
    x = [Variable() for i in range(5)]

    problem = Problem(case="minimize")

    result = solver.solve(problem)

    print(f"best objective: {result.objective}")
    for i in range(5):
        print(f"x[{i}] = {x[i].value}")


if __name__ == "__main__":
    parser = argparser()
    args = parser.parse_args()
    logger = setup_logger.setLevel(args.log_level)
    solver = get_solver(args.solver)
    main(solver)
