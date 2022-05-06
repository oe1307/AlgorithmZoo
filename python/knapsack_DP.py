from argparse import ArgumentParser


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
    return 0


def main(tmp):
    pass


if __name__ == "__main__":
    parser = argparser()
    args = parser.parse_args()
    tmp = read_problem(args.problem)
    main(tmp)
