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


def main(problem):
    for line in open(problem):
        pass


if __name__ == "__main__":
    parser = argparser()
    args = parser.parse_args()
    main(args.problem)
