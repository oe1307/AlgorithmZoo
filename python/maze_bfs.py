from argparse import ArgumentParser
from copy import deepcopy


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
    maze = [""]
    position = {
        "start": (),
        "goal": (),
    }
    for i, line in enumerate(open(problem)):
        if (j := line.find("S")) != -1:
            position["start"] = (i + 1, j + 1)
        elif (j := line.find("G")) != -1:
            position["goal"] = (i + 1, j + 1)
        line = ["■"] + list(line.strip()) + ["■"]
        maze.append(line)
    maze[0] = ["■"] * len(line)
    maze.append(["■"] * len(line))
    return maze, position


def print_maze(maze):
    for line in maze:
        print(" ".join(line))


def main(maze, position):
    print_maze(maze)
    maze_map = deepcopy(maze)
    search_points = [position["start"]]
    time = 0
    while not position["goal"] in search_points:
        next_search = list()
        time += 1
        for point in search_points:
            if maze_map[point[0] - 1][point[1]] in "□G":
                next_search.append((point[0] - 1, point[1]))
                maze_map[point[0] - 1][point[1]] = str(time)
            if maze_map[point[0] + 1][point[1]] in "□G":
                next_search.append((point[0] + 1, point[1]))
                maze_map[point[0] + 1][point[1]] = str(time)
            if maze_map[point[0]][point[1] - 1] in "□G":
                next_search.append((point[0], point[1] - 1))
                maze_map[point[0]][point[1] - 1] = str(time)
            if maze_map[point[0]][point[1] + 1] in "□G":
                next_search.append((point[0], point[1] + 1))
                maze_map[point[0]][point[1] + 1] = str(time)
        search_points = next_search

    print("\n" + "  " * (len(maze[0]) // 2) + "↓" + "  " * (len(maze[0]) // 2) + "\n")

    point = position["goal"]
    for t in reversed(range(time)):
        if maze_map[point[0] - 1][point[1]] == str(t):
            maze[point[0] - 1][point[1]] = "+"
            point = (point[0] - 1, point[1])
        elif maze_map[point[0] + 1][point[1]] == str(t):
            maze[point[0] + 1][point[1]] = "+"
            point = (point[0] + 1, point[1])
        elif maze_map[point[0]][point[1] - 1] == str(t):
            maze[point[0]][point[1] - 1] = "+"
            point = (point[0], point[1] - 1)
        elif maze_map[point[0]][point[1] + 1] == str(t):
            maze[point[0]][point[1] + 1] = "+"
            point = (point[0], point[1] + 1)
    print_maze(maze)


if __name__ == "__main__":
    parser = argparser()
    args = parser.parse_args()
    maze, position = read_problem(args.problem)
    main(maze, position)
