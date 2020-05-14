# assembled position
import argparse


def judge_complete(solution_file, image_file):
    solution = []
    with open(solution_file, "r") as f:
        for line in f:
            solution.append(line.strip().split(" "))
    # origin position
    image = []
    with open(image_file, "r") as f:
        for line in f:
            image.append(line.strip().split(" "))

    size = 0
    complete = 0

    for i in range(len(image)):
        image_row = image[i]
        solution_row = solution[i]
        for j in range(len(solution_row)):
            if solution[i][j] == str(i * len(solution_row) + j):
                complete += 1
            size += 1
    print("complete %s, size %s, %s" % (complete, size, complete / size))


def parse_arguments():
    """Parses input arguments required to create puzzle"""
    description = "judge puzzle pieces restructor rate  by giving file.\n"

    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--image_file", type=str, default="image_position.txt")
    parser.add_argument("--solution_file", type=str, default="solution_position.txt")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    judge_complete(args.solution_file, args.image_file)
