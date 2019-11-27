# assembled position
solution = []
with open("../bin/solution_position.txt", "r") as f:
    for line in f:
        solution.append(line.strip().split(" "))
# origin position
image = []
with open("../bin/image_position.txt", "r") as f:
    for line in f:
        image.append(line.strip().split(" "))

size = 0
complete = 0


def find(piece, i):
    for row in range(len(piece)):
        row_ = piece[row]
        for col in range(len(row_)):
            if piece[row][col] == i:
                return row * len(row_) + col


for i in range(len(image)):
    image_row = image[i]
    solution_row = solution[i]
    for j in range(len(solution_row)):
        if solution[i][j] == find(image, str(i)):
            complete += 1
        size += 1
print("complete %s, size %s, %s", (complete, size, complete / size))
