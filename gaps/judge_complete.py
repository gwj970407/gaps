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

for i in range(len(image)):
    image_row = image[i]
    solution_row = solution[i]
    for j in range(len(solution_row)):
        if solution[i][j] == str(i * len(solution_row) + j) :
            complete += 1
        size += 1
print("complete %s, size %s, %s", (complete, size, complete / size))
