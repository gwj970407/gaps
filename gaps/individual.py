import random
import numpy as np
from gaps import image_helpers
from gaps.image_analysis import ImageAnalysis
from gaps.piece import Piece


class Individual(object):
    """Class representing possible solution to puzzle.

    Individual object is one of the solutions to the problem
    (possible arrangement of the puzzle's pieces).
    It is created by random shuffling initial puzzle.

    :param pieces:  Array of pieces representing initial puzzle.
    :param rows:    Number of rows in input puzzle
    :param columns: Number of columns in input puzzle

    Usage::

        >>> from gaps.individual import Individual
        >>> from gaps.image_helpers import flatten_image
        >>> pieces, rows, columns = flatten_image(...)
        >>> ind = Individual(pieces, rows, columns)

    """

    FITNESS_FACTOR = 1000

    def __init__(self, pieces, rows, columns, shuffle=True):

        # 初始化了一整张图片
        self.pieces = pieces[:]
        self.rows = rows
        self.columns = columns
        self._fitness = None

        # 惩罚函数
        self.penalize_image = None

        if shuffle:
            # [0,1,2,3] -> [3, 1, 2, 0]
            np.random.shuffle(self.pieces)

        # Map piece ID to index in Individual's list
        # pieces : [piece1, piece2, ...]
        # pipce : {id: 2}
        # mapping : {3:0, 1:1, 2:2, 0:3}
        self._piece_mapping = {piece.id: index for index, piece in enumerate(self.pieces)}

    def __getitem__(self, key):
        return self.pieces[key * self.columns:(key + 1) * self.columns]

    @property
    def fitness(self):
        """Evaluates fitness value.

        Fitness value is calculated as sum of dissimilarity measures between each adjacent pieces.

        这个方法会在第一次调用时被执行，懒初始化fitness，结果是适应度因子除以空间距离，（结果越大适应度越好）
        """
        if self._fitness is None:
            fitness_value = 1 / self.FITNESS_FACTOR
            # For each two adjacent pieces in rows
            # 累加所有左右关系的碎片的空间距离
            for i in range(self.rows):
                for j in range(self.columns - 1):
                    ids = (self[i][j].id, self[i][j + 1].id)
                    fitness_value += ImageAnalysis.get_dissimilarity(ids, orientation="LR")
            # For each two adjacent pieces in columns
            # 累加所有上下关系的碎片的空间距离
            for i in range(self.rows - 1):
                for j in range(self.columns):
                    ids = (self[i][j].id, self[i + 1][j].id)
                    fitness_value += ImageAnalysis.get_dissimilarity(ids, orientation="TD")

            self._fitness = self.FITNESS_FACTOR / fitness_value

        return self._fitness

    def clear_fitness(self):
        self._fitness = None

    def piece_size(self):
        """Returns single piece size"""
        return self.pieces[0].size

    def piece_by_id(self, identifier):
        """"Return specific piece from individual"""
        return self.pieces[self._piece_mapping[identifier]]

    def best_adjoin(self, piece_size):
        pieces = np.reshape(self.pieces, (self.rows, self.columns))
        empty_image = np.zeros((piece_size, piece_size, pieces[0][0].shape()[2]))
        empty_piece = Piece(empty_image, -1)
        for row in range(self.rows):
            for col in range(self.columns):
                if row == 0:
                    if col == 0 and ImageAnalysis.in_range(pieces[row][col].id, "D", pieces[row + 1][col].id) \
                            and ImageAnalysis.in_range(pieces[row][col].id, "R", pieces[row][col + 1].id):
                        continue
                    if col < self.columns - 1 and ImageAnalysis.in_range(pieces[row][col].id, "D",
                                                                         pieces[row + 1][col].id) \
                            and ImageAnalysis.in_range(pieces[row][col].id, "R", pieces[row][col + 1].id):
                        continue
                    if col == self.columns - 1 and ImageAnalysis.in_range(pieces[row][col].id, "D",
                                                                          pieces[row + 1][col].id) \
                            and ImageAnalysis.in_range(pieces[row][col].id, "L", pieces[row][col - 1].id):
                        continue
                if 0 < row < self.rows - 1:
                    if col == 0 and ImageAnalysis.in_range(pieces[row][col].id, "D", pieces[row + 1][col].id) \
                            and ImageAnalysis.in_range(pieces[row][col].id, "R", pieces[row][col + 1].id):
                        continue
                    if col < self.columns - 1 and ImageAnalysis.in_range(pieces[row][col].id, "D",
                                                                         pieces[row + 1][col].id) \
                            and ImageAnalysis.in_range(pieces[row][col].id, "R", pieces[row][col + 1].id):
                        continue
                    if col == self.columns - 1 and ImageAnalysis.in_range(pieces[row][col].id, "D",
                                                                          pieces[row + 1][col].id) \
                            and ImageAnalysis.in_range(pieces[row][col].id, "L", pieces[row][col - 1].id):
                        continue
                if row == self.rows - 1:
                    if col == 0 and ImageAnalysis.in_range(pieces[row][col].id, "T", pieces[row - 1][col].id) \
                            and ImageAnalysis.in_range(pieces[row][col].id, "R", pieces[row][col + 1].id):
                        continue
                    if col < self.columns - 1 and ImageAnalysis.in_range(pieces[row][col].id, "R",
                                                                         pieces[row][col + 1].id) \
                            and ImageAnalysis.in_range(pieces[row][col].id, "T", pieces[row - 1][col].id):
                        continue
                    if col == self.columns - 1 and ImageAnalysis.in_range(pieces[row][col].id, "L",
                                                                          pieces[row][col - 1].id) \
                            and ImageAnalysis.in_range(pieces[row][col].id, "T", pieces[row - 1][col].id):
                        continue
                # if row == 0:
                #     if col == 0 and ImageAnalysis.best_match(pieces[row][col].id, "D") == pieces[row + 1][col].id \
                #             and ImageAnalysis.best_match(pieces[row][col].id, "R") == pieces[row][col + 1].id:
                #         continue
                #     if col < self.columns - 1 and ImageAnalysis.best_match(pieces[row][col].id, "D") == pieces[row + 1][col].id \
                #             and ImageAnalysis.best_match(pieces[row][col].id, "R") == pieces[row][col + 1].id \
                #             and ImageAnalysis.best_match(pieces[row][col].id, "L") == pieces[row][col - 1].id:
                #         continue
                #     if col == self.columns - 1 and ImageAnalysis.best_match(pieces[row][col].id, "D") == pieces[row + 1][col].id \
                #             and ImageAnalysis.best_match(pieces[row][col].id, "L") == pieces[row][col - 1].id:
                #         continue
                # if 0 < row < self.rows - 1:
                #     if col == 0 and ImageAnalysis.best_match(pieces[row][col].id, "D") == pieces[row + 1][col].id \
                #             and ImageAnalysis.best_match(pieces[row][col].id, "T") == pieces[row - 1][col].id \
                #             and ImageAnalysis.best_match(pieces[row][col].id, "R") == pieces[row][col + 1].id:
                #         continue
                #     if col < self.columns - 1 and ImageAnalysis.best_match(pieces[row][col].id, "T") == pieces[row - 1][col].id \
                #             and ImageAnalysis.best_match(pieces[row][col].id, "D") == pieces[row + 1][col].id \
                #             and ImageAnalysis.best_match(pieces[row][col].id, "L") == pieces[row][col - 1].id \
                #             and ImageAnalysis.best_match(pieces[row][col].id, "R") == pieces[row][col + 1].id:
                #         continue
                #     if col == self.columns - 1 and ImageAnalysis.best_match(pieces[row][col].id, "T") == pieces[row - 1][col].id \
                #             and ImageAnalysis.best_match(pieces[row][col].id, "D") == pieces[row + 1][col].id \
                #             and ImageAnalysis.best_match(pieces[row][col].id, "L") == pieces[row][col - 1].id:
                #         continue
                # if row == self.rows - 1:
                #     if col == 0 and ImageAnalysis.best_match(pieces[row][col].id, "T") == pieces[row - 1][col].id \
                #             and ImageAnalysis.best_match(pieces[row][col].id, "R") == pieces[row][col + 1].id:
                #         continue
                #     if col < self.columns - 1 and ImageAnalysis.best_match(pieces[row][col].id, "R") == pieces[row][col + 1].id \
                #             and ImageAnalysis.best_match(pieces[row][col].id, "L") == pieces[row][col - 1].id \
                #             and ImageAnalysis.best_match(pieces[row][col].id, "T") == pieces[row - 1][col].id:
                #         continue
                #     if col == self.columns - 1 and ImageAnalysis.best_match(pieces[row][col].id, "L") == pieces[row][col - 1].id \
                #             and ImageAnalysis.best_match(pieces[row][col].id, "T") == pieces[row - 1][col].id:
                #         continue
                pieces[row][col] = empty_piece
        self.penalize_image = pieces
        return image_helpers.assemble_image([each.image for each in pieces.flatten()], self.rows, self.columns)

    def penalize(self):
        pieces = self.penalize_image
        rows = self.rows
        cols = self.columns
        for row in range(0, rows):
            for col in range(0, cols):
                if pieces[row][col].id == -1:
                    self.adjoin_assistance(row, col)
                    self.do_local_select(pieces, row, col)
                    # self.check_around(pieces, row, col)
        return image_helpers.assemble_image([each.image for each in pieces.flatten()], self.rows, self.columns)

    # def check_around(self, pieces, row, col):
    #     for r in range(row - 1, row + 2):
    #         for c in range(col - 1, col + 2):
    #             if pieces[r][c].id == -1:
    #                 return
    #
    #     for r in range(3):
    #         for c in range(2):
    #             self.set_down_lr(pieces[row - 1 + r][col - 1 + c], pieces[row - 1 + r][col + c])
    #     for r in range(2):
    #         for c in range(3):
    #             self.set_down_td(pieces[row - 1 + r][col - 1 + c], pieces[row + r][col - 1 + c])

    # reshaped_pieces = np.reshape(self.pieces, (self.rows, self.columns))
    # best_match_around = ImageAnalysis.best_match_table[reshaped_pieces[row][col].id]
    # top = reshaped_pieces[row - 1][col]
    # bottom = reshaped_pieces[row + 1][col]
    # left = reshaped_pieces[row][col - 1]
    # right = reshaped_pieces[row][col + 1]
    #
    # t_index = self.find_index(reshaped_pieces["T"], top.id)
    # d_index = self.find_index(reshaped_pieces["D"], bottom.id)
    # l_index = self.find_index(reshaped_pieces["L"], left.id)
    # r_index = self.find_index(reshaped_pieces["R"], right.id)
    #
    # reshaped_pieces[t_index] = (top.id, reshaped_pieces[t_index][1] / 2)
    # reshaped_pieces[d_index] = (bottom.id, reshaped_pieces[d_index][1] / 2)
    # reshaped_pieces[l_index] = (left.id, reshaped_pieces[l_index][1] / 2)
    # reshaped_pieces[r_index] = (right.id, reshaped_pieces[r_index][1] / 2)

    def set_down_lr(self, piece1, piece2):
        r_ = ImageAnalysis.best_match_table[piece1.id]["R"]
        l_ = ImageAnalysis.best_match_table[piece2.id]["L"]

        r2_index = self.find_index(r_, piece2.id)
        l1_index = self.find_index(l_, piece1.id)

        r_[r2_index] = (piece2.id, r_[r2_index][1] / 2)
        l_[l1_index] = (piece1.id, r_[l1_index][1] / 2)

        r_["L"].sort(key=lambda x: x[1])
        l_["R"].sort(key=lambda x: x[1])

    def set_down_td(self, piece1, piece2):
        d_ = ImageAnalysis.best_match_table[piece1.id]["D"]
        t_ = ImageAnalysis.best_match_table[piece2.id]["T"]

        d2_index = self.find_index(d_, piece2.id)
        t1_index = self.find_index(t_, piece1.id)

        d_[d2_index] = (piece2.id, d_[d2_index][1] / 2)
        t_[t1_index] = (piece1.id, t_[t1_index][1] / 2)

        d_.sort(key=lambda x: x[1])
        t_.sort(key=lambda x: x[1])

    # def check_around(self, pieces, row, col):
    #     if (row == 0 or row == self.rows - 1) or (col == 0 or col == self.columns - 1):
    #         return
    #     count = 0
    #     if pieces[row - 1][col].id != -1:
    #         count += 1
    #     if pieces[row + 1][col].id != -1:
    #         count += 1
    #     if pieces[row][col - 1].id != -1:
    #         count += 1
    #     if pieces[row][col + 1].id != -1:
    #         count += 1
    #     if count >= 4:
    #         self.find_best_matched_piece(row, col, count)

    # def find_best_matched_piece(self, row, col, count):
    #     pieces = np.reshape(self.pieces, (self.rows, self.columns))
    #     penalize_image = self.penalize_image
    #
    #     piece = penalize_image[row][col]
    #     best_match = ImageAnalysis.best_match_table[self.pieces[row * self.rows + col].id]
    #     top = self.pieces[(row - 1) * self.rows + col].id
    #     down = self.pieces[(row + 1) * self.rows + col].id
    #     left = self.pieces[row * self.rows + col - 1].id
    #     right = self.pieces[row * self.rows + col + 1].id
    #
    #     t_index = self.find_index(best_match["T"], top)
    #     d_index = self.find_index(best_match["D"], down)
    #     l_index = self.find_index(best_match["L"], left)
    #     r_index = self.find_index(best_match["R"], right)
    #
    #     # best_match["T"][t_index] = (top, best_match["T"][t_index][1] / 2)
    #     # best_match["D"][d_index] = (down, best_match["D"][d_index][1] / 2)
    #     # best_match["L"][l_index] = (left, best_match["L"][l_index][1] / 2)
    #     # best_match["R"][r_index] = (right, best_match["R"][r_index][1] / 2)
    #
    #     # around = []
    #     # if penalize_image[row - 1][col].id != -1:
    #     #     around.append(ImageAnalysis.best_match_table[pieces[row - 1][col].id]["D"])
    #     # if penalize_image[row + 1][col].id != -1:
    #     #     around.append(ImageAnalysis.best_match_table[pieces[row + 1][col].id]["T"])
    #     # if penalize_image[row][col - 1].id != -1:
    #     #     around.append(ImageAnalysis.best_match_table[pieces[row][col - 1].id]["R"])
    #     # if penalize_image[row][col + 1].id != -1:
    #     #     around.append(ImageAnalysis.best_match_table[pieces[row][col + 1].id]["L"])
    #
    #     # ele = {}
    #     # candidate = -1
    #     # for i in range(len(around[0])):
    #     #     for l in around:
    #     #         self.dict_count(ele, l[i][0])
    #     #     for l in around:
    #     #         if ele[l[i][0]] >= count:
    #     #             candidate = l[i][0]
    #     #             position = self.find_position(pieces, candidate)
    #     #             if position != None and self.penalize_image[position[0]][position[1]].id == -1:
    #     #                 break;
    #     #             else:
    #     #                 candidate = -1
    #     #
    #     #     if candidate != -1:
    #     #         break
    #     # if candidate == -1:
    #     #     return
    #     #
    #     # index = self._piece_mapping[candidate]
    #     # self.penalize_image[index // self.rows][index % self.columns], self.penalize_image[row][col] = \
    #     #     self.penalize_image[row][col], self.penalize_image[index // self.rows][index % self.columns]
    #     #
    #     # self.pieces[index], self.pieces[row * (self.rows - 1) + col] = self.pieces[row * (self.rows - 1) + col], \
    #     #                                                                self.pieces[index]
    #
    #     # 惩罚操作
    #     # candidate_four_oritation = ImageAnalysis.best_match_table[candidate]
    #     # l_index = self.find_index(candidate_four_oritation["L"], pieces[row][col - 1].id)
    #     # r_index = self.find_index(candidate_four_oritation["R"], pieces[row][col + 1].id)
    #     # d_index = self.find_index(candidate_four_oritation["D"], pieces[row + 1][col].id)
    #     # t_index = self.find_index(candidate_four_oritation["T"], pieces[row - 1][col].id)
    #
    #     # if penalize_image[row][col - 1].id != -1:
    #     #     candidate_four_oritation["L"][l_index] = pieces[row][col - 1].id,  candidate_four_oritation["L"][0][1] - 1
    #     # if penalize_image[row][col + 1].id != -1:
    #     #     candidate_four_oritation["R"][r_index] = pieces[row][col + 1].id,  candidate_four_oritation["R"][0][1] - 1
    #     # if penalize_image[row + 1][col].id != -1:
    #     #     candidate_four_oritation["D"][d_index] = pieces[row + 1][col].id,  candidate_four_oritation["D"][0][1] - 1
    #     # if penalize_image[row - 1][col].id != -1:
    #     #     candidate_four_oritation["T"][t_index] = pieces[row - 1][col].id,  candidate_four_oritation["T"][0][1] - 1
    #     #
    #     # candidate_four_oritation["L"].sort(key=lambda x: x[1])
    #     # candidate_four_oritation["R"].sort(key=lambda x: x[1])
    #     # candidate_four_oritation["D"].sort(key=lambda x: x[1])
    #     # candidate_four_oritation["T"].sort(key=lambda x: x[1])

    @staticmethod
    def find_index(dissimilarity_list, candidate):
        for i in range(len(dissimilarity_list)):
            if dissimilarity_list[i][0] == candidate:
                return i
        return len(dissimilarity_list)

    @staticmethod
    def get_value(dissimilarity_list, candidate):
        if candidate == -1:
            return 2
        for i in range(len(dissimilarity_list)):
            if dissimilarity_list[i][0] == candidate:
                return dissimilarity_list[i][1]
        return None

    @staticmethod
    def dict_count(ele, e):
        if e in ele:
            ele[e] += 1
        else:
            ele[e] = 1

    def find_position(self, pieces, candidate):
        for r in range(self.rows):
            for c in range(self.columns):
                if pieces[r][c].id == candidate:
                    return r, c
        return None

    def to_image(self):
        """Converts individual to showable image"""
        pieces = [piece.image for piece in self.pieces]
        return image_helpers.assemble_image(pieces, self.rows, self.columns)

    def edge(self, piece_id, orientation):
        edge_index = self._piece_mapping[piece_id]

        if (orientation == "T") and (edge_index >= self.columns):
            return self.pieces[edge_index - self.columns].id

        if (orientation == "R") and (edge_index % self.columns < self.columns - 1):
            return self.pieces[edge_index + 1].id

        if (orientation == "D") and (edge_index < (self.rows - 1) * self.columns):
            return self.pieces[edge_index + self.columns].id

        if (orientation == "L") and (edge_index % self.columns > 0):
            return self.pieces[edge_index - 1].id

    def count_around_best_fit(self, piece, index):
        pieces = self.pieces
        row = index // self.columns
        col = index % self.columns
        best_match_table = ImageAnalysis.best_match_table[piece.id]
        count = 0
        if row == 0:
            if col == 0:
                count = self.find_index(best_match_table["R"], pieces[index + 1].id) \
                        + self.find_index(best_match_table["D"], pieces[index + self.columns].id)
            elif col == self.columns - 1:
                count = self.find_index(best_match_table["L"], pieces[index - 1].id) \
                        + self.find_index(best_match_table["D"], pieces[index + self.columns].id)
            elif col < self.columns - 1:
                count = self.find_index(best_match_table["R"], pieces[index + 1].id) \
                        + self.find_index(best_match_table["D"], pieces[index + self.columns].id) \
                        + self.find_index(best_match_table["L"], pieces[index - 1].id)
        elif row == self.rows - 1:
            if col == 0:
                count = self.find_index(best_match_table["R"], pieces[index + 1].id) \
                        + self.find_index(best_match_table["T"], pieces[index - self.columns].id)
            elif col == self.columns - 1:
                count = self.find_index(best_match_table["L"], pieces[index - 1].id) \
                        + self.find_index(best_match_table["T"], pieces[index - self.columns].id)
            elif col < self.columns - 1:
                count = self.find_index(best_match_table["R"], pieces[index + 1].id) \
                        + self.find_index(best_match_table["T"], pieces[index - self.columns].id) \
                        + self.find_index(best_match_table["L"], pieces[index - 1].id)
        elif 0 < row < self.rows - 1:
            if col == 0:
                count = self.find_index(best_match_table["R"], pieces[index + 1].id) \
                        + self.find_index(best_match_table["T"], pieces[index - self.columns].id) \
                        + self.find_index(best_match_table["D"], pieces[index + self.columns].id)
            elif col == self.columns - 1:
                count = self.find_index(best_match_table["L"], pieces[index - 1].id) \
                        + self.find_index(best_match_table["T"], pieces[index - self.columns].id) \
                        + self.find_index(best_match_table["D"], pieces[index + self.columns].id)
            elif col < self.columns - 1:
                count = self.find_index(best_match_table["L"], pieces[index - 1].id) \
                        + self.find_index(best_match_table["R"], pieces[index + 1].id) \
                        + self.find_index(best_match_table["T"], pieces[index - self.columns].id) \
                        + self.find_index(best_match_table["D"], pieces[index + self.columns].id)
        return count

    def mutate(self):
        # # FIXME there still have promotion space, if i shuffle  which piece.id eqaul to -1 by random rather than single pick
        # old_fitness = self.fitness
        # new_fitness = None
        # for i in range(len(self.pieces)):
        #     old_fitness = self.fitness
        #     randint = random.randint(0, len(self.pieces) - 1)
        #     if i == randint:
        #         continue
        #     self.pieces[i], self.pieces[randint] = self.pieces[randint], self.pieces[i]
        #     self.clear_fitness()
        #     new_fitness = self.fitness
        #     if new_fitness < old_fitness:
        #         self.pieces[i], self.pieces[randint] = self.pieces[randint], self.pieces[i]
        #         self.clear_fitness()
        # print("\nnew_fitness %s, old_fitness %s" % (new_fitness, old_fitness), end="")
        for i in range(len(self.pieces)):
            randint = random.randint(0, len(self.pieces) - 1)
            if i == randint:
                continue
            i_piece = self.pieces[i]
            randint_piece = self.pieces[randint]
            old_fit_metric = self.count_around_best_fit(i_piece, i) + self.count_around_best_fit(randint_piece, randint)
            self.pieces[i], self.pieces[randint] = self.pieces[randint], self.pieces[i]
            new_fit_metric = self.count_around_best_fit(i_piece, randint) + self.count_around_best_fit(randint_piece, i)
            if new_fit_metric > old_fit_metric:
                self.pieces[i], self.pieces[randint] = self.pieces[randint], self.pieces[i]
            else:
                print("\nchange pieces new_index %s, old_index %s" % (randint, i))

    def do_local_select(self, pieces, r, c):
        rows = self.rows
        cols = self.columns
        old_fitness = self.fitness
        self.clear_fitness()
        best_fitness = old_fitness
        best_row = None
        best_col = None
        for row in range(0, rows):
            for col in range(0, cols):
                if pieces[row][col].id == -1 and (row != r or col != c):
                    self.pieces[row * cols + col], self.pieces[r * cols + c] = self.pieces[r * cols + c], self.pieces[
                        row * cols + col]
                    new_fitness = self.fitness
                    self.pieces[row * cols + col], self.pieces[r * cols + c] = self.pieces[r * cols + c], self.pieces[
                        row * cols + col]
                    if new_fitness > best_fitness:
                        best_row = row
                        best_col = col
                        best_fitness = new_fitness
                    # else:
                    self.clear_fitness()
        if best_row is not None and best_col is not None:
            self.pieces[best_row * cols + best_col], self.pieces[r * cols + c] = self.pieces[r * cols + c], self.pieces[
                best_row * cols + best_col]
        # print("old_fitness : %s, new_fitness : %s", (old_fitness, best_fitness))

    def like_best_match(self, row, col):
        cur_id = self.pieces[row * (self.rows - 1) + col].id
        match_table = ImageAnalysis.best_match_table[cur_id]
        count = 0
        if col > 0 and match_table["L"][0] == self.pieces[row * self.rows + col - 1].id:
            count += 1
        if col < self.columns - 1 and match_table["R"][0] == self.pieces[row * self.rows + col + 1].id:
            count += 1
        if row > 0 and match_table["T"][0] == self.pieces[(row - 1) * self.rows + col]:
            count += 1
        if row < self.rows - 1 and match_table["D"][0] == self.pieces[(row + 1) * self.rows + col].id:
            count += 1
        return count >= 3

    def adjoin_assistance(self, row, col):
        if row == 0 or row == self.rows - 1:
            return
        if col == 0 or col == self.columns - 1:
            return
        count = 0
        left = self.pieces[row * self.columns + col - 1]
        right = self.pieces[row * self.columns + col + 1]
        top = self.pieces[(row - 1) * self.columns + col]
        bottom = self.pieces[(row + 1) * self.columns + col]

        print("\n candidate position: row %s, col %s" % (row, col), end="")
        cur_id = self.pieces[row * self.columns + col].id
        if ImageAnalysis.best_match_table[left.id]["R"][0][0] == cur_id:
            count += 1
        if ImageAnalysis.best_match_table[right.id]["L"][0][0] == cur_id:
            count += 1
        if ImageAnalysis.best_match_table[top.id]["D"][0][0] == cur_id:
            count += 1
        if ImageAnalysis.best_match_table[bottom.id]["T"][0][0] == cur_id:
            count += 1
        if count < 3:
            return

        print("  change_position: row %s, col %s", (row, col))

        best_match = ImageAnalysis.best_match_table[cur_id]
        l_index = self.find_index(best_match["L"], left.id)
        r_index = self.find_index(best_match["R"], right.id)
        t_index = self.find_index(best_match["T"], top.id)
        d_index = self.find_index(best_match["D"], bottom.id)

        best_match["L"][l_index] = (left.id, best_match["L"][l_index][1] / 2)
        best_match["R"][r_index] = (right.id, best_match["R"][r_index][1] / 2)
        best_match["T"][t_index] = (top.id, best_match["T"][t_index][1] / 2)
        best_match["D"][d_index] = (bottom.id, best_match["D"][d_index][1] / 2)

        best_match["L"].sort(key=lambda x: x[1])
        best_match["R"].sort(key=lambda x: x[1])
        best_match["T"].sort(key=lambda x: x[1])
        best_match["D"].sort(key=lambda x: x[1])

        # if (cur_id, left.id) in ImageAnalysis.dissimilarity_measures:
        ImageAnalysis.dissimilarity_measures[(cur_id, left.id)]["LR"] = \
            ImageAnalysis.dissimilarity_measures[(cur_id, left.id)]["LR"] / 2
        # else :
        ImageAnalysis.dissimilarity_measures[(left.id, cur_id)]["LR"] = \
            ImageAnalysis.dissimilarity_measures[(left.id, cur_id)]["LR"] / 2

        # if (cur_id, right.id) in ImageAnalysis.dissimilarity_measures:
        ImageAnalysis.dissimilarity_measures[(cur_id, right.id)]["LR"] = \
            ImageAnalysis.dissimilarity_measures[(cur_id, right.id)]["LR"] / 2
        # else:
        ImageAnalysis.dissimilarity_measures[(right.id, cur_id)]["LR"] = \
            ImageAnalysis.dissimilarity_measures[(right.id, cur_id)]["LR"] / 2

        # if (cur_id, top.id) in ImageAnalysis.dissimilarity_measures:
        ImageAnalysis.dissimilarity_measures[(cur_id, top.id)]["TD"] = \
            ImageAnalysis.dissimilarity_measures[(cur_id, top.id)]["TD"] / 2
        # else:
        ImageAnalysis.dissimilarity_measures[(top.id, cur_id)]["TD"] = \
            ImageAnalysis.dissimilarity_measures[(top.id, cur_id)]["TD"] / 2

        # if (cur_id, bottom.id) in ImageAnalysis.dissimilarity_measures:
        ImageAnalysis.dissimilarity_measures[(cur_id, bottom.id)]["TD"] = \
            ImageAnalysis.dissimilarity_measures[(cur_id, bottom.id)]["TD"] / 2
        # else:
        ImageAnalysis.dissimilarity_measures[(bottom.id, cur_id)]["TD"] = \
            ImageAnalysis.dissimilarity_measures[(bottom.id, cur_id)]["TD"] / 2

    def shuffle_assembling(self):

        # {position: id}
        uncompleted_piece = {}
        for r in range(self.rows):
            for c in range(self.columns):
                if self.penalize_image[r][c].id == -1:
                    uncompleted_piece[(r, c)] = self.pieces[r * self.columns + c].id

        candidate_piece = uncompleted_piece.copy()

        # check_four_piece_around position
        four_piece_around = []
        for (k, v) in uncompleted_piece.items():
            if self.is_around_four(k[0], k[1]):
                four_piece_around.append(k)
        self.handle_four_piece_around(four_piece_around, uncompleted_piece, candidate_piece)
        while True:
            three_piece_around = []
            for (k, v) in uncompleted_piece.items():
                if self.is_around_three(k[0], k[1]):
                    three_piece_around.append(k)
            if len(three_piece_around) == 0:
                break
            self.handle_three_piece_around(three_piece_around, uncompleted_piece, candidate_piece)
        self.handle_uncompleted_piece(uncompleted_piece, candidate_piece)

        # 回溯
        for r in range(self.rows):
            for c in range(self.columns):
                if self.penalize_image[r][c].id != -1:
                    self.pieces[r * self.columns + c] = self.penalize_image[r][c]

    def is_around_four(self, r, c):
        if r == 0 or r == self.rows - 1 or c == 0 or c == self.columns - 1:
            return False
        penalize_image = self.penalize_image
        return penalize_image[r][c - 1].id is not -1 and penalize_image[r][c + 1].id is not -1 \
               and penalize_image[r - 1][c].id is not -1 and penalize_image[r + 1][c].id is not -1

    def handle_four_piece_around(self, four_piece_around, uncompleted_piece, candidate_piece):

        while len(four_piece_around) != 0:
            best_fitness = float("inf")
            best_position = None
            best_k = None
            for k in four_piece_around:
                for position, id in candidate_piece.items():
                    fitness = self.calculate_fitness(k, id)

                    if fitness < best_fitness:
                        best_fitness = fitness
                        best_k = k
                        best_position = position
            self.penalize_image[best_k[0]][best_k[1]] = self.pieces[best_position[0] * self.columns + best_position[1]]
            print("four piece %s, %s -> %s, %s , best_fitness %s" % (
                best_position[0], best_position[1], best_k[0], best_k[1], best_fitness))
            uncompleted_piece.pop(best_k)
            four_piece_around.remove(best_k)
            candidate_piece.pop(best_position)

    def is_around_three(self, r, c):
        if r == 0:
            if c == 0:
                return False
            if c == self.columns - 1:
                return False
            return self.penalize_image[r][c - 1].id != -1 and self.penalize_image[r][c + 1].id != -1 and \
                   self.penalize_image[r + 1][c].id != -1

        if r == self.rows - 1:
            if c == 0:
                return False
            if c == self.columns - 1:
                return False
            return self.penalize_image[r][c - 1].id != -1 and self.penalize_image[r][c + 1].id != -1 and \
                   self.penalize_image[r - 1][c].id != -1
        if c == 0:
            return self.penalize_image[r + 1][c].id != -1 and self.penalize_image[r][c + 1].id != -1 and \
                   self.penalize_image[r - 1][c].id != -1
        if c == self.columns - 1:
            return self.penalize_image[r + 1][c].id != -1 and self.penalize_image[r][c - 1].id != -1 and \
                   self.penalize_image[r - 1][c].id != -1

        count = 0
        if self.penalize_image[r + 1][c].id != -1:
            count += 1
        if self.penalize_image[r][c - 1].id != -1:
            count += 1
        if self.penalize_image[r - 1][c].id != -1:
            count += 1
        if self.penalize_image[r][c + 1].id != -1:
            count += 1
        if count >= 3:
            return True
        return False

    def handle_three_piece_around(self, three_piece_around, uncompleted_piece, candidate_piece):

        while len(three_piece_around) != 0:
            best_fitness = float("inf")
            best_position = None
            best_k = None
            for k in three_piece_around:
                for position, id in candidate_piece.items():
                    fitness = self.calculate_fitness(k, id)
                    if fitness < best_fitness:
                        best_fitness = fitness
                        best_k = k
                        best_position = position
            self.penalize_image[best_k[0]][best_k[1]] = self.pieces[best_position[0] * self.columns + best_position[1]]
            print("three piece %s, %s -> %s, %s , best_fitness %s" % (
                best_position[0], best_position[1], best_k[0], best_k[1], best_fitness))
            uncompleted_piece.pop(best_k)
            three_piece_around.remove(best_k)
            candidate_piece.pop(best_position)

    def calculate_fitness(self, position, id):
        r = position[0]
        c = position[1]
        if r == 0:
            if c == 0:
                return 4 + Individual.get_value(ImageAnalysis.best_match_table[id]["R"],
                                                self.penalize_image[position[0]][position[1] + 1].id) \
                       + Individual.get_value(ImageAnalysis.best_match_table[id]["D"],
                                              self.penalize_image[position[0] + 1][position[1]].id)
            if c == self.columns - 1:
                return 4 + Individual.get_value(ImageAnalysis.best_match_table[id]["L"],
                                                self.penalize_image[position[0]][position[1] - 1].id) \
                       + Individual.get_value(ImageAnalysis.best_match_table[id]["D"],
                                              self.penalize_image[position[0] + 1][position[1]].id)

            return 2 + Individual.get_value(ImageAnalysis.best_match_table[id]["L"],
                                            self.penalize_image[position[0]][position[1] - 1].id) \
                   + Individual.get_value(ImageAnalysis.best_match_table[id]["R"],
                                          self.penalize_image[position[0]][position[1] + 1].id) \
                   + Individual.get_value(ImageAnalysis.best_match_table[id]["D"],
                                          self.penalize_image[position[0] + 1][position[1]].id)
        if r == self.rows - 1:
            if c == 0:
                return 4 + Individual.get_value(ImageAnalysis.best_match_table[id]["R"],
                                                self.penalize_image[position[0]][position[1] + 1].id) \
                       + Individual.get_value(ImageAnalysis.best_match_table[id]["T"],
                                              self.penalize_image[position[0] - 1][position[1]].id)
            if c == self.columns - 1:
                return 4 + Individual.get_value(ImageAnalysis.best_match_table[id]["L"],
                                                self.penalize_image[position[0]][position[1] - 1].id) \
                       + Individual.get_value(ImageAnalysis.best_match_table[id]["T"],
                                              self.penalize_image[position[0] - 1][position[1]].id)

            return 2 + Individual.get_value(ImageAnalysis.best_match_table[id]["L"],
                                            self.penalize_image[position[0]][position[1] - 1].id) \
                   + Individual.get_value(ImageAnalysis.best_match_table[id]["R"],
                                          self.penalize_image[position[0]][position[1] + 1].id) \
                   + Individual.get_value(ImageAnalysis.best_match_table[id]["T"],
                                          self.penalize_image[position[0] - 1][position[1]].id)
        if c == 0:
            return 2 + Individual.get_value(ImageAnalysis.best_match_table[id]["R"],
                                            self.penalize_image[position[0]][position[1] + 1].id) \
                   + Individual.get_value(ImageAnalysis.best_match_table[id]["T"],
                                          self.penalize_image[position[0] - 1][position[1]].id) \
                   + Individual.get_value(ImageAnalysis.best_match_table[id]["D"],
                                          self.penalize_image[position[0] + 1][position[1]].id)
        if c == self.columns - 1:
            return 2 + Individual.get_value(ImageAnalysis.best_match_table[id]["L"],
                                            self.penalize_image[position[0]][position[1] - 1].id) \
                   + Individual.get_value(ImageAnalysis.best_match_table[id]["T"],
                                          self.penalize_image[position[0] - 1][position[1]].id) \
                   + Individual.get_value(ImageAnalysis.best_match_table[id]["D"],
                                          self.penalize_image[position[0] + 1][position[1]].id)
        return Individual.get_value(ImageAnalysis.best_match_table[id]["L"],
                                    self.penalize_image[position[0]][position[1] - 1].id) \
               + Individual.get_value(ImageAnalysis.best_match_table[id]["R"],
                                      self.penalize_image[position[0]][position[1] + 1].id) \
               + Individual.get_value(ImageAnalysis.best_match_table[id]["T"],
                                      self.penalize_image[position[0] - 1][position[1]].id) \
               + Individual.get_value(ImageAnalysis.best_match_table[id]["D"],
                                      self.penalize_image[position[0] + 1][position[1]].id)

    def handle_uncompleted_piece(self, uncompleted_piece, candidate_piece):
        while len(uncompleted_piece) != 0:
            duplicate = uncompleted_piece.copy()
            best_fitness = float("inf")
            best_position = None
            best_k = None
            for k, v in duplicate.items():
                for position, id in candidate_piece.items():
                    fitness = self.calculate_fitness(position, id)
                    if fitness < best_fitness:
                        best_k = k
                        best_fitness = fitness
                        best_position = position
            self.penalize_image[best_k[0]][best_k[1]] = self.pieces[best_position[0] * self.columns + best_position[1]]
            print("handle_uncompleted_piece %s, %s -> %s, %s , best_fitness %s" % (
                best_position[0], best_position[1], best_k[0], best_k[1], best_fitness))
            uncompleted_piece.pop(best_k)
            candidate_piece.pop(best_position)

    def manually_select(self):
        for i in range(len(self.pieces)):
            for j in range(len(self.pieces)):
                if i == j:
                    continue
                i_piece = self.pieces[i]
                j_piece = self.pieces[j]
                old_fit_metric = self.count_around_best_fit(i_piece, i) + self.count_around_best_fit(j_piece, j)
                self.pieces[i], self.pieces[j] = self.pieces[j], self.pieces[i]
                new_fit_metric = self.count_around_best_fit(i_piece, j) + self.count_around_best_fit(j_piece, i)
                if new_fit_metric >= old_fit_metric:
                    self.pieces[i], self.pieces[j] = self.pieces[j], self.pieces[i]
                else:
                    print("\nchange pieces new_index %s, old_index %s" % (j, i))
