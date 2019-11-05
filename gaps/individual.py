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

    def piece_size(self):
        """Returns single piece size"""
        return self.pieces[0].size

    def piece_by_id(self, identifier):
        """"Return specific piece from individual"""
        return self.pieces[self._piece_mapping[identifier]]

    def best_adjoin(self, piece_size):
        pieces = np.reshape(self.pieces, (self.rows, self.columns))
        empty_image = np.zeros((piece_size, piece_size, pieces[0][0].shape()[2]))
        empty_piece = Piece(empty_image, 0)
        for row in range(self.rows):
            for col in range(self.columns):
                if row == 0:
                    if col == 0 and ImageAnalysis.best_match(pieces[row][col].id, "D") == pieces[row + 1][col].id \
                            and ImageAnalysis.best_match(pieces[row][col].id, "R") == pieces[row][col + 1].id:
                        continue
                    if col < self.columns - 1 and ImageAnalysis.best_match(pieces[row][col].id, "D") == pieces[row + 1][col].id \
                            and ImageAnalysis.best_match(pieces[row][col].id, "R") == pieces[row][col + 1].id \
                            and ImageAnalysis.best_match(pieces[row][col].id, "L") == pieces[row][col - 1].id:
                        continue
                    if col == self.columns - 1 and ImageAnalysis.best_match(pieces[row][col].id, "D") == pieces[row + 1][col].id \
                            and ImageAnalysis.best_match(pieces[row][col].id, "L") == pieces[row][col - 1].id:
                        continue
                if 0 < row < self.rows - 1:
                    if col == 0 and ImageAnalysis.best_match(pieces[row][col].id, "D") == pieces[row + 1][col].id \
                            and ImageAnalysis.best_match(pieces[row][col].id, "T") == pieces[row - 1][col].id \
                            and ImageAnalysis.best_match(pieces[row][col].id, "R") == pieces[row][col + 1].id:
                        continue
                    if col < self.columns - 1 and ImageAnalysis.best_match(pieces[row][col].id, "T") == pieces[row - 1][col].id \
                            and ImageAnalysis.best_match(pieces[row][col].id, "D") == pieces[row + 1][col].id \
                            and ImageAnalysis.best_match(pieces[row][col].id, "L") == pieces[row][col - 1].id \
                            and ImageAnalysis.best_match(pieces[row][col].id, "R") == pieces[row][col + 1].id:
                        continue
                    if col == self.columns - 1 and ImageAnalysis.best_match(pieces[row][col].id, "T") == pieces[row - 1][col].id \
                            and ImageAnalysis.best_match(pieces[row][col].id, "D") == pieces[row + 1][col].id \
                            and ImageAnalysis.best_match(pieces[row][col].id, "L") == pieces[row][col - 1].id:
                        continue
                if row == self.rows - 1:
                    if col == 0 and ImageAnalysis.best_match(pieces[row][col].id, "T") == pieces[row - 1][col].id \
                            and ImageAnalysis.best_match(pieces[row][col].id, "R") == pieces[row][col + 1].id:
                        continue
                    if col < self.columns - 1 and ImageAnalysis.best_match(pieces[row][col].id, "R") == pieces[row][col + 1].id \
                            and ImageAnalysis.best_match(pieces[row][col].id, "L") == pieces[row][col - 1].id \
                            and ImageAnalysis.best_match(pieces[row][col].id, "T") == pieces[row - 1][col].id:
                        continue
                    if col == self.columns - 1 and ImageAnalysis.best_match(pieces[row][col].id, "L") == pieces[row][col - 1].id \
                            and ImageAnalysis.best_match(pieces[row][col].id, "T") == pieces[row - 1][col].id:
                        continue
                pieces[row][col] = empty_piece
        return image_helpers.assemble_image([each.image for each in pieces.flatten()], self.rows, self.columns)

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
