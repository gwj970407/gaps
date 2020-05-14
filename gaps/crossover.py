import random
import heapq

from gaps.image_analysis import ImageAnalysis
from gaps.individual import Individual
import numpy as np

SHARED_PIECE_PRIORITY = -10
BUDDY_PIECE_PRIORITY = -1


# SHARED_PIECE_PRIORITY = -1
# BUDDY_PIECE_PRIORITY = -10


class Crossover(object):

    def __init__(self, first_parent, second_parent):
        self._parents = (first_parent, second_parent)
        self._pieces_length = len(first_parent.pieces)
        self._child_rows = first_parent.rows
        self._child_columns = first_parent.columns

        # Borders of growing kernel
        self._min_row = 0
        self._max_row = 0
        self._min_column = 0
        self._max_column = 0

        self._kernel = {}
        # 占位符，表示某个位置是否已经有图片了
        self._taken_positions = set()

        # Priority queue
        self._shared_candidate_pieces = []
        self._buddy_candidate_pieces = []
        self._best_match_candidate_pieces = []

    def child(self):
        pieces = [None] * self._pieces_length

        for piece, (row, column) in self._kernel.items():
            index = (row - self._min_row) * self._child_columns + (column - self._min_column)
            pieces[index] = self._parents[0].piece_by_id(piece)

        individual = Individual(pieces, self._child_rows, self._child_columns, shuffle=False)
        return individual

    def run(self):
        self._initialize_kernel()

        directions = ["L", "R", "T", "D"]
        # np.random.shuffle(directions)
        while len(self._taken_positions) < self._pieces_length:

            selected = False
            np.random.shuffle(self._shared_candidate_pieces)
            while len(self._shared_candidate_pieces) > 0:
                (position, piece_id), relative_piece = self._shared_candidate_pieces.pop()
                if position in self._taken_positions:
                    continue
                if piece_id in self._kernel:
                    self.add_piece_candidate(relative_piece[0], relative_piece[1], position)
                else:
                    selected = True
                    self._put_piece_to_kernel(piece_id, position)
                    break
            if selected:
                continue
            np.random.shuffle(self._buddy_candidate_pieces)
            while len(self._buddy_candidate_pieces) > 0:
                (position, piece_id), relative_piece = self._buddy_candidate_pieces.pop()
                if position in self._taken_positions:
                    continue
                if piece_id in self._kernel:
                    self.add_piece_candidate(relative_piece[0], relative_piece[1], position)
                else:
                    selected = True
                    self._put_piece_to_kernel(piece_id, position)
                    break
            if selected:
                continue
            np.random.shuffle(self._best_match_candidate_pieces)
            while len(self._best_match_candidate_pieces) > 0:
                (position, piece_id), relative_piece = self._best_match_candidate_pieces.pop()
                if position in self._taken_positions:
                    continue
                if piece_id in self._kernel:
                    self.add_piece_candidate(relative_piece[0], relative_piece[1], position)
                else:
                    self._put_piece_to_kernel(piece_id, position)
                    break
            # # 弹出一个最优先级的候选快，共享区域最小，互为最优匹配为次之，普通的最优相似度最慢pop出
            # _, (position, piece_id), relative_piece = heapq.heappop(self._candidate_pieces)
            #
            # if position in self._taken_positions:
            #     continue
            #
            # # If piece is already placed, find new piece candidate and put it back to
            # # priority queue
            # if piece_id in self._kernel:
            #     # relative_piece[0] 代表piece_id, [1]代表方向，寻找这个位置这个方向的新最优匹配块
            #     self.add_piece_candidate(relative_piece[0], relative_piece[1], position)
            #     continue
            #
            # # 把这一块放入拼图中
            # self._put_piece_to_kernel(piece_id, position)

    def _initialize_kernel(self):
        # 取左父亲图片的随机一个piece作为初始根piece
        root_piece = self._parents[0].pieces[int(random.uniform(0, self._pieces_length))]
        self._put_piece_to_kernel(root_piece.id, (0, 0))

    def _put_piece_to_kernel(self, piece_id, position):
        """
        将这一块各个位置的最优匹配块都添加到候选块中
        """
        self._kernel[piece_id] = position
        self._taken_positions.add(position)
        self._update_candidate_pieces(piece_id, position)

    def _update_candidate_pieces(self, piece_id, position):
        available_boundaries = self._available_boundaries(position)

        for orientation, position in available_boundaries:
            # 把这个块各个方向上的最优匹配块加入候选块中
            self.add_piece_candidate(piece_id, orientation, position)

    # FIXME 这里无法将单边相似的块区分出来，需要重新设计！！！
    def add_piece_candidate(self, piece_id, orientation, position):
        # 判断这一块在父母中是否有同样的共享块，（这个方向上的块一样）
        shared_piece = self._get_shared_piece(piece_id, orientation)
        if self._is_valid_piece(shared_piece):
            self._add_shared_piece_candidate(shared_piece, position, (piece_id, orientation))
            return
        # 判断这一块是否在父母中任意一方中有最有匹配块
        buddy_piece = self._get_buddy_piece(piece_id, orientation)
        if self._is_valid_piece(buddy_piece):
            self._add_buddy_piece_candidate(buddy_piece, position, (piece_id, orientation))
            return
        # 取得这一块的最优匹配块， 优先级为相似度
        best_match_piece, priority = self._get_best_match_piece(piece_id, orientation)
        if self._is_valid_piece(best_match_piece):
            self._add_best_match_piece_candidate(best_match_piece, position, priority, (piece_id, orientation))
            return

    def _get_shared_piece(self, piece_id, orientation):
        first_parent, second_parent = self._parents
        first_parent_edge = first_parent.edge(piece_id, orientation)
        second_parent_edge = second_parent.edge(piece_id, orientation)

        if first_parent_edge == second_parent_edge:
            return first_parent_edge

    def _get_buddy_piece(self, piece_id, orientation):
        first_buddy = ImageAnalysis.best_match(piece_id, orientation)
        second_buddy = ImageAnalysis.best_match(first_buddy, complementary_orientation(orientation))

        # 如果两块互相为匹配度最高的块，则表示很有可能是相邻块
        if second_buddy == piece_id:
            for edge in [parent.edge(piece_id, orientation) for parent in self._parents]:
                if edge == first_buddy:
                    # 如果父母中有任意一方有最佳匹配块，则返回
                    return edge

    def _get_best_match_piece(self, piece_id, orientation):
        # 根据取得piece_id取得当前方向上相似度最高的碎片，如果已存在于kernel，则取下一个
        for piece, dissimilarity_measure in ImageAnalysis.best_match_table[piece_id][orientation]:
            if self._is_valid_piece(piece):
                return piece, dissimilarity_measure

    def _add_shared_piece_candidate(self, piece_id, position, relative_piece):
        piece_candidate = ((position, piece_id), relative_piece)
        self._shared_candidate_pieces.append(piece_candidate)
        # heapq.heappush(self._candidate_pieces, piece_candidate)

    def _add_buddy_piece_candidate(self, piece_id, position, relative_piece):
        piece_candidate = ((position, piece_id), relative_piece)
        self._buddy_candidate_pieces.append(piece_candidate)
        # heapq.heappush(self._candidate_pieces, piece_candidate)

    def _add_best_match_piece_candidate(self, piece_id, position, priority, relative_piece):
        piece_candidate = ((position, piece_id), relative_piece)
        self._best_match_candidate_pieces.append(piece_candidate)
        # heapq.heappush(self._candidate_pieces, piece_candidate)

    def _available_boundaries(self, row_and_column):
        (row, column) = row_and_column
        boundaries = []

        if not self._is_kernel_full():
            positions = {
                "T": (row - 1, column),
                "R": (row, column + 1),
                "D": (row + 1, column),
                "L": (row, column - 1)
            }

            for orientation, position in positions.items():
                # 如果这个位置还没有碎片，并且这个position在图片范围内(当前碎片未超过边界)
                if position not in self._taken_positions and self._is_in_range(position):
                    self._update_kernel_boundaries(position)
                    boundaries.append((orientation, position))

        return boundaries

    def _is_kernel_full(self):
        return len(self._kernel) == self._pieces_length

    def _is_in_range(self, row_and_column):
        (row, column) = row_and_column
        return self._is_row_in_range(row) and self._is_column_in_range(column)

    def _is_row_in_range(self, row):
        current_rows = abs(min(self._min_row, row)) + abs(max(self._max_row, row))
        return current_rows < self._child_rows

    def _is_column_in_range(self, column):
        current_columns = abs(min(self._min_column, column)) + abs(max(self._max_column, column))
        return current_columns < self._child_columns

    def _update_kernel_boundaries(self, row_and_column):
        (row, column) = row_and_column
        self._min_row = min(self._min_row, row)
        self._max_row = max(self._max_row, row)
        self._min_column = min(self._min_column, column)
        self._max_column = max(self._max_column, column)

    def _is_valid_piece(self, piece_id):
        return piece_id is not None and piece_id not in self._kernel


# 取得传入方向的互补方向
def complementary_orientation(orientation):
    return {
        "T": "D",
        "D": "T",
        "R": "L",
        "L": "R"
    }.get(orientation, None)
