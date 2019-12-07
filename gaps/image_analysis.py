from gaps.fitness import dissimilarity_measure
from gaps.progress_bar import print_progress


class ImageAnalysis(object):
    """Cache for dissimilarity measures of individuals

    Class have static lookup table where keys are Piece's id's.
    For each pair puzzle pieces there is a map with values representing
    dissimilarity measure between them. Each next generation have greater chance to use
    cached value instead of calculating measure again.

    Attributes:
        dissimilarity_measures  Dictionary with cached dissimilarity measures for puzzle pieces
        best_match_table        Dictionary with best matching piece for each edge and each piece

    """
    dissimilarity_measures = {}
    best_match_table = {}

    @classmethod
    def analyze_image(cls, pieces):
        # 图片分析
        for piece in pieces:
            # For each edge we keep best matches as a sorted list.
            # Edges with lower dissimilarity_measure have higher priority.
            # 保存每一块的每个方向匹配度最高的块 420 * 4
            cls.best_match_table[piece.id] = {
                # "T": [(piece.id,dissimilarity)],
                "T": [],
                "R": [],
                "D": [],
                "L": []
            }
            cls.max_match_d = -1
            cls.max_match_mgc = -1

        def update_best_match_table(first_piece, second_piece):
            # 保存每次计算出来的两个碎片的相似度
            # 记录每两块之间的相似度，加入到列表之中 (用颜色空间计算两个edge的距离)
            measure = dissimilarity_measure(first_piece, second_piece, orientation)

            if (measure[0] > cls.max_match_d):
                cls.max_match_d = measure[0]
            if (measure[1] > cls.max_match_mgc):
                cls.max_match_mgc = measure[1]

            # 保存每对的相似度
            cls.put_dissimilarity((first_piece.id, second_piece.id), orientation, measure)
            cls.best_match_table[second_piece.id][orientation[0]].append((first_piece.id, measure))
            cls.best_match_table[first_piece.id][orientation[1]].append((second_piece.id, measure))

        # Calculate dissimilarity measures and best matches for each piece.计算相似度
        iterations = len(pieces) - 1
        for first in range(iterations):
            print_progress(first, iterations - 1, prefix="=== Analyzing image:")
            for second in range(first + 1, len(pieces)):
                # 对每一种方向都计算相似度
                for orientation in ["LR", "TD"]:
                    update_best_match_table(pieces[first], pieces[second])
                    update_best_match_table(pieces[second], pieces[first])

        # normalize d_mgc
        for piece in pieces:
            for orientation in ["T", "L", "R", "D"]:
                d_mgc = cls.best_match_table[piece.id][orientation]
                new_value = []
                for piece_id, measure in d_mgc:
                    new_value.append((piece_id, measure[0] / cls.max_match_d + measure[1] / cls.max_match_mgc))
                    # new_value.append((piece_id, measure[0] / cls.max_match_d))
                cls.best_match_table[piece.id][orientation] = new_value

        # normalize dissimilarity
        for id1, id2 in cls.dissimilarity_measures:
            for orientation in cls.dissimilarity_measures[(id1, id2)]:
                v = cls.dissimilarity_measures[(id1, id2)][orientation]
                new_value = v[0] / cls.max_match_d + v[1] / cls.max_match_mgc
                # new_value = v[0] / cls.max_match_d
                cls.dissimilarity_measures[(id1, id2)][orientation] = new_value

        for piece in pieces:
            for orientation in ["T", "L", "R", "D"]:
                # 对每一块的每个方向的相似度进行排序
                cls.best_match_table[piece.id][orientation].sort(key=lambda x: x[1])

    @classmethod
    def put_dissimilarity(cls, ids, orientation, value):
        """Puts a new value in lookup table for given pieces

        :params ids:         Identfiers of puzzle pieces
        :params orientation: Orientation of puzzle pieces. Possible values are:
                             'LR' => 'Left-Right'
                             'TD' => 'Top-Down'
        :params value:       Value of dissimilarity measure

        Usage::

            >>> from gaps.image_analysis import ImageAnalysis
            >>> ImageAnalysis.put_dissimilarity([1, 2], "TD", 42)
        """
        if ids not in cls.dissimilarity_measures:
            cls.dissimilarity_measures[ids] = {}
        cls.dissimilarity_measures[ids][orientation] = value

    @classmethod
    def get_dissimilarity(cls, ids, orientation):
        """Returns previously cached dissimilarity measure for input pieces

        :params ids:         Identfiers of puzzle pieces
        :params orientation: Orientation of puzzle pieces. Possible values are:
                             'LR' => 'Left-Right'
                             'TD' => 'Top-Down'

        Us

            >>> from gaps.image_analysis import ImageAnalysis
            >>> ImageAnalysis.get_dissimilarity([1, 2], "TD")

        """
        return cls.dissimilarity_measures[ids][orientation]

    @classmethod
    def best_match(cls, piece, orientation):
        """"Returns best match piece for given piece and orientation"""
        return cls.best_match_table[piece][orientation][0][0]

    @classmethod
    def in_range(cls, source_id, oritation, target_id):
        v = cls.best_match_table[source_id][oritation]
        return target_id in (v[0][0], v[1][0], v[2][0])
