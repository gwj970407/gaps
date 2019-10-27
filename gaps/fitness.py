import numpy as np


def dissimilarity_measure(first_piece, second_piece, orientation="LR"):
    """Calculates color difference over all neighboring pixels over all color channels.

    The dissimilarity measure relies on the premise that adjacent jigsaw pieces in the original image tend to share
    similar colors along their abutting edges, i.e., the sum (over all neighboring pixels) of squared color differences
    (over all three color bands) should be minimal. Let pieces pi , pj be represented in normalized L*a*b*
    space by corresponding W x W x 3 matrices, where W is the height/width of each piece (in pixels).

    :params first_piece:  First input piece for calculation.
    :params second_piece: Second input piece for calculation.
    :params orientation:  How input pieces are oriented.

                          LR => 'Left - Right'
                          TD => 'Top - Down'

    Usage::

        >>> from gaps.fitness import dissimilarity_measure
        >>> from gaps.piece import Piece
        >>> p1, p2 = Piece(), Piece()
        >>> dissimilarity_measure(p1, p2, orientation="TD")

    """
    rows, columns, _ = first_piece.shape()
    color_difference = None


    # piece.shape 应该是三维的矩阵 第一维代表行，第二维代表列
    # 第三维度如果是彩色图像，则为3 灰度图像和黑白图像为1
    # | L | - | R |
    if orientation == "LR":
        # 如果是左右关系，则取左边的最右一列的三个通道减去右边的最左一列的三个通道
        color_difference = first_piece[:rows, columns - 1, :] - second_piece[:rows, 0, :]

    # | T |
    #   |
    # | D |
    if orientation == "TD":
        # 如果是上下关系，则取上边的最下一行的三个通道减去下边的最上一列的三个通道
        color_difference = first_piece[rows - 1, :columns, :] - second_piece[0, :columns, :]

    # 先归一化，再利用np计算每个通道距离的平方
    squared_color_difference = np.power(color_difference / 255.0, 2)
    # 每个通道距离平方和相加就是颜色空间距离（没有开平方）
    color_difference_per_row = np.sum(squared_color_difference, axis=1)
    # 每个像素点的颜色空间距离相加
    total_difference = np.sum(color_difference_per_row, axis=0)

    # 对结果开方
    value = np.sqrt(total_difference)

    return value
