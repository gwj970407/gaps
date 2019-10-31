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


def dissimilarity_measure_advanced(first_piece, second_piece, orientation="LR"):
    # FIXME
    # def dissimilarity_measure(first_piece, second_piece, orientation="LR"):
    rows, columns, _ = first_piece.shape()
    color_difference = None

    # piece.shape 应该是三维的矩阵 第一维代表行，第二维代表列
    # 第三维度如果是彩色图像，则为3 灰度图像和黑白图像为1
    # | L | - | R |
    if orientation == "LR":
        # 如果是左右关系，则取左边的最右一列的三个通道减去右边的最左一列的三个通道
        color_difference = DLR(first_piece, second_piece) + DRL(first_piece, second_piece) \
                           + DGLR(first_piece, second_piece) + DGRL(first_piece, second_piece)

    # | T |
    #   |
    # | D |
    if orientation == "TD":
        # 如果是上下关系，则取上边的最下一行的三个通道减去下边的最上一列的三个通道
        color_difference = DTD(first_piece, second_piece) + DTD(first_piece, second_piece) \
                           + DGTD(first_piece, second_piece) + DGTD(first_piece, second_piece)

    return color_difference


def DGLR(first_piece, second_piece):
    rows, columns, _ = first_piece.shape()
    x = np.ndim(3)
    v = np.linalg.inv(np.var(VGiLLR(first_piece)))
    for i in range(rows):
        left = CSDGLR(first_piece, second_piece) - ECABGLR(first_piece, second_piece)
        right = left.T
        p = np.dot(left, v)
        x += np.dot(p, right)


def DGRL(first_piece, second_piece):
    rows, columns, _ = first_piece.shape()
    x = np.ndim(3)
    v = np.linalg.inv(np.var(VGiLLR(second_piece)))
    for i in range(rows):
        left = CSDGRL(first_piece, second_piece) - ECABGRL(first_piece, second_piece)
        right = left.T
        p = np.dot(left, v)
        x += np.dot(p, right)


def DGTD(first_piece, second_piece):
    rows, columns, _ = first_piece.shape()
    x = np.ndim(3)
    v = np.linalg.inv(np.var(VGiLTD(first_piece)))
    for i in range(rows):
        left = CSDGTD(first_piece, second_piece) - ECABGTD(first_piece, second_piece)
        right = left.T
        p = np.dot(left, v)
        x += np.dot(p, right)


def DGDT(first_piece, second_piece):
    rows, columns, _ = first_piece.shape()
    x = np.ndim(3)
    v = np.linalg.inv(np.var(VGiLTD(second_piece)))
    for i in range(rows):
        left = CSDGDT(first_piece, second_piece) - ECABGDT(first_piece, second_piece)
        right = left.T
        p = np.dot(left, v)
        x += np.dot(p, right)


def DLR(first_piece, second_piece):
    rows, columns, _ = first_piece.shape()
    x = np.ndim(3)
    v = np.linalg.inv(np.cov(ViLLR(first_piece)))
    for i in range(columns):
        left = CSDLR(first_piece, second_piece, i) - ECABLR(first_piece, second_piece, i)
        right = left.T
        p = np.dot(left, v)
        x += np.dot(np.linalg.inv(p), right)


def DTD(first_piece, second_piece):
    rows, columns, _ = first_piece.shape()
    x = np.ndim(3)
    v = np.linalg.inv(np.cov(ViLTD(first_piece)))
    for i in range(columns):
        left = CSDTD(first_piece, second_piece, i) - ECABTD(first_piece, second_piece, i)
        right = left.T
        p = np.dot(left, v)
        x += np.dot(np.linalg.inv(p), right)


def DRL(first_piece, second_piece):
    rows, columns, _ = first_piece.shape()
    x = np.ndim(3)
    v = np.linalg.inv(np.cov(ViLLR(second_piece)))
    for i in range(rows):
        left = CSDRL(first_piece, second_piece, i) - CSDRL(first_piece, second_piece, i)
        right = left.T
        p = np.dot(left, v)
        x += np.dot(np.linalg.inv(p), right)


def DDT(first_piece, second_piece):
    rows, columns, _ = first_piece.shape()
    x = np.ndim(3)
    v = np.linalg.inv(np.cov(ViLLR(second_piece)))
    for i in range(rows):
        left = CSDDT(first_piece, second_piece) - CSDDT(first_piece, second_piece)
        right = left.T
        p = np.dot(left, v)
        x += np.dot(np.linalg.inv(p), right)


def CSDLR(first_piece, second_piece, s):
    """
        color space distance LR
        ΛijLR
    """
    rows, columns, _ = first_piece.shape()
    return second_piece[s, 0, :] - first_piece[s, columns - 1, :]


def CSDTD(first_piece, second_piece, s):
    """
        color space distance LR
        ΛijLR
    """
    rows, columns, _ = first_piece.shape()
    return second_piece[0, s, :] - first_piece[columns - 1, s, :]


def CSDGLR(first_piece, second_piece, s):
    """
        color space distance LR
        Λ'ijLR
    """
    rows, columns, _ = first_piece.shape()
    return epxlD(first_piece, s, 0) - epxlD(second_piece, s, columns - 1)


def CSDGTD(first_piece, second_piece, s):
    """
        color space distance LR
        Λ'ijLR
    """
    rows, columns, _ = first_piece.shape()
    return epxlH(first_piece, 0, s) - epxlH(second_piece, columns - 1, s)


def CSDRL(first_piece, second_piece, s):
    """
        color space distance RL
        ΛijRL
    """
    rows, columns, _ = first_piece.shape()
    return first_piece[s, columns - 1, :] - second_piece[s, 0, :]


def CSDDT(first_piece, second_piece, s):
    """
        color space distance RL
        ΛijRL
    """
    rows, columns, _ = first_piece.shape()
    return first_piece[columns - 1, s, :] - second_piece[0, s, :]


def CSDGRL(first_piece, second_piece, s):
    """
        color space distance LR
        Λ'ijLR
    """
    rows, columns, _ = first_piece.shape()
    return epxlD(first_piece, s, columns - 1) - epxlD(second_piece, s, 0)


def CSDGDT(first_piece, second_piece, s):
    """
        color space distance LR
        Λ'ijLR
    """
    rows, columns, _ = first_piece.shape()
    return epxlH(first_piece, columns - 1, s) - epxlH(second_piece, 0, s)


def ECABLR(first_piece, second_piece, s):
    """
        expected change across the boundary of the two piece
    """
    rows, columns, _ = first_piece.shape()
    value = 0.5 * (first_piece[s, columns - 1, :] - first_piece[s, columns - 2, :]
                   + second_piece[s, 1, :] - second_piece[s, 0, :])
    return value


def ECABTD(first_piece, second_piece, s):
    """
        expected change across the boundary of the two piece
    """
    rows, columns, _ = first_piece.shape()
    value = 0.5 * (first_piece[columns - 1, s, :] - first_piece[columns - 2, s, :]
                   + second_piece[1, s, :] - second_piece[0, s, :])
    return value


def ECABGLR(first_piece, second_piece, s):
    """
        expected change across the boundary of the two piece
    """
    rows, columns, _ = first_piece.shape()
    value = 0.5 * (epxlD(first_piece, s, columns - 1) - epxlD(first_piece, s, columns - 2)
                   + epxlD(second_piece, s, 1) - epxlD(second_piece, s, 0))
    return value


def ECABGTD(first_piece, second_piece, s):
    """
        expected change across the boundary of the two piece
    """
    rows, columns, _ = first_piece.shape()
    value = 0.5 * (epxlH(first_piece, columns - 1, s) - epxlH(first_piece, columns - 2, s)
                   + epxlH(second_piece, 1, s) - epxlH(second_piece, 0, s))
    return value


def ECABRL(first_piece, second_piece, s):
    """
        expected change across the boundary of the two piece
    """
    rows, columns, _ = first_piece.shape()
    value = 0.5 * (first_piece[s, columns - 2, :] - first_piece[s, columns - 1, :]
                   + second_piece[s, 0, :] - second_piece[s, 1, :])
    return value


def ECABDT(first_piece, second_piece, s):
    """
        expected change across the boundary of the two piece
    """
    rows, columns, _ = first_piece.shape()
    value = 0.5 * (first_piece[columns - 2, s, :] - first_piece[columns - 1, s, :]
                   + second_piece[0, s, :] - second_piece[1, s, :])
    return value


def ECABGRL(first_piece, second_piece, s):
    """
        expected change across the boundary of the two piece
    """
    rows, columns, _ = first_piece.shape()
    value = 0.5 * (epxlD(first_piece, s, columns - 2) - epxlD(first_piece, s, columns - 1)
                   + epxlD(second_piece, s, 0) - epxlD(second_piece, s, 1))
    return value


def ECABGDT(first_piece, second_piece, s):
    """
        expected change across the boundary of the two piece
    """
    rows, columns, _ = first_piece.shape()
    value = 0.5 * (epxlH(first_piece, columns - 2, s) - epxlH(first_piece, columns - 1, s)
                   + epxlH(second_piece, 0, s) - epxlH(second_piece, 1, s))
    return value


def ViLLR(piece):
    """
     ViL calculated from samples,{xi(s,S)−xi(s,S−1))∣s=1,2,...S}.
    """
    rows, columns, _ = piece.shape()
    result = []
    for i in range(rows):
        result.append(piece[i, columns - 1, :] - piece[i, columns - 2, :])
    return np.array(result)


def ViLTD(piece):
    """
     ViL calculated from samples,{xi(s,S)−xi(s,S−1))∣s=1,2,...S}.
    """
    rows, columns, _ = piece.shape()
    result = []
    for i in range(columns):
        result.append(piece[rows - 1, i, :] - piece[rows - 2, i, :])
    return np.array(result)


def VGiLLR(piece):
    """
     ViL calculated from samples,{xi(s,S)−xi(s,S−1))∣s=1,2,...S}.
    """
    rows, columns, _ = piece.shape()
    result = []
    for i in range(1, rows):
        result.append(epxlD(piece, i, columns - 1) - epxlD(piece, i, columns - 2))
    return np.array(result)


def VGiLTD(piece, s):
    """
     ViL calculated from samples,{xi(s,S)−xi(s,S−1))∣s=1,2,...S}.
    """
    rows, columns, _ = piece.shape()
    result = []
    for i in range(1, columns):
        result.append(epxlH(piece, rows - 1, s) - epxlH(piece, rows - 2, s))
    return np.array(result)


def epxlD(piece, x, y):
    return piece[x, y, :] - piece[x - 1, y, :]


def epxlH(piece, x, y):
    return piece[x, y, :] - piece[x, y - 1, :]
