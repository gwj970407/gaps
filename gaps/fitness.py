import numpy as np

VGiL_dict = {}
ViL_dict = {}


def dissimilarity_measure0(first_piece, second_piece, orientation="LR"):
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


# def dissimilarity_measure_advanced(first_piece, second_piece, orientation="LR"):
def dissimilarity_measure(first_piece, second_piece, orientation="LR"):
    rows, columns, _ = first_piece.shape()
    color_difference = None

    # piece.shape 应该是三维的矩阵 第一维代表行，第二维代表列
    # 第三维度如果是彩色图像，则为3 灰度图像和黑白图像为1
    # | L | - | R |
    if orientation == "LR":
        # 如果是左右关系，则取左边的最右一列的三个通道减去右边的最左一列的三个通道
        color_difference = D(first_piece, second_piece, 'L') + D(first_piece, second_piece, 'R'), \
                           DG(first_piece, second_piece, 'L') + DG(first_piece, second_piece, 'R')
        # color_difference = D(first_piece, second_piece, 'L') + D(first_piece, second_piece, 'R') \
        #                    + DG(first_piece, second_piece, 'L') + DG(first_piece, second_piece, 'R')
        # v = D(first_piece, second_piece, 'L') + D(first_piece, second_piece, 'R')
        # v2 = DG(first_piece, second_piece, 'L') + DG(first_piece, second_piece, 'R')
        # print("v= %s", v)
        # print("v2= %s", v2)

    # | T |
    #   |
    # | D |
    if orientation == "TD":
        # 如果是上下关系，则取上边的最下一行的三个通道减去下边的最上一列的三个通道
        color_difference = D(first_piece, second_piece, 'T') + D(first_piece, second_piece, 'D'), \
                           DG(first_piece, second_piece, 'T') + DG(first_piece, second_piece, 'D')

    return color_difference


def get_VGiL_inversion(*args):
    if args in VGiL_dict:
        return VGiL_dict[args]
    v = np.linalg.pinv(np.cov(VGiL(args[0], args[1])))
    VGiL_dict[args] = v
    return v


def DG(first_piece, second_piece, position):
    rows, columns, _ = first_piece.shape()
    res = []
    if position == 'L':
        v = get_VGiL_inversion(first_piece, 'L')
        for i in range(rows):
            left = 1.5 * epxl(second_piece, i, 0, 'D') - 1.5 * epxl(first_piece, i, columns - 1, 'D') + 0.5 * epxl(
                first_piece, i, columns - 2, 'D') - 0.5 * epxl(second_piece, i, 1, 'D')
            # left = CSDG(first_piece, second_piece, i, 'L') - ECABG(first_piece, second_piece, i, 'L')
            right = left.T
            res.append(np.dot(np.dot(left, v), right))
    elif position == 'R':
        v = get_VGiL_inversion(second_piece, 'R')
        for i in range(rows):
            left = 1.5 * epxl(first_piece, i, columns - 1, 'D') - 1.5 * epxl(second_piece, i, 0, 'D') - 0.5 * epxl(
                first_piece, i, columns - 2, 'D') + 0.5 * epxl(second_piece, i, 1, 'D')
            # left = CSDG(first_piece, second_piece, i, 'R') - ECABG(first_piece, second_piece, i, 'R')
            right = left.T
            res.append(np.dot(np.dot(left, v), right))
    elif position == 'T':
        v = get_VGiL_inversion(first_piece, 'T')
        for i in range(columns):
            left = 1.5 * epxl(second_piece, 0, i, 'H') - 1.5 * epxl(first_piece, columns - 1, i, 'H') + 0.5 * epxl(
                first_piece, columns - 2, i, 'H') - 0.5 * epxl(second_piece, 1, i, 'H')
            # left = CSDG(first_piece, second_piece, i, 'T') - ECABG(first_piece, second_piece, i, 'T')
            right = left.T
            res.append(np.dot(np.dot(left, v), right))
    elif position == 'D':
        v = get_VGiL_inversion(second_piece, 'D')
        for i in range(columns):
            left = 1.5 * epxl(first_piece, columns - 1, i, 'H') - 1.5 * epxl(second_piece, 0, i, 'H') - 0.5 * epxl(
                first_piece, columns - 2, i, 'H') + 0.5 * epxl(second_piece, 1, i, 'H')
            # left = CSDG(first_piece, second_piece, i, 'D') - ECABG(first_piece, second_piece, i, 'D')
            right = left.T
            res.append(np.dot(np.dot(left, v), right))
    return np.sum((np.array(res)))


def get_ViL_inversion(*args):
    if args in ViL_dict:
        return ViL_dict[args]
    v = np.linalg.pinv(np.cov(ViL(args[0], args[1])))
    ViL_dict[args] = v
    return v


def D(first_piece, second_piece, position):
    rows, columns, _ = first_piece.shape()
    # res = []
    left = None
    if position == 'L':
        left = second_piece[:, 0, :] - first_piece[:, columns - 1, :] + (0.5 * second_piece[:, 1, :] - 0.5 * first_piece[:, columns - 1, :])
        # left = 1.5 * second_piece[:, 0, :] - 1.5 * first_piece[:, columns - 1, :] + 0.5 * first_piece[:, columns - 2, :] - 0.5 * second_piece[:, 1, :]
        # size = 3 * 3
        # v = get_ViL_inversion(first_piece, 'L')
        # for i in range(rows):
        #     left = 1.5 * second_piece[i, 0, :] - 1.5 * first_piece[i, columns - 1, :] + 0.5 * first_piece[i, columns - 2, :] - 0.5 * second_piece[i, 1, :]
        # left = CSD(first_piece, second_piece, i, 'L') - ECAB(first_piece, second_piece, i, 'L')
        # right = left.T
        # res.append(np.dot(np.dot(left, v), right))
        # res.append(np.power(left, 2))
    elif position == 'R':
        left = first_piece[:, columns - 1, :] -  second_piece[:, 0, :] + (0.5 * first_piece[:, columns - 1, :] - 0.5 * second_piece[:, 1, :])
        # left = 1.5 * first_piece[:, columns - 1, :] - 1.5 * second_piece[:, 0, :] - 0.5 * first_piece[:, columns - 2, :] + 0.5 * second_piece[:, 1, :]
        # v = get_ViL_inversion(second_piece, 'R')
        # for i in range(rows):
        #     left = 1.5 * first_piece[i, columns - 1, :] - 1.5 * second_piece[i, 0, :] - 0.5 * first_piece[i, columns - 2, :] + 0.5 * second_piece[i, 1, :]
        # left = CSD(first_piece, second_piece, i, 'R') - ECAB(first_piece, second_piece, i, 'R')
        # right = left.T
        # res.append(np.dot(np.dot(left, v), right))
        # res.append(np.power(left, 2))
    elif position == 'T':
        left = second_piece[0, :, :] - first_piece[columns - 1, :, :] + (0.5 * second_piece[1, :, :] - 0.5 * first_piece[columns - 1, :, :])
        # v = get_ViL_inversion(first_piece, 'T')
        # for i in range(columns):
        #     left = 1.5 * second_piece[0, i, :] - 1.5 * first_piece[columns - 1, i, :] + 0.5 * first_piece[columns - 2, i, :] - 0.5 * second_piece[1, i, :]
        # left = CSD(first_piece, second_piece, i, 'T') - ECAB(first_piece, second_piece, i, 'T')
        # right = left.T
        # res.append(np.dot(np.dot(left, v), right))
        # res.append(np.power(left, 2))
    elif position == 'D':
        left = first_piece[columns - 1, :, :] - second_piece[0, :, :] + (0.5 * first_piece[columns - 1, :, :] - 0.5 * second_piece[1, :, :])
        # v = get_ViL_inversion(second_piece, 'D')
        # for i in range(rows):
        #     left = 1.5 * first_piece[columns - 1, i, :] - 1.5 * second_piece[0, i, :] - 0.5 * first_piece[columns - 2, i, :] + 0.5 * second_piece[1, i, :]
        # left = CSD(first_piece, second_piece, i, 'D') - ECAB(first_piece, second_piece, i, 'D')
        # right = left.T
        # res.append(np.dot(np.dot(left, v), right))
        # res.append(np.power(left, 2))
    return np.sum(np.power(np.array(left), 2))


def CSD(first_piece, second_piece, s, position):
    """
        color space distance LR
        ΛijLR
    """
    rows, columns, _ = first_piece.shape()
    if position == 'L':
        return second_piece[s, 0, :] - first_piece[s, columns - 1, :]
    elif position == 'R':
        return first_piece[s, columns - 1, :] - second_piece[s, 0, :]
    elif position == 'T':
        return second_piece[0, s, :] - first_piece[columns - 1, s, :]
    elif position == 'D':
        return first_piece[columns - 1, s, :] - second_piece[0, s, :]


def CSDG(first_piece, second_piece, s, position):
    """
        color space distance gradients
        Λ'ijLR
    """
    rows, columns, _ = first_piece.shape()
    if position == 'L':
        return epxl(second_piece, s, 0, 'D') - epxl(first_piece, s, columns - 1, 'D')
    elif position == 'R':
        return epxl(first_piece, s, columns - 1, 'D') - epxl(second_piece, s, 0, 'D')
    elif position == 'T':
        return epxl(second_piece, 0, s, 'H') - epxl(first_piece, columns - 1, s, 'H')
    elif position == 'D':
        return epxl(first_piece, columns - 1, s, 'H') - epxl(second_piece, 0, s, 'H')


def ECAB(first_piece, second_piece, s, position):
    """
        expected change across the boundary of the two piece
    """
    rows, columns, _ = first_piece.shape()
    value = None
    if position == 'L':
        value = 0.5 * (first_piece[s, columns - 1, :] - first_piece[s, columns - 2, :]
                       + second_piece[s, 1, :] - second_piece[s, 0, :])
    elif position == 'R':
        value = 0.5 * (first_piece[s, columns - 2, :] - first_piece[s, columns - 1, :]
                       + second_piece[s, 0, :] - second_piece[s, 1, :])
    elif position == 'T':
        value = 0.5 * (first_piece[columns - 1, s, :] - first_piece[columns - 2, s, :]
                       + second_piece[1, s, :] - second_piece[0, s, :])
    elif position == 'D':
        value = 0.5 * (first_piece[columns - 2, s, :] - first_piece[columns - 1, s, :]
                       + second_piece[0, s, :] - second_piece[1, s, :])
    return value


def ECABG(first_piece, second_piece, s, position):
    """
        expected change across the boundary of the two piece
    """
    rows, columns, _ = first_piece.shape()
    value = None
    if position == 'L':
        value = 0.5 * (epxl(first_piece, s, columns - 1, 'D') - epxl(first_piece, s, columns - 2, 'D')
                       + epxl(second_piece, s, 1, 'D') - epxl(second_piece, s, 0, 'D'))
    elif position == 'R':
        value = 0.5 * (epxl(first_piece, s, columns - 2, 'D') - epxl(first_piece, s, columns - 1, 'D')
                       + epxl(second_piece, s, 0, 'D') - epxl(second_piece, s, 1, 'D'))
    elif position == 'T':
        value = 0.5 * (epxl(first_piece, columns - 1, s, 'H') - epxl(first_piece, columns - 2, s, 'H')
                       + epxl(second_piece, 1, s, 'H') - epxl(second_piece, 0, s, 'H'))
    elif position == 'D':
        value = 0.5 * (epxl(first_piece, columns - 2, s, 'H') - epxl(first_piece, columns - 1, s, 'H')
                       + epxl(second_piece, 0, s, 'H') - epxl(second_piece, 1, s, 'H'))
    return value


def ViL(piece, position):
    """
     ViL calculated from samples,{xi(s,S)−xi(s,S−1))∣s=1,2,...S}.
    """
    rows, columns, _ = piece.shape()
    result = np.zeros(shape=[3, rows + 3])
    if position == 'L':
        for i in range(rows):
            # [1,2,3]
            result[:, i] = piece[i, columns - 1, :] - piece[i, columns - 2, :]
    elif position == 'R':
        for i in range(rows):
            result[:, i] = piece[i, 0, :] - piece[i, 1, :]
    elif position == 'T':
        for i in range(columns):
            result[:, i] = piece[rows - 1, i, :] - piece[rows - 2, i, :]
    elif position == 'D':
        for i in range(columns):
            result[:, i] = piece[0, i, :] - piece[1, i, :]
    # result[:, -3] = [1, 0, 0]
    # result[:, -2] = [0, 1, 0]
    # result[:, -1] = [0, 0, 1]
    return result


def VGiL(piece, position):
    """
     VGiL calculated from samples,{δ(s,S)−δ(s,S−1))∣s=2,2,...S}.
    """
    rows, columns, _ = piece.shape()
    result = np.zeros(shape=[3, rows + 3])
    if position == 'L':
        for i in range(1, rows):
            result[:, i] = epxl(piece, i, columns - 1, 'D') - epxl(piece, i, columns - 2, 'D')
    elif position == 'R':
        for i in range(1, rows):
            result[:, i] = epxl(piece, i, 0, 'D') - epxl(piece, i, 1, 'D')
    elif position == 'T':
        for i in range(1, columns):
            result[:, i] = epxl(piece, rows - 1, i, 'H') - epxl(piece, rows - 2, i, 'H')
    elif position == 'D':
        for i in range(1, columns):
            result[:, i] = epxl(piece, 0, i, 'H') - epxl(piece, 1, i, 'H')
    # result[:, -3] = [1, 0, 0]
    # result[:, -2] = [0, 1, 0]
    # result[:, -1] = [0, 0, 1]
    return result


epxl_dict = {}


def epxl(*args):
    if args in epxl_dict:
        return epxl_dict[args]
    piece = args[0]
    x = args[1]
    y = args[2]
    position = args[3]
    v = None
    if position == 'D':
        v = piece[x, y, :] - piece[x - 1, y, :]
    elif position == 'H':
        v = piece[x, y, :] - piece[x, y - 1, :]
    epxl_dict[args] = v
    return v
