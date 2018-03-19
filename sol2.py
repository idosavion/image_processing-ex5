import numpy as np
from numpy.linalg import inv as inv
from skimage.color import rgb2gray
from skimage.io import imread
from scipy import signal
from skimage import img_as_float

MONKEY_JPG = "C:\\Users\\idosa\\PycharmProjects\\ex2\\external\\monkey.jpg"

# constants
Y = 1
X = 0
PI = np.math.pi
I = complex(0, 1)
RGB_CODE = 2
GRAY_SCALE_CODE = 1
BINOMIAL_FACTOR = np.array([[1, 1], [1, 1]])


def read_image(filename, representation):
    """
    The function receives a file path and open the image in the matching mode
    :param filename: file path
    :param representation: 2 = RGB , 1 = Gray scale
    :return: im object represented by floats
    """
    im = imread(filename, mode='RGB')
    im = img_as_float(im)
    if representation == GRAY_SCALE_CODE:
        im_g = rgb2gray(im)
        return im_g
    elif representation == RGB_CODE:
        return im


def getTwiddleMatrix(N):
    """
    The function return the matrix required for preforming the fourier transform using linear functions
    :param N: size of matrix needed
    :return: the requested matrix
    """
    twiddle_vec = np.matrix(np.arange(N))
    twiddle_mat = np.dot(twiddle_vec.transpose(), twiddle_vec)
    twiddle_mat = np.exp(-2 * PI * I * twiddle_mat / N)
    return twiddle_mat


def DFT(signal):
    """
    1 dimension DFT
    :param signal: to preform on
    :return: the DFT of requested vector
    """
    signal = np.asarray(signal)  # to allow the function be compatible
    # signal = np.asarray(signal, dtype=float64)  # to allow the function be compatible
    # with matrixes in addition
    N = signal.size
    twiddle_mat = getTwiddleMatrix(N)
    return twiddle_mat @ signal


def IDFT(fourier_signal):
    """

    :param fourier_signal: to preform on
    :return: the inverse DFT of requested vector
    """
    fourier_signal = np.asarray(fourier_signal)  # to allow the function be compatible
    # fourier_signal = np.asarray(fourier_signal, dtype=complex128)
    N = fourier_signal.size
    twiddle_mat = getTwiddleMatrix(N)
    inv_mat = inv(twiddle_mat)
    return_vec = inv_mat @ fourier_signal
    return np.asmatrix(return_vec)


def DFT2(image):
    """
    2D DFT
    :param image: to operate on
    :return: the DFT of image
    """
    m, n = image.shape
    twiddle_m = getTwiddleMatrix(m)
    twiddle_n = getTwiddleMatrix(n)
    # using linear process the image returned is the fourier transormed picture
    return (twiddle_m @ image) @ twiddle_n


def IDFT2(fourier_image):
    """

    :param fourier_image: to operate on
    :return: the IDFT of image
    """
    m, n = fourier_image.shape
    inv_twiddle_m = inv(getTwiddleMatrix(m))
    inv_twiddle_n = inv(getTwiddleMatrix(n))
    return (inv_twiddle_m @ fourier_image) @ inv_twiddle_n


def conv_der(im):
    """
    Return the derivative of the image using the root of square of sum of dx and dy
    :param im: to operate on
    :return: derivative of image
    """
    # using roll to preform convolution
    dx = im - np.roll(im, 1, axis=X)
    dy = im - np.roll(im, 1, axis=Y)
    magnitude = np.sqrt(np.abs(dx) ** 2 + np.abs(dy) ** 2)
    return magnitude


def multiplyByCol(matrix):
    """
    To achieve the derivative through linear process each column will be multiplied
    :param matrix: to multiply each column in (transposed for rows)
    :return: the mutliplied matrix
    """
    range_arr = matrix.shape[1]
    range_vec = np.arange(range_arr / 2)
    mirror_vec = range_arr / 2 - range_vec
    multiply_vec = np.concatenate((range_vec, mirror_vec), axis=0)
    return matrix * multiply_vec


def fourier_der(im):
    """
    using the previous function as prefix
    :param im: to get derivative of
    :return: the derivative of image
    """
    fourier_image = np.array(DFT2(im))
    im_u = multiplyByCol(fourier_image)
    im_v = multiplyByCol(fourier_image.transpose()).transpose()
    dx = IDFT2(im_u)
    dy = IDFT2(im_v)
    magnitude = np.sqrt(np.square(np.abs(dx)) + np.square(np.abs(dy)))
    return magnitude / im.size  # divided to fit values, should happen in transform


def getKern(size):
    """
    get kernel in the needed size
    :param size: needed size
    :return: 2D kernel
    """
    return_kern = BINOMIAL_FACTOR.copy()
    for i in range(size - 2):
        return_kern = np.asmatrix(signal.convolve2d(return_kern, BINOMIAL_FACTOR))
    return return_kern / np.matrix.sum(return_kern)


def blur_spatial(im, kernel_size):
    """
    Blur the image using the kernel from previous function
    :param im: to blur
    :param kernel_size: size of the requested kernel
    :return: blurred image
    """
    kern = getKern(kernel_size)
    return signal.convolve2d(im, kern, mode='same')


def blur_fourier(im, kernel_size):
    """
    blur the image using the fourier transform of the kernel
    :param im: image to preform on
    :param kernel_size: size of kernel
    :return: blurred image
    """
    fourier_im = np.array(DFT2(im))
    fourier_im = np.array(np.fft.fftshift(fourier_im))
    pad_x, pad_y = (np.array(im.shape) - kernel_size)//2
    orig_kern = getKern(kernel_size)
    kernal = np.pad(orig_kern, pad_width=((pad_x, pad_x+1), (pad_y, pad_y+1)), constant_values=0, mode='constant')
    fourier_kern = np.array(DFT2(kernal))
    fourier_kern = np.array(np.fft.fftshift(fourier_kern))
    product = np.multiply(fourier_kern,fourier_im)
    return np.abs(np.fft.ifftshift(IDFT2(product)))

