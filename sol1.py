import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread as imread, imsave as imsave
from skimage import img_as_float
from skimage.color import rgb2gray

BINS = 256
RGB_CODE = 2
GRAY_SCALE_CODE = 1


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


def imidisplay(filename, representation):
    """
    receive file and representation and display the image
    :param filename: file path
    :param representation: representation code
    :return: None, shows the picture
    """
    im = read_image(filename, representation)
    if representation == GRAY_SCALE_CODE:
        plt.imshow(im, cmap=plt.cm.gray)
    elif representation == RGB_CODE:
        plt.imshow(im)
    plt.show()


def rgb2yiq(imRGB):
    """
    Transfer the image from RGB to YIQ format using the given matrix
    :param imRGB: image to transfer
    :return: transferred image
    """
    imYIQ = imRGB.copy()  # so changes won't appear on both imYIQ and imRGB
    # get RGB on seperate 2D nd arrays
    Rmap = imYIQ[:, :, 0]
    Gmap = imYIQ[:, :, 1]
    Bmap = imYIQ[:, :, 2]
    # should've been done using numpy.dot , but didn't return the matching values
    Ymap = Rmap * 0.299 + Gmap * 0.587 + Bmap * 0.114
    Imap = Rmap * 0.596 + Gmap * -0.275 + Bmap * -0.321
    Qmap = Rmap * 0.212 + Gmap * -0.523 + Bmap * 0.311
    imYIQ[:, :, 0] = Ymap
    imYIQ[:, :, 1] = Imap
    imYIQ[:, :, 2] = Qmap
    return imYIQ


def yiq2rgb(imYIQ):
    """
    Transfer the image from YIQ to RGB format using the given matrix
    :param imYIQ: image to transfer
    :return: transferred image
    """
    imRGB = imYIQ.copy()  # so changes won't appear on both imYIQ and imRGB
    Ymap = imYIQ[:, :, 0]
    Imap = imYIQ[:, :, 1]
    Qmap = imYIQ[:, :, 2]
    Rmap = Ymap + Imap * 0.95 + Qmap * 0.616
    Gmap = Ymap + Imap * -0.266 + Qmap * -0.644
    Bmap = Ymap + Imap * -1.121 + Qmap * 1.7
    imRGB[:, :, 0] = Rmap
    imRGB[:, :, 1] = Gmap
    imRGB[:, :, 2] = Bmap
    return imRGB


def histogram_equalize(im_orig):
    """
    performs histogram equalization of a given grayscale or RGB image.
    :param im_orig:- is the input grayscale or RGB float64 image with values in [0, 1]
    :return:
        im_eq - is the equalized image. grayscale or RGB float64 image with values in [0, 1].
        hist_orig - is a 256 bin histogram of the original image (array with shape (256,) ).
        hist_eq - is a 256 bin histogram of the equalized image (array with shape (256,) ).
    """
    isRGB = np.size(im_orig[1, 1]) != 1
    if isRGB:
        toOperate = rgb2yiq(im_orig)[:, :, 0]
    else:
        toOperate = im_orig
    new_channel, hist, new_hist = equalize_channel(toOperate)
    if isRGB:
        imYIQ = rgb2yiq(im_orig)
        imYIQ[:, :, 0] = new_channel
        return_img = yiq2rgb(imYIQ)
        return return_img, hist, new_hist
    return new_channel, hist[0], new_channel[0]


def equalize_channel(channel):
    """
    Equalize the channel following the algorithm shown in class
    :param channel: channel to be equalize
    :return: the new channel, old and new histograms (respectively)
    """
    chInt = (channel * 255).astype(int)
    hist = np.histogram(chInt, bins=256)
    chInt = ((chInt - np.amin(chInt)) / np.amax(channel)).astype(int)
    cum_hist = np.cumsum(hist[0]) / np.size(channel)  # normalized cum_hist
    cum_hist *= np.amax(channel)
    if np.amax(cum_hist) < 1 or np.amin(cum_hist) > 0:
        cum_hist = (cum_hist - np.amin(cum_hist)) / np.amax(cum_hist)
    cum_hist *= 255
    cum_hist = cum_hist.astype(int)
    fixed_ch = cum_hist[chInt]
    new_hist = np.histogram(fixed_ch, bins=256)
    return fixed_ch / 255, hist, new_hist


def findInitZ(cum_hist, n_quant):
    """
    To avoid division by 0, the function find the initial z values so between any z partition there will
    be an equal amount of pixels
    :param im_orig: picture to operate on
    :param n_quant: number of partition needed
    :return: list containing all the partition's values
    """
    # to improve performance, 0 and 255 are added without searching
    z_list = [0]
    im_size = cum_hist[-1]
    for i in range(1, n_quant):
        curr_amount = i * im_size / (n_quant)
        # at each iteration, we find the first pixel which has enough pixel to add a partition
        z_list.append(((np.where(cum_hist > curr_amount))[0])[0])
    z_list.append(255)
    return z_list


def findZ(q_list):
    """
    return a list of the matching z value
    :param q_list: to consider when assigning z
    :return: z_list: which have the matching z values
    """
    z_list = [0]
    for i in range(len(q_list) - 1):
        z_list.append((q_list[i] + q_list[i + 1]) / 2)
    z_list.append(255)
    return z_list


def quantize_helper(channel, n_quant, n_iter):
    """
    prefix function for quantize, return the matching q and partition list
    :param channel: to operate on
    :param n_quant: number of colors requested
    :param n_iter: number of max iteration to preform
    :return: error vector, matching q and z lists
    """
    hist = np.histogram((channel * 255).astype(int), bins=256)
    cum_hist = np.cumsum(hist[0])
    errVec = []
    z_list = findInitZ(cum_hist, n_quant)
    prev_z = z_list.copy()
    for i in range(n_iter):
        q_list, curr_err = findQ(hist, z_list)
        errVec.append(curr_err)
        prev_z = z_list.copy()
        z_list = findZ(q_list)
        if (np.array_equal(prev_z, z_list)):  # stop in case of no change between iterations
            return errVec, q_list, z_list

    return errVec, q_list, z_list


def findQ(hist, z_list):
    """
    Given the histogram and the partition list, find the matching q values to minimize SSE
    for readability, each values are found using the helper function getCurrQandError
    :param hist: picture's histogram
    :param z_list: partitions list
    :return: q_list :q's values and errorSum as requested
    """
    errorSum = 0
    q_list = []
    for i in range(len(z_list) - 1):
        curr_q, err = getCurrQandError(hist[0], i, z_list)
        q_list.append(curr_q)
        errorSum += err  # errorSum sums all the partition's SSE
    return q_list, errorSum


def getCurrQandError(hist, i, z_list):
    """
    helper function for findQ , find the q and error on the i iteration
    :param hist: picture's histogram
    :param i: iteration number
    :param z_list: list of partitions
    :return: the the current q and SSE of the specific partition
    """
    fromC = int(z_list[i])
    toC = int(z_list[i + 1])
    subHist = hist[fromC:toC]
    zVec = np.dot(subHist, np.arange(fromC, toC))
    denominator = np.sum(subHist)
    curr_q = zVec / denominator
    innerSum = calcCurrErr(np.arange(fromC, toC), subHist, curr_q)
    return curr_q, innerSum


def calcCurrErr(range, hist, q):
    """
    receive the range of partition and all the data below to calculate the SSE error
    :param range: to calculate on
    :param hist: histogram
    :param q: matching current q
    :return: the sum of SSE of all pixels in the partition
    """
    newDist = range - q
    squareDist = np.square(newDist)
    hist = np.matrix(hist)
    hist = hist.transpose()
    err = np.dot(squareDist, hist)
    return err[0, 0]


def quantImage(channel, q_list, z_list):
    """
    after quantization have done, the function gets the following variables and return the new channel
    :param channel: to operate on
    :param q_list: list of colors
    :param z_list: partitions to defer between the range of the new colors
    :return: the channel after modifications
    """
    q_list = np.array(q_list) / 255
    z_list = np.array(z_list) / 255
    quantCh = (channel.copy()) * 0
    for i in range(len(q_list)):
        # each iteration the function creates curr_cords which sends all matching pixels to their
        # matching values and send the rest to zero, till all colors are received
        curr_cords = np.where(np.logical_and(channel >= z_list[i], channel <= z_list[i + 1]), q_list[i], 0)
        quantCh += curr_cords
    return quantCh


def quantize(im_orig, n_quant, n_iter):
    """
     performs optimal quantization of a given grayscale or RGB image
    :param im_orig: input grayscale or RGB image to be quantized (float64 image with values in [0, 1])
    :param n_quant: is the number of intensities your output im_quant image should have
    :param n_iter: is the maximum number of iterations of the optimization procedure (may converge earlier.)
    :return:
        im_quant - is the quantized output image.
        error - is an array with shape (n_iter,) (or less) of the total intensities error for each iteration of the
        quantization procedure
    """
    isRGB = np.ndim(im_orig) == 3
    if isRGB:
        toOperate = rgb2yiq(im_orig)[:, :, 0]
    else:
        toOperate = im_orig
    errVec, q_list, z_list = quantize_helper(toOperate, n_quant, n_iter)
    newIm = quantImage(toOperate, q_list, z_list)
    return newIm, errVec


