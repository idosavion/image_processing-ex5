import numpy as np
import scipy.ndimage
import os.path
import matplotlib.pyplot as plt

MINIMAL_OP_SIZE = 16


def build_filter(size):
    """
    Build binomial factor of the requested size
    :param size: size of kernel
    :return: nd matrix of size (size,1)
    """
    source = np.array([1, 1])
    toReturn = source.copy()
    for i in range(size - 2):
        toReturn = np.convolve(source, toReturn)
    return np.asmatrix(toReturn / sum(toReturn))  # return the convolved vector normalized


def blur_image(im, g_filter):
    """
    Blur the given image using convolution with the given filter
    :param im: to blur
    :param g_filter: to be used in convolution
    :return: the blurred image
    """
    # blur at one dimension at a time for performance optimization
    convolved_im = scipy.ndimage.filters.convolve(im, g_filter, mode='reflect')
    convolved_im = scipy.ndimage.filters.convolve(convolved_im, g_filter.transpose(), mode='reflect')
    return convolved_im


def reduce_im(im, g_kernel):
    """blur the image and return a picture of half the size sub sampling"""
    blurred_im = blur_image(im, g_kernel)
    return blurred_im[::2, ::2].copy()


def expand_im(im, g_kernel):
    """
    Pad the images with zeroes
    :param im: image to expand
    :param g_kernel: to blur with
    :return: the expanded image
    """
    padded_im = np.zeros((im.shape[0] * 2, im.shape[1] * 2), dtype=float)
    padded_im[::2, ::2] = im.copy()
    toReturn = blur_image(padded_im, g_kernel * 2)
    return toReturn


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    build the pyramid of the required size
    :param im: to build pyramid of
    :param max_levels: the required number of levels (if the image is large enough
    :param filter_size: size of filter for blurring and expending
    :return: python list of all pyramid levels and the filter which was used
    """
    gaussian_filter = build_filter(filter_size)
    pyramid = [im]
    curr_im = im.copy()
    for i in range(max_levels - 1):
        curr_im = reduce_im(curr_im, gaussian_filter)
        if min(np.shape(curr_im)) < MINIMAL_OP_SIZE:
            return pyramid, gaussian_filter
        pyramid.append(curr_im)
    return pyramid, gaussian_filter


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    build the pyramid of the required size using the gaussian pyramid
    :param im: to build pyramid of
    :param max_levels: the required number of levels (if the image is large enough
    :param filter_size: size of filter for blurring and expending
    :return: python list of all pyramid levels and the filter which was used
    """
    prev_im = im
    l_pyr = []
    g_pyr, g_kern = build_gaussian_pyramid(im, max_levels, filter_size)
    for i in range(min(max_levels - 1, len(g_pyr) - 1)):
        if min(np.shape(prev_im)) < MINIMAL_OP_SIZE:
            return l_pyr, g_kern
        l_pyr.append(g_pyr[i] - expand_im(g_pyr[i + 1], g_kern))
    l_pyr.append(reduce_im(g_pyr[i], g_kern))
    return l_pyr, g_kern


def laplacian_to_image(lpyr, filter_vec, coeff):
    """
    The function build the original image using the laplacian pyramid
    :param lpyr: laplacian pyramid
    :param filter_vec: vector which was used for processing the image
    :param coeff: coefficient vector which multiply each level
    :return: the reconstructed image
    """
    lpyr = [lpyr[i] * coeff[i] for i in range(len(lpyr))]
    for i in range(len(lpyr) - 2, -1, -1):
        lpyr[i] = lpyr[i] + expand_im(lpyr[i + 1], filter_vec)
    return lpyr[i]


def normalize_pyr(pyr, levels):
    """
    the function normalize each image in the pyramid into [0,1]
    :param pyr: list of photos to normalize
    :param levels: number of levels
    :return: the normalized list of images
    """
    for i in range(levels):
        im_min = pyr[i].min()
        pyr[i] -= im_min
        pyr[i] *= (1 / pyr[i].max())
    return pyr


def render_pyramid(pyr, levels):
    """
    create a large photo containing all levels of pyramid side by side
    :param pyr: pyramid of images
    :param levels: number of levels
    :return: image as described
    """
    norm_pyr = normalize_pyr(pyr, levels)
    height = norm_pyr[0].shape[0]
    width = sum(im.shape[1] for im in norm_pyr[0:levels])
    black = np.zeros((height, width))
    curr_x = 0
    for i in range(levels):
        im_y, im_x = norm_pyr[i].shape
        end_x, end_y = curr_x + im_x, im_y
        black[0:end_y, curr_x:end_x] = norm_pyr[i]
        curr_x += im_x
    return black


def display_pyramid(pyr, levels):
    """
    display the required pyramid to enable
    :param pyr: photo's pyramid
    :param levels: number of levels
    :return: None, shows the output of render_pyramid
    """
    pyr_show = render_pyramid(pyr, levels)
    plt.figure()
    plt.imshow(pyr_show, cmap='gray')


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
    Implement pyramid blending as described in the lecture
    :param im1: input grayscale image to be blended.
    :param im2: input grayscale image to be blended.
    :param mask: is a boolean (i.e. dtype == np.bool) mask containing True and False representing which parts
    of im1 and im2 should appear in the resulting im_blend. Note that a value of True corresponds to 1,
    and False corresponds to 0.
    :param max_levels: parameter to be used when generating the Gaussian and Laplacian pyramids.
    :param filter_size_im:  size of the Gaussian filter (an odd scalar that represents a squared filter) which
    defining the filter used in the construction of the Laplacian pyramids of im1 and im2.
    :param filter_size_mask: is the size of the Gaussian filter(an odd scalar that represents a squared filter) which
    defining the filter used in the construction of the Gaussian pyramid of mask.
    :return:
    """
    lapl_pyr1, kern = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    lapl_pyr2, kern = build_laplacian_pyramid(im2, max_levels, filter_size_im)
    float_mask = mask.astype(float)
    mask_pyr, mask_kern = build_gaussian_pyramid(float_mask, max_levels, filter_size_mask)
    Lout = [0] * len(lapl_pyr1)
    for i in range(len(lapl_pyr1)):
        Lout[i] = np.multiply(lapl_pyr1[i], mask_pyr[i]) + np.multiply(1 - mask_pyr[i], lapl_pyr2[i])
    coeff = [1] * len(lapl_pyr2)
    blend_im = laplacian_to_image(Lout, mask_kern, coeff)
    return blend_im.clip(0, 1)


def relpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)


def blend_color_im(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
    blend each channel and returned the blended image
    :param all as described in the blending function
    :return: the blended image
    """
    blended = np.zeros(im1.shape)
    for i in range(3):
        blended[:, :, i] = pyramid_blending(im1[:, :, i], im2[:, :, i], mask, max_levels, filter_size_im,
                                            filter_size_mask)
    return blended


def blending_example1():
    color_lenna = sol1.read_image(relpath("example1/lena_color.jpg"), 2)
    nat = sol1.read_image(relpath("example1/nat_pic for mask.jpg"), 2)
    mask = sol1.read_image(relpath("example1/mask_2.jpg"), 1)
    blended = blend_color_im(color_lenna, nat, mask, 11, 7, 7)
    show_sim(color_lenna,nat,mask,blended)
    return color_lenna, nat, mask.astype(bool), blended


def blending_example2():
    givat_ram = sol1.read_image(relpath("example2/givat_ram.JPG"), 2)
    wolves = sol1.read_image(relpath("example2/wolf_im.JPG"), 2)
    mask = sol1.read_image(relpath("example2/wolf_mask.JPG"), 1)
    blended = blend_color_im(givat_ram, wolves, mask, 3, 3, 3)
    show_sim(givat_ram, wolves, mask, blended)
    return givat_ram, wolves, mask.astype(bool), blended


def show_sim(im1, im2, mask, blended):
    """
    Show all four photos simultaneously
    """
    plt.figure(1)
    plt.subplot(221)
    plt.imshow(im1)
    plt.subplot(222)
    plt.imshow(im2)
    plt.subplot(223)
    plt.imshow(mask)
    plt.subplot(224)
    plt.imshow(blended)
    plt.show()

