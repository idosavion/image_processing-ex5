import numpy as np
import scipy.ndimage
from keras.engine import Model
from keras.layers import Input, Convolution2D, Activation
from keras.layers import merge
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import sol4_utils
import sol5_utils


MIN_SIGMA = 0

MAX_SIGMA = 0.2

memo = {}


def read_image(path):
    if path not in memo:
        memo[path] = sol4_utils.read_image(path, 1)
    return memo[path]


def load_dataset(filenames, batch_size, corruption_func, crop_size):
    """
    outputs data_generator, which is a tuple of source and target batch which each one
    of them is an array of shape (batch_size,1,h,w)
    :param filenames:
    :param batch_size:
    :param corruption_func:
    :param crop_size:
    :return:
    """
    i = 0
    while True:
        rand_range = np.random.choice(len(filenames), batch_size, True)
        im_list = [random_crop(filenames[i], crop_size)-0.5 for i in rand_range]
        scrambled_list = [corruption_func(im[0])[np.newaxis] for im in im_list]
        yield [np.asarray(scrambled_list), np.asarray(im_list)]


def random_crop(path, crop_size):

    im = read_image(path)
    height, width = im.shape
    end_height, end_width = np.random.randint(crop_size[0], height), np.random.randint(crop_size[1], width)
    start_height, start_width = end_height - crop_size[0], end_width - crop_size[1]
    cropped_im = im[start_height:end_height, start_width:end_width]
    cropped_im = np.expand_dims(cropped_im, 0)
    return cropped_im


def resblock(input_tensor, num_channels):
    b = Convolution2D(num_channels, 3, 3, border_mode='same')(input_tensor)
    b = Activation('relu')(b)
    b = Convolution2D(num_channels, 3, 3, border_mode='same')(b)
    b = merge([input_tensor, b], mode='sum')
    return b


def build_nn_model(height, width, num_channels, num_res_blocks):
    input = Input(shape=(1, height, width))
    a = Convolution2D(num_channels, 3, 3, border_mode='same')(input)
    a = Activation('relu')(a)
    b = resblock(a, num_channels)
    for i in range(num_res_blocks):
        b = resblock(b, num_channels)
    c = merge([a, b], mode='sum')
    c = Convolution2D(1, 3, 3, border_mode='same')(c)
    model = Model(input=input, output=c)
    return model


def train_model(model, images, corruption_func, batch_size,
                samples_per_epoch, num_epochs, num_valid_samples):
    crop = model.input_shape[2:]
    training_size = int(len(images) * 0.8)
    training_gen = load_dataset(images[:training_size], batch_size, corruption_func, crop)
    validation_gen = load_dataset(images[training_size:], batch_size, corruption_func, crop )
    model.compile(loss='mean_squared_error', optimizer=Adam(beta_2=0.9))
    model.fit_generator(training_gen, samples_per_epoch, num_epochs, validation_data=validation_gen,
                        nb_val_samples=num_valid_samples)
    return model


def get_sets(images):
    set_size = len(images)
    training_size = int(set_size * 0.8)
    images = sol5_utils.list_images(images, True)
    training_set = images[:training_size]
    validation_set = images[training_size:]
    return training_set, validation_set


def restore_image(corrupted_image, base_model):
    corrupted_tensor = corrupted_image[np.newaxis]
    a = Input(corrupted_tensor.shape)
    b = base_model(a)
    new_model = Model(input=a, output=b)
    y = new_model.predict(corrupted_tensor[np.newaxis])
    return np.array(y[0][0] + 0.5,dtype=np.float64)


def add_gaussian_noise(image, min_sigma, max_sigma):
    rand_sig = np.random.uniform(min_sigma, max_sigma)
    ret_image =image + np.random.normal(scale=rand_sig, size=image.shape)
    ret_image = (1 / 255) * np.round(ret_image* 255)
    return ret_image


def learn_denoising_model(num_res_blocks=5, quick_mode=False):
    images = sol5_utils.images_for_denoising()
    func = lambda x: add_gaussian_noise(x, MIN_SIGMA, MAX_SIGMA)
    model = build_nn_model(24, 24, 48, num_res_blocks)
    if (quick_mode):
        train_model(model, images, func, 10, 30, 2, 30)
    else:
        train_model(model, images, func, 100, 10000, 10, 1000)
    return model


def add_motion_blur(image, kernel_size, angle):
    kern = sol5_utils.motion_blur_kernel(kernel_size, angle)
    ret_image = scipy.ndimage.filters.convolve(image, kern)
    return ret_image


def random_motion_blur(image, list_of_kernel_sizes):
    rand_index = np.random.randint(len(list_of_kernel_sizes))
    rand_size = list_of_kernel_sizes[rand_index]
    rand_angle = np.random.random() * np.pi
    return add_motion_blur(image, rand_size, rand_angle)


def learn_deblurring_model(num_res_blocks=5, quick_mode=False):
    model = build_nn_model(16, 16, 32, num_res_blocks)
    images = sol5_utils.images_for_deblurring()
    func = lambda x: random_motion_blur(x, [7])
    if (quick_mode):
        train_model(model, images, func, 10, 30, 2, 30) 
    else:
        train_model(model, images, func, 100, 10000, 10, 1000)
    return model

#
# trained_blurring = learn_denoising_model(5, True)
# im = sol4_utils.read_image('image_dataset/train/2092.jpg', 1)
# blurred_im = add_motion_blur(im,3,0.2)
# fixed = restore_image(blurred_im, trained_blurring)
# err = []
# for i in range(4,6):
#     trained_blurring = learn_denoising_model(i)
#     err.append(trained_blurring.history.history['loss'][-1])
#     print(trained_blurring.history.history['loss'][-1])
#     trained_blurring.save_weights("blurred_weights.txt")
# model = train_model(model,IMAGES,scramble_im,(18,18),1000,100,1000)
# model.save_weights('weights.txt')
# model.load_weights('weights.txt')
# print(err)
# exemp = scramble_im(exemp)
# im = restore_image(exemp,model)
# print('shit')
#

