"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from typing import List
import cv2
import numpy as np
import matplotlib.pyplot as plt

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2
NUM_OF_PIXELS_255 = 255
NUM_OF_PIXELS_256 = 256
EPSILON = 0.001
MY_ID = 208980359


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return MY_ID


def normalize_image(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    if representation == LOAD_GRAY_SCALE:
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        return normalize_image(img=img)

    elif representation == LOAD_RGB:
        img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
        return normalize_image(img=img)


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    img = imReadAndConvert(filename, representation)
    if representation == LOAD_GRAY_SCALE:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    plt.show()


def split_to_channels(img: np.ndarray):
    return img[:, :, 0], img[:, :, 1], img[:, :, 2]


def assign_channels(img: np.ndarray, x: int, y: int, z: int):
    img[:, :, 0], img[:, :, 1], img[:, :, 2] = x, y, z


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    imgRGB = normalize_image(imgRGB)
    r, g, b = split_to_channels(imgRGB)
    y, i, q = r * 0.299 + g * 0.587 + b * 0.114, r * 0.596 + g * -0.275 + b * -0.321, r * 0.212 + g * -0.523 + b * 0.311
    yiq_img = imgRGB
    assign_channels(yiq_img, y, i, q)
    return normalize_image(yiq_img)


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    imgYIQ = normalize_image(imgYIQ)
    y, i, q = split_to_channels(imgYIQ)
    r, g, b = y + i * 0.956 + q * 0.619, y + i * -0.272 + q * -0.647, y + i * -1.106 + q * 1.703
    r, g, b = normalize_image(r), normalize_image(g), normalize_image(b)
    rgb_img = imgYIQ
    assign_channels(rgb_img, r, g, b)

    return rgb_img


def hsitogram_norm(norm_255, num_of_pixels):
    hist_origin, edges = np.histogram(norm_255, NUM_OF_PIXELS_256, [0, NUM_OF_PIXELS_256])
    cum_sum_origin = np.cumsum(hist_origin)
    return np.ceil(cum_sum_origin * NUM_OF_PIXELS_255 / num_of_pixels), hist_origin


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Equalizes the histogram of an image
    :param imgOrig: Original Histogram
    :return
    """
    num_of_pixels = imgOrig.shape[0] * imgOrig.shape[1]
    if len(imgOrig.shape) == 2:
        norm_255 = normalize_image(imgOrig) * NUM_OF_PIXELS_255
        lut, hist_origin = hsitogram_norm(norm_255, num_of_pixels)
        img_eq = norm_255
        for i in range(imgOrig.shape[0]):
            for j in range(imgOrig.shape[1]):
                img_eq[i][j] = lut[int(norm_255[i][j])]

        hist_eq, edges = np.histogram(img_eq, NUM_OF_PIXELS_256, [0, NUM_OF_PIXELS_256])
        return normalize_image(img_eq), hist_origin, hist_eq
    else:
        yiq = transformRGB2YIQ(imgOrig)
        norm_255 = normalize_image(yiq[:, :, 0]) * NUM_OF_PIXELS_255
        lut, hist_origin = hsitogram_norm(norm_255, num_of_pixels)
        norm_255_new = norm_255
        for i in range(imgOrig.shape[0]):
            for j in range(imgOrig.shape[1]):
                norm_255_new[i][j] = lut[int(norm_255[i][j])]

        hist_eq, edges = np.histogram(norm_255_new, NUM_OF_PIXELS_256, [0, NUM_OF_PIXELS_256])
        yiq[:, :, 0] = normalize_image(norm_255_new)
        img_eq = normalize_image(transformYIQ2RGB(yiq))
        return img_eq, hist_origin, hist_eq


def next_z(z, q):
    i = 0
    while i < len(q) - 1:
        z[i + 1] = (q[i] + q[i + 1]) / 2
        i += 1
    return z


def find_new_q(z, q, hist, num_pixel):
    ans, new_q, t = 1, q, 0
    i = 0
    while i < len(z) - 1:
        res1 = res2 = 0
        j = z[i]
        while j < z[i + 1]:
            res1 += hist[j] / num_pixel * j
            res2 += hist[j] / num_pixel
            j += 1

        if res2 != 0:
            qi = int(res1 / res2)
            q[t] = qi
            t += 1
        else:
            return q, 0
        i += 1
    return np.ceil(new_q), ans


def new_pic(imOrig255, nQuant, z, q):
    shape, new_img = imOrig255.shape, imOrig255
    for i in range(shape[0]):
        for j in range(shape[1]):
            x = int(imOrig255[i][j])
            for t in range(nQuant):
                if z[t] <= x <= z[t + 1]:
                    new_img[i][j] = q[t]
    return new_img


def calc_mse(nQuant, z, q, hist, pixel_num):
    mse = i = 0
    while i < nQuant:
        temp_mse = 0
        j = z[i]
        while j < z[i + 1]:
            temp_mse += (q[i] - j) * (q[i] - j) * hist[j] / pixel_num
            j += 1
        mse += temp_mse
        i += 1
    return mse


def origin_z(pixel_num, nQuant, cumsum, z):
    temp_bound = pixel_num / nQuant
    bound = temp_bound
    t, i = 1, 0
    while i < NUM_OF_PIXELS_256:
        if cumsum[i] >= bound:
            z[t] = i
            t += 1
            bound = bound + temp_bound
        i += 1
    return z.astype(int)


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    z, q = np.zeros(nQuant + 1), np.zeros(nQuant)
    list_of_pic, list_of_mse = [], []
    pixel_num = imOrig.shape[0] * imOrig.shape[1]

    if len(imOrig.shape) == 2:
        imOrig255 = normalize_image(imOrig) * NUM_OF_PIXELS_255
        hist, edges = np.histogram(imOrig255, NUM_OF_PIXELS_256, [0, NUM_OF_PIXELS_256])

        cum_sum = np.cumsum(hist)

        z = origin_z(pixel_num, nQuant, cum_sum, z)
        q, ans = find_new_q(z, q, hist, pixel_num)

        new_img = new_pic(imOrig255, nQuant, z, q)
        list_of_pic.append(new_img)
        temp_mse = calc_mse(nQuant, z, q, hist, pixel_num)
        list_of_mse.append(temp_mse)

        old_mse, old_q = temp_mse, q
        for k in range(1, nIter):
            z = next_z(z, q)
            q, ans = find_new_q(z, q, hist, pixel_num)
            if np.array_equal(q, old_q) and ans == 0:
                break

            new_img = new_pic(imOrig255, nQuant, z, q)
            mse = calc_mse(nQuant, z, q, hist, pixel_num)
            if abs(old_mse - mse) < EPSILON or mse > old_mse:
                break

            list_of_mse.append(mse)
            list_of_pic.append(new_img)
            old_mse, old_q = mse, q
        return list_of_pic, list_of_mse
    else:
        yiq = transformRGB2YIQ(imOrig)
        y = yiq[:, :, 0]
        y255 = normalize_image(y) * NUM_OF_PIXELS_255
        hist, edges = np.histogram(y255, NUM_OF_PIXELS_256, [0, NUM_OF_PIXELS_256])
        cum_sum = np.cumsum(hist)
        z = origin_z(pixel_num, nQuant, cum_sum, z)
        q, ans = find_new_q(z, q, hist, pixel_num)
        new_img = new_pic(y255, nQuant, z, q)
        yiq[:, :, 0] = normalize_image(new_img)
        new_image = normalize_image(transformYIQ2RGB(yiq))
        temp_mse = calc_mse(nQuant, z, q, hist, pixel_num)
        list_of_pic.append(new_image)
        list_of_mse.append(temp_mse)

        old_mse, old_img, old_q = temp_mse, new_img, q
        i = 1
        while i < nIter:
            z = next_z(z, q)
            q, ans = find_new_q(z, q, hist, pixel_num)
            if np.array_equal(q, old_q) and ans == 0:
                break

            new_img = new_pic(y255, nQuant, z, q)
            mse = calc_mse(nQuant, z, q, hist, pixel_num)
            if abs(old_mse - mse) < EPSILON:
                break
            if mse > old_mse:
                new_img = old_img
                print("mse got bigger")
                break
            list_of_mse.append(mse)
            yiq[:, :, 0] = normalize_image(new_img)
            new_image = normalize_image(transformYIQ2RGB(yiq))
            list_of_pic.append(new_image)
            old_mse, old_img, old_q = mse, new_img, q
            i += 1

        return list_of_pic, list_of_mse
