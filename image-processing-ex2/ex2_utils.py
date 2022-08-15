import numpy as np
import cv2
import math

BASE_KERNEL_DERV = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]])
OP_BASE_KERNEL_DERV = np.array([[-1, 0, 1], [0, 0, 0], [1, 0, -1]])
LAPLACIAN_KERNEL = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
IMAGE_PIXELS_SIZE = 255
STEPS = 100
STEPS_2 = 200
CIRCLE_THRESH_RATIO = 0.46
PI = math.pi


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return MY_ID


# help functions
def help_zero_crossing(in_image: np.ndarray) -> np.ndarray:
    """
    help zero crossing function on binary image
    :param in_image: input image
    :return:
    """
    negative = in_image < 0
    positive = in_image >= 0
    results = np.zeros_like(in_image)
    for i in range(1, in_image.shape[0]):
        for j in range(1, in_image.shape[1]):
            case_negative_i = negative[i - 1, j] and positive[i, j]
            case_positive_i = positive[i - 1, j] and negative[i, j]
            case_negative_j = negative[i, j - 1] and positive[i, j]
            case_positive_j = positive[i, j - 1] and negative[i, j]
            if case_negative_i or case_positive_i or case_negative_j or case_positive_j:
                results[i, j] = 1
    return results


def help_create_gaussian(k_size: int, sigma: float) -> np.ndarray:
    mid = k_size // 2
    kernel = np.zeros((k_size, k_size))
    for i in range(k_size):
        for j in range(k_size):
            x = i - mid
            y = j - mid
            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)) / (2 * PI * sigma ** 2)
    return kernel


def help_gaussian(x, sigma):
    return math.exp(- (x ** 2) / (2 * sigma ** 2)) * (1.0 / (2 * PI * (sigma ** 2)))


def get_sigma_blur_image(k_size: int) -> float:
    return 0.3 * ((k_size - 1) * 0.5 - 1) + 0.8


def conv1D(in_signal: np.ndarray, k_size: np.ndarray) -> np.ndarray:
    """
    Convolve a 1-D array with a given kernel
    :param in_signal: 1-D array
    :param k_size: 1-D array as a kernel
    :return: The convolved array
    """
    kernel_size = len(k_size)
    signal = np.pad(k_size, (kernel_size - 1, kernel_size - 1), )
    signal_size = len(signal)
    conv = np.zeros(signal_size - kernel_size + 1)
    k = 0
    while k < len(conv):
        conv[k] = (signal[k:k + kernel_size] * k_size).sum()
        k += 1
    return conv


def conv2D(in_image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param in_image: 2D image
    :param kernel: A kernel
    :return: The convolved image
    """
    flipped_kernel = np.flip(kernel)
    kernel_height, kernel_width = flipped_kernel.shape
    image_height, image_width = in_image.shape
    image_padded = np.pad(in_image, (kernel_height // 2, kernel_width // 2), "edge")
    image_conv = np.zeros((image_height, image_width))
    i = 0
    while i < image_height:
        j = 0
        while j < image_width:
            image_conv[i, j] = (image_padded[i:i + kernel_height, j:j + kernel_width] * flipped_kernel).sum()
            j += 1
        i += 1
    return image_conv


def convDerivative(in_image: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculate gradient of an image
    :param in_image: Grayscale image
    :return: (directions, magnitude)
    """
    kernel = BASE_KERNEL_DERV
    transposed_kernel = kernel.transpose()
    dx, dy = conv2D(in_image, kernel), conv2D(in_image, transposed_kernel)
    directions = np.arctan(dy, dx)
    magnitude = np.sqrt(np.square(dy) + np.square(dx))
    return directions, magnitude


def blurImage1(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """
    return conv2D(in_image, help_create_gaussian(k_size=k_size, sigma=get_sigma_blur_image(k_size=k_size)))


def blurImage2(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """
    kernel = cv2.getGaussianKernel(k_size, int(round(get_sigma_blur_image(k_size))))
    return cv2.filter2D(in_image, -1, kernel, borderType=cv2.BORDER_REPLICATE)


def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossing" method
    :param img: Input image
    :return: Edge matrix
    """
    img = conv2D(blurImage2(img, 5), OP_BASE_KERNEL_DERV)
    return help_zero_crossing(img)


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossingLOG" method
    :param img: Input image
    :return: Edge matrix
    """
    # blur = blurImage2(img, np.array([3, 3]))
    # blur = blurImage2(img, 3)
    # return edgeDetectionZeroCrossingSimple(blur)
    img = conv2D(blurImage2(img, 11), LAPLACIAN_KERNEL)
    return help_zero_crossing(img)


def houghCircle(img: np.ndarray, min_radius: int, max_radius: int) -> list:
    """
    Find Circles in an image using a Hough Transform algorithm extension
    To find Edges you can Use OpenCV function: cv2.Canny
    :param img: Input image
    :param min_radius: Minimum circle radius
    :param max_radius: Maximum circle radius
    :return: A list containing the detected circles, [(x,y,radius),(x,y,radius),...]
    """
    out_image = (img * IMAGE_PIXELS_SIZE).astype(np.uint8)
    canny_image = cv2.Canny(out_image, STEPS, STEPS_2)
    rows, cols = canny_image.shape
    edges, points, results, circles = [], [], [], {}

    for radius in range(min_radius, max_radius + 1):
        for step in range(STEPS):
            x = int(np.cos(PI * (step / STEPS) * 2) * radius)
            y = int(np.sin(PI * (step / STEPS) * 2) * radius)
            points.append((x, y, radius))

    for x in range(rows):
        for y in range(cols):
            if canny_image[x, y] == IMAGE_PIXELS_SIZE:
                edges.append((x, y))

    for x1, y1 in edges:
        for x2, y2, radius in points:
            dx, dy = x1 - x2, y1 - y2
            circle = circles.get((dy, dx, radius))
            if circle is None:
                circles[(dy, dx, radius)] = 1
            else:
                circles[(dy, dx, radius)] = circle + 1

    for circle, count in sorted(circles.items(), key=lambda v: -v[1]):
        nx, ny, radios = circle
        if count / STEPS >= CIRCLE_THRESH_RATIO and all((nx - x) ** 2 + (ny - y) ** 2 > r ** 2 for x, y, r in results):
            results.append((nx, ny, radios))

    return results


def bilateral_filter_implement(in_image: np.ndarray, k_size: int, sigma_color: float, sigma_space: float) -> (
        np.ndarray, np.ndarray):
    """
    :param in_image: input image
    :param k_size: Kernel size
    :param sigma_color: represents the filter sigma in the color space.
    :param sigma_space: represents the filter sigma in the coordinate.
    :return: OpenCV implementation, my implementation
    """
    out_image_open_cv = cv2.bilateralFilter(in_image, k_size, sigma_color, sigma_space)
    out_image = np.zeros(in_image.shape)
    x_len, y_len = len(in_image), len(in_image[0])
    for x in range(x_len):
        for y in range(y_len):
            filtered_sum = 0
            sum_of_gaussian = 0
            for i in range(k_size):
                for j in range(k_size):
                    nx = int(x - ((k_size / 2) - i))
                    ny = int(y - ((k_size / 2) - j))
                    if nx >= x_len:
                        nx = x_len - 1
                    if nx < 0:
                        nx = 0

                    if ny >= y_len:
                        ny = y_len - 1
                    if ny < 0:
                        ny = 0

                    gaussian_1 = help_gaussian(int(in_image[nx][ny]) - int(in_image[x][y]), sigma_color)
                    gaussian_2 = help_gaussian(np.sqrt((nx - x) ** 2 + (ny - y) ** 2), sigma_space)
                    filtered_sum += in_image[nx][ny] * (gaussian_1 * gaussian_2)
                    sum_of_gaussian += gaussian_1 * gaussian_2
            out_image[x][y] = int(round(filtered_sum / sum_of_gaussian))
    return out_image_open_cv, out_image
