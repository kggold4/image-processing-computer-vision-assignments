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
import cv2

from ex1_utils import LOAD_GRAY_SCALE, LOAD_RGB, normalize_image

TRACKER_POS = 100.0
TRACKER_MAX = 200
WINDOW_NAME = 'image'
GAMMA = 'gamma'
BAC_CON_IMAGE = 'bac_con.png'


def pass_function():
    """
    function that do nothing
    :return:
    """
    pass


def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    cv2.namedWindow(WINDOW_NAME)
    cv2.createTrackbar(GAMMA, WINDOW_NAME, 0, TRACKER_MAX, pass_function)
    while True:
        gamma = cv2.getTrackbarPos(GAMMA, WINDOW_NAME) / TRACKER_POS
        if rep == LOAD_GRAY_SCALE:
            img_gray = normalize_image(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE))
            img_gray_copy = img_gray ** gamma
            cv2.imshow(WINDOW_NAME, img_gray_copy)
        elif rep == LOAD_RGB:
            img_color = normalize_image(cv2.imread(img_path))
            img_color_copy = img_color ** gamma
            cv2.imshow(WINDOW_NAME, img_color_copy)
        cv2.waitKey(1)


def main():
    gammaDisplay(BAC_CON_IMAGE, LOAD_GRAY_SCALE)


if __name__ == '__main__':
    main()
