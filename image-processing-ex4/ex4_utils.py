import math
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

DIM_3 = 3


def disparitySSD(img_l: np.ndarray, img_r: np.ndarray, disp_range: (int, int), k_size: int) -> np.ndarray:
    """
    img_l: Left image
    img_r: Right image
    range: Minimum and Maximum disparity range. Ex. (10,80)
    k_size: Kernel size for computing the SSD, kernel.shape = (k_size*2+1,k_size*2+1)

    return: Disparity map, disp_map.shape = Left.shape
    """
    if img_l.ndim == DIM_3:
        img_l = img_l[:, :, 0]
        img_r = img_r[:, :, 0]

    dm = np.zeros((img_l.shape[0], img_l.shape[1]))

    for x in range(k_size, img_l.shape[0] - k_size):
        for y in range(k_size, img_l.shape[1] - k_size):
            win_l = img_l[x - k_size:x + k_size + 1, y - k_size:y + k_size + 1]
            ssd = sys.maxsize
            disparity = 0

            for d in range(disp_range[0], disp_range[1]):
                t = 0
                if (y + k_size + 1 - d) < img_r.shape[1] and (y - k_size - d) >= 0:
                    win_r = img_r[x - k_size:x + k_size + 1, y - k_size - d:y + k_size + 1 - d]
                    for u in range(k_size):
                        for v in range(k_size):
                            t = t + ((win_l[u, v] - win_r[u, v]) ** 2)
                if ssd > t:
                    ssd = t
                    disparity = d
            dm[x, y] = disparity

    return dm


def disparityNC(img_l: np.ndarray, img_r: np.ndarray, disp_range: int, k_size: int) -> np.ndarray:
    """
    img_l: Left image
    img_r: Right image
    range: The Maximum disparity range. Ex. 80
    k_size: Kernel size for computing the NormCorolation, kernel.shape = (k_size*2+1,k_size*2+1)

    return: Disparity map, disp_map.shape = Left.shape
    """
    if img_l.ndim == DIM_3:
        img_l = img_l[:, :, 0]
        img_r = img_r[:, :, 0]

    dm = np.zeros((img_l.shape[0], img_l.shape[1]))

    for x in range(k_size, img_l.shape[0] - k_size):
        for y in range(k_size, img_l.shape[1] - k_size):
            win_l_tmp = img_l[x - k_size:x + k_size + 1, y - k_size:y + k_size + 1]
            win_l = win_l_tmp.copy().flatten() - win_l_tmp.mean()
            norm1 = np.linalg.norm(win_l, 2)
            NCC = -1
            disparity = 0
            for d in range(disp_range[0], disp_range[1]):
                NCC_tmp = 0
                if (y + k_size - d) < img_r.shape[1] and (y - k_size - d) >= 0:
                    win_r_tmp = img_r[x - k_size:x + k_size + 1, y - k_size - d:y + k_size + 1 - d]
                    win_r = win_r_tmp.copy().flatten() - win_r_tmp.mean()
                    norms = norm1 * np.linalg.norm(win_r, 2)

                    if norms == 0:
                        NCC_tmp = 0
                    else:
                        np.sum(win_l * win_r) / norms

                if NCC < NCC_tmp:
                    NCC = NCC_tmp
                    disparity = d
            dm[x, y] = disparity

    return dm


def computeHomography(src_pnt: np.ndarray, dst_pnt: np.ndarray) -> (np.ndarray, float):
    """
    Finds the homography matrix, M, that transforms points from src_pnt to dst_pnt.
    returns the homography and the error between the transformed points to their
    destination (matched) points. Error = np.sqrt(sum((M.dot(src_pnt)-dst_pnt)**2))

    src_pnt: 4+ keypoints locations (x,y) on the original image. Shape:[4+,2]
    dst_pnt: 4+ keypoints locations (x,y) on the destenation image. Shape:[4+,2]

    return: (Homography matrix shape:[3,3], Homography error)
    """
    A = []
    for i in range(0, src_pnt.shape[0]):
        Xs, Ys = src_pnt[i]
        Xd, Yd = dst_pnt[i]
        A.append([Xs, Ys, 1, 0, 0, 0, -Xd * Xs, -Xd * Ys, -Xd])
        A.append([0, 0, 0, Xs, Ys, 1, -Yd * Xs, -Yd * Ys, -Yd])
    A = np.array(A)

    _, _, vh = np.linalg.svd(A)
    h = (vh[-1].reshape((3, 3)) / vh[-1, -1])

    src = np.vstack((src_pnt.transpose(), np.ones(src_pnt.shape[0])))
    dst = np.vstack((dst_pnt.transpose(), np.ones(dst_pnt.shape[0])))
    pnt_new = h.dot(src)
    return h, np.sqrt(np.sum((pnt_new / pnt_new[-1] - dst) ** 2))


def error_4():
    print("Didn't get 4 points")


def warpImag(src_img: np.ndarray, dst_img: np.ndarray) -> None:
    """
    Displays both images, and lets the user mark 4 or more points on each image.
    Then calculates the homography and transforms the source image on to the destination image.
    Then transforms the source image onto the destination image and displays the result.

    src_img: The image that will be ’pasted’ onto the destination image.
    dst_img: The image that the source image will be ’pasted’ on.

    output: None.
    """
    dst, src = [], []
    flag = True

    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            if flag:
                dst.append((x, y))
            else:
                src.append((x, y))

    cv2.namedWindow('dst_img', cv2.WINDOW_NORMAL)
    cv2.imshow('dst_img', dst_img)
    cv2.setMouseCallback('dst_img', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if len(dst) != 4:
        error_4()
        return
    rect_warp_image(dst)
    dst_pts = np.array(dst)

    flag = False
    img_name = 'src_img'
    cv2.namedWindow(img_name, cv2.WINDOW_NORMAL)
    cv2.imshow(img_name, src_img)
    cv2.setMouseCallback(img_name, click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if len(src) != 4:
        error_4()
        return
    rect_warp_image(src)
    src_pts = np.array(src)
    h, _ = cv2.findHomography(src_pts, dst_pts)
    for i in range(src_pts[0, 0], src_pts[1, 0]):
        for j in range(src_pts[0, 1], src_pts[2, 1]):
            homo_pixel = np.array([[i], [j], [1]])
            idx_new = h.dot(homo_pixel)
            x, y, _ = (idx_new // idx_new[-1]).astype(int)

            if 0 <= x < dst_img.shape[1] and 0 <= y < dst_img.shape[0]:
                dst_img[y, x] = src_img[j, i]

    plt.imshow(dst_img)
    plt.show()


def rect_warp_image(lst: list):
    length = len(lst)
    mlat = np.sum(x[0] for x in lst) / length
    mlng = sum(x[1] for x in lst) / length

    def algo(x):
        return (math.atan2(x[0] - mlat, x[1] - mlng) + 2 * math.pi) % (2 * math.pi)

    lst.sort(key=algo)
    # swap
    tmp = lst[0]
    lst[0] = lst[2]
    lst[2] = lst[3]
    lst[3] = tmp
