import numpy as np
import cv2


def warp_pts(pts_array, homo_mat):
    homog_pts = np.concatenate([pts_array, np.ones((pts_array.shape[0], 1))], axis=-1)
    warp_homog_pts = np.matmul(homog_pts, homo_mat.T)
    warp_homog_pts /= warp_homog_pts[:, 2:]
    return warp_homog_pts[:, :-1]


def default_corners(img):
    return np.array([[0, 0], [img.shape[1] - 1, 0], [
        0, img.shape[0] - 1], [img.shape[1] - 1, img.shape[0] - 1]])


def adaptive_rot(img_array, random=True, rot=None):
    corners = default_corners(img_array)
    if random:
        rot = np.random.random() * 360
    rot = np.deg2rad(rot)
    s, c = np.sin(rot), np.cos(rot)
    mat = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    rot_corners = warp_pts(corners, mat)
    x, y, w, h = cv2.boundingRect(np.int32(rot_corners))
    translation = np.array([[-x, -y]])
    corners = rot_corners + translation
    mat[0, -1], mat[1, -1] = -x, -y
    rot_img = cv2.warpPerspective(img_array, mat, (w, h))
    cv2.imwrite('./rot.jpg', rot_img)
    return rot_img, corners


def center_square(img, content_corners):
    x_sort = np.int32(np.sort(content_corners[:, 0]))
    y_sort = np.int32(np.sort(content_corners[:, 1]))
    crop = img[y_sort[1]:y_sort[2] + 1, x_sort[1]:x_sort[2] + 1, :].copy()
    w, h = crop.shape[1], crop.shape[0]
    if w > h:
        diff = w - h
        offset = diff // 2
        crop = cv2.copyMakeBorder(crop, top=0, bottom=0, left=offset, right=diff - offset,
                                  borderType=cv2.BORDER_CONSTANT, value=(128, 128, 128))
    elif w < h:
        diff = h - w
        offset = diff // 2
        crop = cv2.copyMakeBorder(crop, top=offset, bottom=diff - offset, left=0, right=0,
                                  borderType=cv2.BORDER_CONSTANT, value=(128, 128, 128))
    return crop
