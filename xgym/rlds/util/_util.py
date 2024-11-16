import os.path as osp
from pprint import pprint

import cv2
import imageio
import numpy as np
import torch
from PIL import Image, ImageEnhance
from scipy.optimize import least_squares
from tqdm import tqdm

import local


def add_col(x):
    return np.concatenate([x, np.ones((*x.shape[:-1], 1))], axis=-1)


def remove_col(x):
    if len(x.shape) == 2:
        return x[:, :-1]
    elif len(x.shape) == 3:
        return x[:, :, :-1]


def apply_uv(image, mat, **rules):
    H, W = image.shape[:2]
    img = cv2.warpPerspective(image, mat[:-1, :-1], **rules)
    return img


def apply_xyz(points, mat):
    points_hom = add_col(points)  # 4d
    # bs = points_hom.shape[0]
    # points_hom = np.einsum("bij,bkj->bki", np.stack([mat] * bs), points_hom)
    points_hom = np.einsum("ij,kj->ki", mat, points_hom)
    points = remove_col(points_hom)  # 3d
    return points


def perspective_projection(focal_length, H, W):
    """computes perspective projection of 3D points"""
    f = focal_length
    P = np.array(
        [
            [f, 0, W / 2, 0],
            [0, f, H / 2, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
        ]
    )
    return P

def apply_persp(points, mat):
    points_hom = add_col(points)  # 4d
    # bs = points_hom.shape[0]
    # points_hom = np.einsum("bij,bkj->bki", np.stack([mat] * bs), points_hom)
    points_hom = np.einsum("ij,kj->ki", mat, points_hom)
    points = remove_col(points_hom)  # 3d
    # points2d = points / points[:, :, -1:]
    points2d = points / points[:, -1:]
    return points2d


def matrix_err(T, P, uv, xyz, a):
    """computes the error between the projection of XYZ and UV
    xyz2 = T @ xyz
    uv2 = P @ xyz2
    minimize a * (uv - uv2) + (1 - a) * (xyz - xyz2)
        minimize the pixel error and deviation from the original XYZ
    """

    T = T.reshape(4, 4)

    xyz = xyz.reshape(-1, 3)

    xyz2 = remove_col(np.einsum("ij,kj->ki", T, add_col(xyz)))
    uv2 = np.einsum("ij,kj->ki", P, add_col(xyz2))
    uv2 = remove_col(uv2)
    uv2 = uv2 / uv2[:, -1:]

    lmatch = uv.flatten() - uv2.flatten()
    l2 = (xyz2 - xyz).flatten()
    errors = (a * lmatch) + (1 - a) * l2
    return errors


def solve_uv2xyz(xyz, P, uv=None, U=None):
    """solves for a matrix T such that U @ P ≈ P @ T"""

    assert not uv is None or not U is None, "Either uv or U must be provided"
    if uv is None:
        uv = np.einsum("ij,kj->ki", U @ P, add_col(xyz.reshape(-1, 3)))
        uv = remove_col(uv)
        uv = uv / uv[:, -1:]

    a = 0.5  # Regularization parameter
    xyz, uv = xyz.flatten(), uv.flatten()

    T0 = np.eye(4).flatten()

    # Run optimization
    result = least_squares(
        matrix_err,
        x0=T0,
        args=(P, uv, xyz, a),
        method="lm",  # Levenberg-Marquardt algorithm
    )

    # Extract optimized XYZ_new
    T = result.x.reshape(4, 4)
    return T


"""
def reprojection_error(XYZ_new_flat, P, uv_new, XYZ_original, a):
    XYZ_original = XYZ_original.reshape(-1, 3)
    bs = XYZ_original.shape[0]
    XYZ_new = XYZ_new_flat.reshape(-1, 3)
    errors = []

    XYZ_new_hom = add_col(XYZ_new)
    uv = np.einsum("ij,kj->ki", P, XYZ_new_hom)
    uv = remove_col(uv)
    uv = uv / uv[:, -1:]

    lmatch = uv_new.flatten() - uv.flatten()
    l2 = (XYZ_new - XYZ_original).flatten()
    errors = (a * lmatch) + (1 - a) * l2
    return errors.flatten().tolist()

"""
