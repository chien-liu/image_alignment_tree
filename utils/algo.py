from typing import List, NewType
import numpy as np
import cv2
from .io import print_progress

Image = NewType("Image", np.ndarray)

def get_feature_points(img: Image):
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # Mask
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    h, w = h//5, w//5
    mask[h: -h, w: -w] = 255

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    blur = cv2.blur(gray, (5, 5))
    keypoints, descriptions = sift.detectAndCompute(blur, mask)
    featured_img = cv2.drawKeypoints(img, keypoints, np.array([]))

    return keypoints, descriptions, featured_img


def match_features(des1, des2, GOOD_RATIO=0.15):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    match = bf.match(des1, des2)
    match = sorted(match, key=lambda x: x.distance)
    match = match[:int(len(match) * GOOD_RATIO)]
    return match


def align_image(img1, kp1, img2, kp2, match):
    """

    Arguments
    ---------
    img1:   Target image
    kp1 :   Keypoint of target image
    img2:   Aligned image
    kp2 :   Keypoint of aligned image
    """

    # Extract location of good matches
    points1 = np.zeros((len(match), 2), dtype=np.float32)
    points2 = np.zeros((len(match), 2), dtype=np.float32)
    for j, m in enumerate(match):
        points1[j, :] = kp1[m.queryIdx].pt
        points2[j, :] = kp2[m.trainIdx].pt
    # Find homography
    h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

    # Use homography
    height, width, channels = img1.shape
    img2Reg = cv2.warpPerspective(img2, h, (width, height))

    return img2Reg, h
