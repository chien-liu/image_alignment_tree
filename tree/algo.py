import numpy as np
import cv2
from .io import print_progress

TARGET = 25


def get_feature_points(images: list):
    print("Start getting features for each image")
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # Mask
    h, w = images[0].shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    h, w = int(h/5), int(w/5)
    mask[h: -h, w: -w] = 255
    featured_images = []
    keypoints = []
    descriptions = []
    for i, img in enumerate(images):
        print_progress(i, len(images))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        blur = cv2.blur(gray, (5, 5))
        kp, des = sift.detectAndCompute(blur, mask)
        img = cv2.drawKeypoints(img, kp, img)
        featured_images.append(img)
        keypoints.append(kp)
        descriptions.append(des)
    return keypoints, descriptions, featured_images


def match_features(descriptions, GOOD_RATIO=0.15, TARGET=TARGET):
    print("Start matching features")
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = []
    des1 = descriptions[TARGET]
    for i in range(len(descriptions)):
        print_progress(i, len(descriptions))
        des2 = descriptions[i]
        match = bf.match(des1, des2)
        match = sorted(match, key=lambda x: x.distance)
        match = match[:int(len(match) * GOOD_RATIO)]
        matches.append(match)
    return matches


def demo_matched_features(keypoints, matches, images: list, TARGET=TARGET):
    matched_images = []
    img1 = images[TARGET]
    kp1 = keypoints[TARGET]
    for i in range(len(images)):
        img2 = images[i]
        kp2 = keypoints[i]
        match = matches[i]
        img3 = cv2.drawMatches(
            img1, kp1, img2, kp2, match[:100],
            None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        img3 = cv2.resize(img3, (1344, 1344))  # For IG display
        matched_images.append(img3)
    return matched_images


def align_images(keypoints, matches, images: list, TARGET=TARGET):
    print("Start aligning images")
    aligned_imgs = []
    hs = []
    img1 = images[TARGET]
    kp1 = keypoints[TARGET]

    for i in range(len(images)):
        print_progress(i, len(images))
        img2 = images[i]
        kp2 = keypoints[i]
        match = matches[i]
        # Extract location of good matches
        points1 = np.zeros((len(match), 2), dtype=np.float32)
        points2 = np.zeros((len(match), 2), dtype=np.float32)
        for j, m in enumerate(match):
            points1[j, :] = kp1[m.queryIdx].pt
            points2[j, :] = kp2[m.trainIdx].pt
        # Find homography
        h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)
        hs.append(h)

        # Use homography
        height, width, channels = img1.shape
        img2Reg = cv2.warpPerspective(img2, h, (width, height))

        aligned_imgs.append(img2Reg)
        hs.append(h)

    return aligned_imgs, hs
