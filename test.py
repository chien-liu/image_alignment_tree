import unittest
from operator import itemgetter
import numpy as np
from tree import io, algo


class TestIO(unittest.TestCase):
    def setUp(self):
        self.IMAGE_DIR = "./photos"

    def test_load_image_path(self):
        path = io.load_path(self.IMAGE_DIR)
        self.assertEqual(
            itemgetter(0, -1)(path),
            ('./photos/IMG_8199.JPG', './photos/IMG_8406.JPG')
        )
        path = io.load_path(self.IMAGE_DIR, MAX=1)
        self.assertEqual(
            itemgetter(0, -1)(path),
            ('./photos/IMG_8199.JPG', './photos/IMG_8199.JPG')
        )
        path = io.load_path(self.IMAGE_DIR, MAX=300000)
        self.assertEqual(
            itemgetter(0, -1)(path),
            ('./photos/IMG_8199.JPG', './photos/IMG_8406.JPG')
        )

    def test_load_photos(self):
        path = io.load_path(self.IMAGE_DIR, MAX=1)
        images = io.load_photos(path)
        img = images[0]
        self.assertEqual(img.shape, (1344, 1008, 3))


class TestAlgo(unittest.TestCase):
    def setUp(self):
        self.IMAGE_DIR = "./photos"
        pass

    def test_match_features(self):
        img_paths = io.load_path(self.IMAGE_DIR, MAX=2)
        imgs = io.load_photos(img_paths)
        keypoints, descriptions, featured_imgs = algo.get_feature_points(imgs)
        matches = algo.match_features(descriptions, GOOD_RATIO=0.5, TARGET=0)
        matched_imgs = algo.demo_matched_features(
                                keypoints, matches, featured_imgs, TARGET=0)
        # io.view_photos(matched_imgs, t=0.5, save=True)
        self.assertTrue(type(matched_imgs) == list)
        self.assertTrue(type(matched_imgs[0]) == np.ndarray)