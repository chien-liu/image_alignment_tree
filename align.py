""" Align Photos
Apply image alignment algorithm to every image in source directory.

Algorithm
---------
1. Feature extraction with scale-invariant feature transform (SIFT)
2. Feature matching with brute force matcher
3. Determine homography transformation with Random sample consensus (RANSAC) 
"""
import os
import argparse
from tqdm import tqdm
import cv2
from utils import io, algo

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--srcdir", type=str,
                    default="./TreePhotos", help="Directory to tree photos.")
parser.add_argument("-o", "--outputdir", type=str,
                    default="./Output", help="Directory to save output photos.")
args = parser.parse_args()


def run():
    """Main function"""
    if not os.path.isdir(args.outputdir):
        os.mkdir(args.outputdir)
    img_paths = io.load_path(args.srcdir)
    target_path = "./TreePhotos/IMG_8268.JPG"
    target = io.load_photo(target_path)
    t_keypoints, t_descriptions, t_featured_img = algo.get_feature_points(target)
    
    for i, path in enumerate(tqdm(img_paths)):
        img = io.load_photo(path)
        keypoints, descriptions, featured_img = algo.get_feature_points(img)
        match = algo.match_features(t_descriptions, descriptions)        
        aligned_img, h = algo.align_image(target, t_keypoints, img, keypoints, match)

        opath = os.path.join(args.outputdir, f"{i:04}.jpg")
        cv2.imwrite(opath, aligned_img)


if __name__ == '__main__':
    run()
