import sys
from tree import io, algo

IMAGE_DIR = "./photos"


def main(NUM_IMG=None):
    img_paths = io.load_path(IMAGE_DIR, MAX=NUM_IMG)
    imgs = io.load_photos(img_paths)
    keypoints, descriptions, _ = algo.get_feature_points(imgs)
    matches = algo.match_features(descriptions)
    aligned_imgs, hs = algo.align_images(keypoints, matches, imgs)
    io.view_photos(aligned_imgs, t=0.5, save=True)


if __name__ == '__main__':
    NUM_IMG = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(NUM_IMG)
