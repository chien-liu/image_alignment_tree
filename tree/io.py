import os
import re
import cv2


def load_path(img_dir: list, MAX: int=None) -> list:
    img_path = []
    for root, dirs, files in os.walk(img_dir, topdown=True):
        for name in files:
            if name[-4:] == ".JPG":
                img_path.append(os.path.join(root, name))
    img_path.sort(key=lambda path: re.search(r'\d+', path).group(0))
    if MAX:
        MAX = min(MAX, len(img_path))
    return img_path[:MAX] if MAX else img_path


def load_photos(img_path: list) -> list:
    print("Start loading images to memory")
    L = len(img_path)

    images = []
    for i in range(L):
        print_progress(i, L, len_bar=20)

        name = img_path[i]
        img = cv2.imread(name)
        h, w = img.shape[0], img.shape[1]
        h, w = int(h/3), int(w/3)
        img = cv2.resize(img, (w, h))
        images.append(img)
    return images


def view_photos(images: list, t=1, save=False):
    n = len(str(len(images)))
    for i, img in enumerate(images):
        if save:
            cv2.imwrite(f"./outputs/{str(i).zfill(n)}.jpg", img)
        cv2.imshow("output", img)
        if cv2.waitKey(int(t*1000)) & 0xFF == ord("q"):
            break
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def print_progress(i, total, len_bar=20):
    progress = int(i/total * len_bar)
    if i < total-1:
        print("Progress [" + "#" * progress + ">" + " " * (len_bar - progress - 1) + "]", end="\r")
    else:
        print("Progress [" + "#" * len_bar + "]")
