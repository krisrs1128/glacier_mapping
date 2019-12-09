import numpy as np
import random
import cv2

def to_numpy_img(img):
    """From tensor to numpy, with channels last"""

    img = img.data.numpy()
    img = np.moveaxis(img, 0, -1)

    return img

def rotate(img, mask, rot=(-10, 10), p=0.5):
    """Rotate image and crossponding mask with the same random angle"""

    if random.random() > p:
        angle = random.uniform(rot[0], rot[1])
        w, h = mask.shape
        center = int(w / 2), int(h / 2)

        rot_mat = cv2.getRotationMatrix2D(center, angle, 1)
        rotated_img = cv2.warpAffine(img, rot_mat, (w, h))
        rotated_mask = cv2.warpAffine(mask, rot_mat, (w, h),
                                      flags=cv2.INTER_NEAREST)
        return rotated_img, rotated_mask

    return img, mask

def flip(img, mask, direction, percent=0.5):
    """Flip image and crossponding mask"""
    
    p = random.random()
    if p > percent:
        img = cv2.flip(img, direction)
        mask = cv2.flip(mask, direction)

        return img, mask
    return img, mask