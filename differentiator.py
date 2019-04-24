import cv2
import imutils
import argparse
import numpy as np
from skimage.measure import compare_ssim

def args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--original", required=True, help="Path to original image")
    ap.add_argument("-c", "--changed", required=True, help="Path to changed image")

    return vars(ap.parse_args())

def display_imgs(args):
    org_image = cv2.imread(args['original'])
    org_image = cv2.cvtColor(org_image, cv2.COLOR_BGR2GRAY)

    cha_image = cv2.imread(args['changed'])
    cha_image = cv2.cvtColor(cha_image, cv2.COLOR_BGR2GRAY)

    (score, diff) = compare_ssim(org_image, cha_image, full=True)
    diff = (diff * 255).astype("uint8")

    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(org_image, (x, y), (x+w, y+h), (0, 0, 0), 2)
        cv2.rectangle(cha_image, (x, y), (x+w, y+h), (0, 0, 0), 2)

    rs = np.hstack((org_image, cha_image))

    cv2.imshow(f"Structural Similarity (SSIM): {score}", rs)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    display_imgs(args())
