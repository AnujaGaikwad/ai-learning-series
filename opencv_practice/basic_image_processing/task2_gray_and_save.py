import cv2
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(BASE_DIR, "images", "sample.png")

img = cv2.imread(image_path)

if img is None:
    raise FileNotFoundError(image_path)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite(os.path.join(BASE_DIR, "images", "gray_image.png"), gray)

cv2.imshow("Original", img)
cv2.imshow("Gray", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
