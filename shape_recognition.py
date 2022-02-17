import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

paths = ['road156.png', 'road158.png', 'road160.png', 'road165.png', 'road166.png',
         'road167.png', 'road176.png', 'road178.png', 'road180.png', 'road183.png',
         'road189.png', 'road190.png', 'road193.png', 'road194.png', 'road200.png']

for n in range(len(paths)):
    img = cv2.imread(os.path.join('train/images/', paths[n]))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # setting threshold of gray image
    _, threshold = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)
    _, threshold2 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # using a findContours() function
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(threshold2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    i = 0

    for contour in contours:
        if i == 0:
            i = 1
            continue

        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)

        if (cv2.contourArea(contour) > 30 * 40) and len(approx) <= 5:  # or len(approx) == 5):
            cv2.drawContours(img, [contour], 0, (0, 0, 255), 5)

    i = 0

    for contour in contours2:
        if i == 0:
            i = 1
            continue

        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)

        if (cv2.contourArea(contour) > 30 * 40) and len(approx) <= 5:  # or len(approx) == 5):
            cv2.drawContours(img, [contour], 0, (0, 0, 255), 5)

    # displaying the image after drawing contours
    cv2.imshow(paths[n], img)

cv2.waitKey(0)
cv2.destroyAllWindows()


# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# lower_blue = (100, 50, 0)#(60, 40, 20)#(130, 50, 0)
# upper_blue = (220, 120, 65)#(200, 120, 55)#(220, 120, 55)
# mask = cv2.inRange(img, lower_blue, upper_blue)
# res = cv2.bitwise_and(img, img, mask=mask)
# img[mask > 0] = (0, 0, 255)
# #cv2.imshow("Result", res)
# #cv2.imshow('Crosswalk recognition', img)
#
# from matplotlib.colors import hsv_to_rgb
# lo_square = np.full((10, 10, 3), lower_blue, dtype=np.uint8) / 255.0
# do_square = np.full((10, 10, 3), upper_blue, dtype=np.uint8) / 255.0
#
# plt.subplot(1, 2, 1)
# plt.imshow(hsv_to_rgb(lo_square))
# plt.subplot(1, 2, 2)
# plt.imshow(hsv_to_rgb(do_square))
# #plt.show()
# cv2.waitKey(0)
# cv2.destroyAllWindows()

