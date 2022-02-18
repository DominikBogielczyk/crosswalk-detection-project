import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

paths = ['road122.png', 'road123.png', 'road123.png', 'road125.png', 'road126.png',
         'road127.png', 'road128.png', 'road129.png', 'road130.png', 'road131.png',
         'road132.png', 'road133.png', 'road134.png', 'road135.png', 'road136.png',
         'road137.png', 'road138.png', 'road139.png', 'road140.png', 'road141.png',
         'road142.png', 'road143.png', 'road144.png', 'road145.png', 'road146.png',
         'road147.png', 'road148.png', 'road149.png', 'road150.png', 'road151.png',
         'road152.png', 'road153.png', 'road154.png', 'road155.png']

for n in range(len(paths)):
    img = cv2.imread(os.path.join('test/images/', paths[n]))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # setting threshold of gray image
    _, threshold = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)
    _, threshold2 = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY)

    res = cv2.bitwise_and(threshold, threshold2)

    contours, _ = cv2.findContours(res, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    print(paths[n])
    k = 0

    for i in range(len(contours)):
        if i != 0:
            x, y, w, h = cv2.boundingRect(contours[i])

            approx = cv2.approxPolyDP(contours[i], 0.01 * cv2.arcLength(contours[i], True), True)
            if (1200 < cv2.contourArea(contours[i])) and (0.6 < (w / h) < 1.6) and k == 0 and \
                    (len(approx) == 4 or len(approx) == 5 or len(approx) == 7):
                print(int(x), int(x+w), int(y), int(y+h))
                cv2.drawContours(img, [contours[i]], 0, (0, 0, 255), 4)
                k = k + 1

    if k == 0:
        print('Classified but not detected')


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

