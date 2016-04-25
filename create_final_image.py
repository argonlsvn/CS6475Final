__author__ = 'Anh Nguyen'

import cv2
import numpy as np,sys
import scipy as sp

src1 = cv2.imread("images/portrait.jpg")
src2 = cv2.imread("images/blend.jpg")
final_img = src1.copy()
for i in range(0,len(src1)):
    for j in range(0,len(src1[0])):
        final_img[i][j] = 0.5*src1[i][j] + 0.5*src2[i][j]
        #print src1[i][j], src2[i][j]
cv2.imwrite('images/final_img.jpg', final_img)