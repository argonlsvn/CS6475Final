__author__ = 'Anh Nguyen'

import cv2
import numpy as np,sys
import scipy as sp

def get_mask(image):
    img_hsv=cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_bgr = np.array([0,0,0])
    upper_bgr = np.array([255,255,200])
    mask0 = cv2.inRange(img_hsv, lower_bgr, upper_bgr)

    lower_hsv = np.array([0,0,220])
    upper_hsv = np.array([20,125,255])
    mask1 = cv2.inRange(img_hsv, lower_hsv, upper_hsv)

    # join my masks
    #mask = mask0+mask1+mask2
    mask = mask0 + mask1
    #cv2.imwrite('mask.jpg', mask)
    return mask


source1 = cv2.imread("khang2.jpg")
source2 = cv2.imread("brick1.jpg")
cv2.imwrite('images/portrait.jpg', source1)
mask1 = get_mask(source1)
img = cv2.cvtColor( source1, cv2.COLOR_RGB2GRAY)
cv2.imwrite('images/mask.jpg', mask1)

height1, width1 = source1.shape[:2]
height2, width2 = source2.shape[:2]
print height1, width1, height2, width2
crop_source2 = source2[(height2-height1)/2:(height2-height1)/2+height1,(width2-width1)/2:(width2-width1)/2+width1]
cv2.imwrite('images/crop.jpg', crop_source2)

#final_img = source1.copy()
#for i in range(0,len(source1)):
#    for j in range(0,len(source1[0])):
#        if mask1[i][j] > 100:
#            final_img[i][j] = crop_source2[i][j]
#            final_img[i][j] = img[i][j]

#cv2.imwrite('final_img.jpg', final_img)

