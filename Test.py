# #encoding:utf-8
# import cv2
# import numpy as np
#
# img = cv2.imread('eye.png')
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#灰度图像
# print '123'
# cv2.imshow('gray',gray)
# cv2.waitKey(0)
# #hough transform
# circles1 = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,
# 100,param1=100,param2=30,minRadius=200,maxRadius=300)
# circles = circles1[0,:,:]#提取为二维
# circles = np.uint16(np.around(circles))#四舍五入，取整
# for i in circles[:]:
#     cv2.circle(img,(i[0],i[1]),i[2],(255,0,0),5)#画圆
#     cv2.circle(img,(i[0],i[1]),2,(255,0,255),10)#画圆心
#
# cv2.imshow('img',img)
# cv2.waitKey(0)

import cv2
import numpy as np
img = cv2.imread('opencv-logo.jpg', 0)
img = cv2.medianBlur(img,5)
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=0,maxRadius=0)
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
cv2.imshow('circles', cimg)
cv2.waitKey(0)
cv2.destroyAllWindows()
