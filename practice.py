# encoding:utf-8

# img = cv2.imread('image.jpg', 1)
# cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# img = cv2.imread("image.jpg", 1)
# img = img[:, :, ::-1]
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# plt.imshow(img, cmap='gray', interpolation='bicubic')
# plt.xticks([]), plt.yticks([])
# plt.show()

# cap = cv2.VideoCapture(0)
#
# while(True):
#     ret, frame = cap.read()
#
#     # print ret
#
#     # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#     cv2.imshow('frame', frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()

# img = np.zeros((512, 512, 3), np.uint8)
#
# cv2.line(img, (0, 0), (511, 511), (255, 0, 0), 5)
# cv2.rectangle(img,(384,0),(510,128),(0,255,0),3)
# cv2.circle(img,(447,63), 63, (0,0,255), -1)
# cv2.ellipse(img,(256,256),(100,50),0,0,250,255,-1)
#
# font=cv2.FONT_HERSHEY_SIMPLEX
# cv2.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2)
# winname = 'example'
# cv2.namedWindow(winname)
# cv2.imshow(winname, img)
# cv2.waitKey(0)
# cv2.destroyWindow(winname)

# def nothing(x):
#     pass
# # 创建一副黑色图像
# img=np.zeros((300,512,3),np.uint8)
# cv2.namedWindow('image')
# cv2.createTrackbar('R','image',0,255,nothing)
# cv2.createTrackbar('G','image',0,255,nothing)
# cv2.createTrackbar('B','image',0,255,nothing)
# switch='0:OFF\n1:ON'
# cv2.createTrackbar(switch,'image',0,1,nothing)
# while(1):
#     cv2.imshow('image',img)
#     k=cv2.waitKey(1)&0xFF
#     if k==27:
#         break
#     r=cv2.getTrackbarPos('R','image')
#     g=cv2.getTrackbarPos('G','image')
#     b=cv2.getTrackbarPos('B','image')
#     s=cv2.getTrackbarPos(switch,'image')
#     if s==0:
#         img[:]=0
#     else:
#         img[:]=[b,g,r]
# cv2.destroyAllWindows()

# 当鼠标按下时变为True
# drawing=False
# # 如果mode 为true 绘制矩形。按下'm' 变成绘制曲线。
# mode=True
# ix,iy=-1,-1
# # 创建回调函数
# def draw_circle(event,x,y,flags,param):
#     r=cv2.getTrackbarPos('R','image')
#     g=cv2.getTrackbarPos('G','image')
#     b=cv2.getTrackbarPos('B','image')
#     color=(b,g,r)
#     global ix,iy,drawing,mode
#     # 当按下左键是返回起始位置坐标
#     if event==cv2.EVENT_LBUTTONDOWN:
#         drawing=True
#         ix,iy=x,y
#     # 当鼠标左键按下并移动是绘制图形。event 可以查看移动，flag 查看是否按下
#     elif event==cv2.EVENT_MOUSEMOVE and flags==cv2.EVENT_FLAG_LBUTTON:
#         if drawing==True:
#             if mode==True:
#                 cv2.rectangle(img,(ix,iy),(x,y),color,-1)
#             else:
#                 # 绘制圆圈，小圆点连在一起就成了线，3 代表了笔画的粗细
#                 cv2.circle(img,(x,y),3,color,-1)
#     # 下面注释掉的代码是起始点为圆心，起点到终点为半径的
#     # r=int(np.sqrt((x-ix)**2+(y-iy)**2))
#     # cv2.circle(img,(x,y),r,(0,0,255),-1)
#     # 当鼠标松开停止绘画。
#         elif event==cv2.EVENT_LBUTTONUP:
#             drawing = False
#     # if mode==True:
#     # cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
#     # else:
#     # cv2.circle(img,(x,y),5,(0,0,255),-1)
# img=np.zeros((512,512,3),np.uint8)
# cv2.namedWindow('image')
# cv2.createTrackbar('R','image',0,255,nothing)
# cv2.createTrackbar('G','image',0,255,nothing)
# cv2.createTrackbar('B','image',0,255,nothing)
# cv2.setMouseCallback('image',draw_circle)
# while(1):
#     cv2.imshow('image',img)
#     k=cv2.waitKey(1)&0xFF
#     if k==ord('m'):
#         mode=not mode
#     elif k==27:
#         break
#
# cv2.destroyAllWindows()

# cap=cv2.VideoCapture(0)
# while(1):
#     # 获取每一帧
#     ret,frame=cap.read()
#     # 转换到HSV
#     hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
#     # 设定蓝色的阈值
#     lower_blue=np.array([110,50,50])
#     upper_blue=np.array([130,255,255])
#     # 根据阈值构建掩模
#     mask=cv2.inRange(hsv,lower_blue,upper_blue)
#     # 对原图像和掩模进行位运算
#     res=cv2.bitwise_and(frame,frame,mask=mask)
#     # 显示图像
#     cv2.imshow('frame',frame)
#     cv2.imshow('mask',mask)
#     cv2.imshow('res',res)
#     k=cv2.waitKey(5)&0xFF
#     if k==27:
#         break
# # 关闭窗口
# cv2.destroyAllWindows()

# img = cv2.imread('image.jpg', 0)
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# cl1 = clahe.apply(img)
#
# cv2.imwrite('image2.jpg', cl1)

# img = cv2.imread('image.jpg', 0)
# img2 = img.copy()
# template = cv2.imread('image_face.png', 0)
# w, h = template.shape[::-1]
#
# img = img2.copy()
#
# res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF)
# min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
# # 使用不同的比较方法，对结果的解释不同
# # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
#
# bottom_right = (max_loc[0] + w, max_loc[1] + h)
# cv2.rectangle(img, max_loc, bottom_right, 255, 2)
# plt.subplot(121), plt.imshow(res, cmap='gray')
# plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(img, cmap='gray')
# plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
# plt.suptitle('Test')
# plt.show()

# def drawMatches(img1, kp1, img2, kp2, matches):
#     """
#     My own implementation of cv2.drawMatches as OpenCV 2.4.9
#     does not have this function available but it's supported in
#     OpenCV 3.0.0
#
#     This function takes in two images with their associated
#     keypoints, as well as a list of DMatch data structure (matches)
#     that contains which keypoints matched in which images.
#
#     An image will be produced where a montage is shown with
#     the first image followed by the second image beside it.
#
#     Keypoints are delineated with circles, while lines are connected
#     between matching keypoints.
#
#     img1,img2 - Grayscale images
#     kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint
#               detection algorithms
#     matches - A list of matches of corresponding keypoints through any
#               OpenCV keypoint matching algorithm
#     """
#
#     # Create a new output image that concatenates the two images together
#     # (a.k.a) a montage
#     rows1 = img1.shape[0]
#     cols1 = img1.shape[1]
#     rows2 = img2.shape[0]
#     cols2 = img2.shape[1]
#
#     out = np.zeros((max([rows1,rows2]), cols1+cols2, 3), dtype='uint8')
#
#     # Place the first image to the left
#     out[:rows1, :cols1] = np.dstack([img1])
#
#     # Place the next image to the right of it
#     out[:rows2, cols1:] = np.dstack([img2])
#
#     # For each pair of points we have between both images
#     # draw circles, then connect a line between them
#     for mat in matches:
#
#         # Get the matching keypoints for each of the images
#         img1_idx = mat.queryIdx
#         img2_idx = mat.trainIdx
#
#         # x - columns
#         # y - rows
#         (x1, y1) = kp1[img1_idx].pt
#         (x2, y2) = kp2[img2_idx].pt
#
#         # Draw a small circle at both co-ordinates
#         # radius 4
#         # colour blue
#         # thickness = 1
#         cv2.circle(out, (int(x1), int(y1)), 4, (255, 0, 0), 1)
#         cv2.circle(out, (int(x2)+cols1, int(y2)), 4, (255, 0, 0), 1)
#
#         # Draw a line in between the two points
#         # thickness = 1
#         # colour blue
#         cv2.line(out, (int(x1), int(y1)), (int(x2)+cols1, int(y2)), (255, 0, 0), 1)
#
#
#     # Show the image
#     # cv2.imshow('Matched Features', out)
#     # cv2.waitKey(0)
#     # cv2.destroyWindow('Matched Features')
#
#     # Also return the image if you'd like a copy
#     return out
#
# img1 = cv2.imread('box.png', 0) # queryImage
# img2 = cv2.imread('box_in_scene.png', 0) # trainImage
# # Initiate SIFT detector
# sift = cv2.SIFT()
# # find the keypoints and descriptors with SIFT
# kp1, des1 = sift.detectAndCompute(img1, None)
# kp2, des2 = sift.detectAndCompute(img2, None)
# # BFMatcher with default params
# bf = cv2.BFMatcher()
# matches = bf.knnMatch(des1, des2, k=2)
# # Apply ratio test
# # 比值测试，首先获取与A 距离最近的点B（最近）和C（次近），只有当B/C
# # 小于阈值时（0.75）才被认为是匹配，因为假设匹配是一一对应的，真正的匹配的理想距离为0
# good = []
# for m, n in matches:
#     if m.distance < 0.75*n.distance:
#         good.append(m)
# # cv2.drawMatchesKnn expects list of lists as matches.
# img3 = drawMatches(img1, kp1, img2, kp2, good[:10])
# plt.imshow(img3), plt.show()
# -*- coding: utf-8 -*-

import cv2
import numpy as np
cap=cv2.VideoCapture(0)
while(1):
    # 获取每一帧
    ret,frame = cap.read()
    # frame = cv2.imread('IMG_3776.JPG', 1)
    # 转换到HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cimg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 设定蓝色的阈值
    # lower_blue=np.array([110,50,50])
    # upper_blue=np.array([130,255,255])
    # lower_blue = np.array([26, 43, 46])
    # upper_blue = np.array([34, 255, 255])
    # lower_blue = np.array([0, 130, 50])
    # upper_blue = np.array([34, 255, 255])
    # lower_blue = np.array([30, 4, 219])
    # upper_blue = np.array([51, 150, 255])

    lower_blue = np.array([20, 241, 129])
    upper_blue = np.array([25, 255, 157])
    # 根据阈值构建掩模
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    img, contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    area_max_contour = 0
    # if  contours:
    #     a = contours[0]
    #     x,y,w,h = cv2.boundingRect(a)
    #
    #     for cnt in contours:
    #         contour_area_temp = np.fabs(cv2.contourArea(cnt))
    #         x, y, w, h = cv2.boundingRect(cnt)
    #         # print x,y,w,h
    #         # 轮廓外接矩形宽长相差不超过指定像素(根据实际情况调整)
    #         if np.fabs(w - h) < 80:
    #             # 找寻满足条件的最大外接矩形面积的轮廓
    #             if contour_area_temp > area_max_contour:
    #                 area_max_contour = contour_area_temp
    #                 a = cnt
    #     # print a
    #     # (x, y) 轮廓a的外接矩形左上角的点   w为宽 h为高
    #     x,y,w, h = cv2.boundingRect(a)
    # # print x,y,w,h
    # # print '123455655'
    # if w+h>100:
    #     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    if  contours:
        a = contours[0]
        x,y,w,h = cv2.boundingRect(a)

        for cnt in contours:
            # contour_area_temp = np.fabs(cv2.contourArea(cnt))
            x, y, w, h = cv2.boundingRect(cnt)
            print x,y,w,h
            if w + h > 20:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # print x,y,w,h
            # 轮廓外接矩形宽长相差不超过指定像素(根据实际情况调整)
            # if np.fabs(w - h) < 80:
            #     # 找寻满足条件的最大外接矩形面积的轮廓
            #     if contour_area_temp > area_max_contour:
            #         area_max_contour = contour_area_temp
            #         a = cnt
        # print a
        # (x, y) 轮廓a的外接矩形左上角的点   w为宽 h为高
        # x,y,w, h = cv2.boundingRect(a)
    # print x,y,w,h
    # print '123455655'



    # cv2.HoughCircles(cimg, cv2.HOUGH_STANDARD, 1, 20, circles, param1=50, param2=30, minRadius=0, maxRadius=0)
    # circles1 = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 1,
    #                             100, param1=100, param2=30, minRadius=200, maxRadius=300)
    # print type(circles1)
    # if circles1 is not None:
    #     circles = circles1[0, :, :]  # 提取为二维
    #     circles = np.uint16(np.around(circles))  # 四舍五入，取整
    #     for i in circles[:]:
    #         # draw the outer circle
    #         cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
    #         # draw the center of the circle
    #         cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)
    #     # 对原图像和掩模进行位运算

    # 显示图像
    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)
    k=cv2.waitKey(5) & 0xFF
    if k==27:
        break
# 关闭窗口
# frame = cv2.imread('123.png', 1)
# cv2.imshow('frame', frame)
# cv2.waitKey(0)
cv2.destroyAllWindows()