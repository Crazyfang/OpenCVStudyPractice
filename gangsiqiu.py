#encoding:utf-8
import numpy as np
import cv2
from matplotlib import pyplot as plt
# img1 = cv2.imread('IMG_3781.JPG', 0) # queryImage
# img2 = cv2.imread('IMG_3776.JPG', 0) # trainImage
# img1 = cv2.imread('box.png', 0) # queryImage
# img2 = cv2.imread('box_in_scene.png', 0) # trainImage
img1 = cv2.imread('image2.jpg', 0) # queryImage
img2 = cv2.imread('image_face.png', 0) # trainImage
# Initiate SIFT detector
sift = cv2.xfeatures2d.SURF_create()
MIN_MATCH_COUNT = 20
# cv2.xfeatures2d.SURF_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
# BFMatcher with default params
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(np.asarray(des1,np.float32),np.asarray(des2,np.float32),k=2)
# Apply ratio test
# 比值测试，首先获取与A 距离最近的点B（最近）和C（次近），只有当B/C
# 小于阈值时（0.75）才被认为是匹配，因为假设匹配是一一对应的，真正的匹配的理想距离为0
good = []
for m,n in matches:
    if m.distance < 0.73*n.distance:
        good.append(m)
print good
if len(good) > MIN_MATCH_COUNT:
    # 获取关键点的坐标
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    print src_pts
    print dst_pts
    # 第三个参数Method used to computed a homography matrix. The following methods are possible:
    # 0 - a regular method using all the points
    # CV_RANSAC - RANSAC-based robust method
    # CV_LMEDS - Least-Median robust method
    # 第四个参数取值范围在1 到10，􁲁绝一个点对的阈值。原图像的点经过变换后点与目标图像上对应点的误差
    # 超过误差就认为是outlier
    # 返回值中M 为变换矩阵。
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    # 获得原图像的高和宽
    h, w = img1.shape
    # 使用得到的变换矩阵对原图像的四个角进行变换，获得在目标图像上对应的坐标。
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    # 原图像为灰度图
    cv2.polylines(img2, [np.int32(dst)], True, 255, 1, cv2.LINE_AA)
    print "寻找到目标图像"
else:
    print "Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT)
    matchesMask = None
# # cv2.drawMatchesKnn expects list of lists as matches.
# img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good[:10],flags=2)
draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                    singlePointColor = None,
                    matchesMask = matchesMask, # draw only inliers
                    flags = 2)
img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
plt.imshow(img3, 'gray'), plt.show()
