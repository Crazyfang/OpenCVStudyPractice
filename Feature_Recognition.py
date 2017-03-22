# encoding:utf-8
import cv2
import numpy as np
from matplotlib import pyplot as plt

def Picture_divide(img_src):
    img = cv2.imread(img_src)
    sp = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # 寻找霍夫直线
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, 10, 0)

    # for i in lines:
    #     for x1, y1, x2, y2 in i:
    #         cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 30)

    # 横线列表
    transverseline = []
    # 竖线列表
    verticalline = []

    for i in lines:
        for x1, y1, x2, y2 in i:
            if abs(y1 - y2) < 10:
                if abs(x1 - x2) > 15:
                    transverseline.append([x1, y1, x2, y2])
            if abs(x1 - x2) < 10:
                if abs(y1 - y2) > 10:
                    verticalline.append([x1, y1, x2, y2])

    # 横线通过纵坐标从小到大排序
    transverseline = sorted(transverseline, key=lambda keys: keys[1])
    # print transverseline
    # 竖线通过横坐标从小到大排序
    verticalline = sorted(verticalline, key=lambda keys: keys[0])
    # print verticalline

    # transverseline_assumption = [1455, 2755, 4015]
    # verticalline_assumption = [845, 2700]
    transverseline_assumption = [300, 588, 840]
    verticalline_assumption = [168, 564]
    # cv2.line(img, (845, 1455), (2700,1455), (0, 255, 0), 10)
    # cv2.line(img, (845, 2755), (2700, 2755), (0, 255, 0), 10)
    # cv2.line(img, (845, 4015), (2700, 4015), (0, 255, 0), 10)
    # cv2.line(img, (845, 1455), (845, 4015), (0, 255, 0), 10)
    # cv2.line(img, (2700, 1455), (2700, 4015), (0, 255, 0), 10)

    try:
        Feature_Measure(transverseline_assumption, verticalline_assumption, img)
    except Exception, e:
        print e
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.show()

def Feature_Measure(transverseline, verticalline, img_all):
    for i in range(2):
        img = img_all.copy()
        if i == 0:
            crop_img = img[int(transverseline[0]):int(transverseline[1]), int(verticalline[0]):int(verticalline[1])]
            Feature_function(crop_img)
        else:
            crop_img = img[int(transverseline[1]):int(transverseline[2]), int(verticalline[0]):int(verticalline[1])]
            Feature_function(crop_img)

def Feature_function(img1):
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    # cv2.imshow("123", img1)
    # cv2.waitKey(0)
    # img1 = cv2.imread('image2.jpg', 0)  # queryImage
    img2 = cv2.imread('../1712.jpg', 0)  # trainImage
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SURF_create()
    MIN_MATCH_COUNT = 50
    # cv2.xfeatures2d.SURF_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # BFMatcher with default params
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(np.asarray(des1, np.float32), np.asarray(des2, np.float32), k=2)
    # Apply ratio test
    # 比值测试，首先获取与A 距离最近的点B（最近）和C（次近），只有当B/C
    # 小于阈值时（0.75）才被认为是匹配，因为假设匹配是一一对应的，真正的匹配的理想距离为0
    good = []
    for m, n in matches:
        if m.distance < 0.73 * n.distance:
            good.append(m)
    print len(good)
    if len(good) > MIN_MATCH_COUNT:
        # 获取关键点的坐标
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        # print src_pts
        # print dst_pts
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
    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
    # cv2.imshow("test", img3)
    # cv2.waitKey(0)
    plt.imshow(img3, 'gray'), plt.show()

if __name__ == "__main__":
    img_src = "../1711.jpg"
    Picture_divide(img_src)