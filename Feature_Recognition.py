# encoding:utf-8
import cv2
import numpy as np
from matplotlib import pyplot as plt
import threading
import time
import logging

Template_Image = ["../IMG_3859.png", "../IMG_3855.jpg", "../IMG_3860.jpg"]      # 模板图片路径
Template_Image_name = ["zhonghuapencil", "yangleduo", "sprite"]
Image_up_data = []                                                              # 图片上的数据存储列表
Image_down_data = []                                                            # 图片下的数据存储列表
lock = threading.Lock()                                                         # 资源锁
cap = cv2.VideoCapture(0)

# 切图加运行函数
def Picture_divide(img_src):
    start = time.clock()
    img = cv2.imread(img_src)
    sp = img.shape
    print "图片像素:"
    print sp[0]
    print sp[1]
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
            if sp[1] / 3 < x1 < 2 * sp[1] / 3:
                if abs(y1 - y2) < 10:
                    if abs(x1 - x2) > 5:
                        transverseline.append([x1, y1, x2, y2])
            if abs(x1 - x2) < 10:
                if abs(y1 - y2) > 5:
                    verticalline.append([x1, y1, x2, y2])

    # 横线通过纵坐标从小到大排序
    transverseline = sorted(transverseline, key=lambda keys: keys[1])
    # # print transverseline
    # 竖线通过横坐标从小到大排序
    verticalline = sorted(verticalline, key=lambda keys: keys[0])
    # # print verticalline
    # print len(transverseline)
    # print len(verticalline)
    # print transverseline
    verticalline_left = []
    verticalline_right = []
    verticalline_left_sign = []                             # 竖线左的纵坐标
    verticalline_right_sign = []                            # 竖线右的纵坐标
    total_length = 0
    abscissa = 0
    abscissa_list = 0                                            # 横坐标收集
    sum_amount = 0
    last_number = 0
    verticalline_assumption = []
    for i in verticalline:
        if sp[1] / 6 < i[0] < sp[1] / 3:
            verticalline_left.append(i)
        elif sp[1] * 2 / 3 < i[0] < sp[1] * 5 / 6:
            verticalline_right.append(i)
        else:
            pass
    # print verticalline_left
    # print verticalline_right

    for i, j in enumerate(verticalline_left):
        sum_amount += 1
        if i == 0:
            last_number = j[0]
            abscissa_list += j[0]
            verticalline_left_sign.append(j[1])
            verticalline_left_sign.append(j[3])
            total_length = abs(j[3] - j[1])
        elif i != len(verticalline_left) - 1:
            if j[0] - last_number <= 50:
                abscissa_list += j[0]
                last_number = j[0]
                if verticalline_left_sign[0] > j[1]:
                    verticalline_left_sign[0] = j[1]
                elif verticalline_left_sign[1] < j[3]:
                    verticalline_left_sign[1] = j[3]
                else:
                    pass
            else:
                if total_length < abs(verticalline_left_sign[1] - verticalline_left_sign[0]):
                    total_length = abs(verticalline_left_sign[1] - verticalline_left_sign[0])
                    abscissa = abscissa_list / sum_amount
                    sum_amount = 1
                    abscissa_list = j[0]
                    last_number = j[0]
                else:
                    sum_amount = 1
                    abscissa_list = j[0]
                    last_number = j[0]
                verticalline_left_sign[0] = j[1]
                verticalline_left_sign[1] = j[3]
        else:
            if abscissa == 0:
                total_length = abs(verticalline_left_sign[1] - verticalline_left_sign[0])
                abscissa = (abscissa_list + j[0]) / sum_amount
                sum_amount = 0
                abscissa_list = 0
            elif abscissa != 0 and j[0] - last_number <= 50:
                if total_length < abs(verticalline_left_sign[1] - verticalline_left_sign[0]):
                    total_length = abs(verticalline_left_sign[1] - verticalline_left_sign[0])
                    abscissa = abscissa_list / sum_amount
                    sum_amount = 0
                    abscissa_list = 0
                    last_number = 0
                else:
                    pass

    # print "左边竖线:"
    # print abscissa
    verticalline_assumption.append(abscissa)

    abscissa = 0
    for i, j in enumerate(verticalline_right):
        sum_amount += 1
        if i == 0:
            last_number = j[0]
            abscissa_list += j[0]
            verticalline_right_sign.append(j[1])
            verticalline_right_sign.append(j[3])
            total_length = abs(j[3] - j[1])
        elif i != len(verticalline_right) - 1:
            if j[0] - last_number <= 50:
                abscissa_list += j[0]
                last_number = j[0]
                if verticalline_right_sign[0] > j[1]:
                    verticalline_right_sign[0] = j[1]
                elif verticalline_right_sign[1] < j[3]:
                    verticalline_right_sign[1] = j[3]
                else:
                    pass
            else:
                if total_length < abs(verticalline_right_sign[1] - verticalline_right_sign[0]):
                    total_length = abs(verticalline_right_sign[1] - verticalline_right_sign[0])
                    abscissa = abscissa_list / sum_amount
                    sum_amount = 1
                    abscissa_list = j[0]
                    last_number = j[0]
                else:
                    sum_amount = 1
                    abscissa_list = j[0]
                    last_number = j[0]
                verticalline_right_sign[0] = j[1]
                verticalline_right_sign[1] = j[3]
        elif i == len(verticalline_right) - 1:
            if abscissa == 0:
                total_length = abs(verticalline_right_sign[1] - verticalline_right_sign[0])
                abscissa = (abscissa_list + j[0]) / sum_amount
            elif abscissa != 0 and j[0] - last_number <= 50:
                if total_length < abs(verticalline_right_sign[1] - verticalline_right_sign[0]):
                    total_length = abs(verticalline_right_sign[1] - verticalline_right_sign[0])
                    abscissa = (abscissa_list + j[0]) / sum_amount
                    sum_amount = 0
                    abscissa_list = 0
                    last_number = 0
                else:
                    sum_amount = 0
                    abscissa_list = 0
                    last_number = 0
            else:
                pass
            # print sum_amount
            # print len(verticalline_right)
            # print "测试文字"
        else:
            pass
    # print "右边竖线:"
    # print abscissa
    verticalline_assumption.append(abscissa)
    # print verticalline_assumption

    if abs(verticalline_assumption[1] - verticalline_assumption[0]) < sp[1] / 3:
        verticalline_assumption[0] = sp[1] / 4
        verticalline_assumption[1] = sp[1] * 3 / 4
    else:
        pass

    # print verticalline_assumption

    transverseline_up = []
    # verticalline_right = []
    transverseline_up_sign = []  # 横线的纵坐标
    # verticalline_right_sign = []  # 竖线右的纵坐标
    for i in transverseline:
        if verticalline_assumption[0] < i[0] < verticalline_assumption[1]:
            if 300 < i[1] < sp[0] / 3:
                transverseline_up.append(i)

    abscissa = 0
    # print len(transverseline)
    # print transverseline_up
    for i, j in enumerate(transverseline_up):
        sum_amount += 1
        if i == 0:
            last_number = j[1]
            abscissa_list += j[1]
            transverseline_up_sign.append(j[0])
            transverseline_up_sign.append(j[2])
            total_length = abs(j[2] - j[0])
        elif i != len(transverseline_up) - 1:
            if j[1] - last_number <= 50:
                abscissa_list += j[1]
                last_number = j[1]
                if transverseline_up_sign[0] > j[0]:
                    transverseline_up_sign[0] = j[0]
                elif transverseline_up_sign[1] < j[2]:
                    transverseline_up_sign[1] = j[2]
                else:
                    pass
            else:
                if total_length < abs(transverseline_up_sign[1] - transverseline_up_sign[0]):
                    total_length = abs(transverseline_up_sign[1] - transverseline_up_sign[0])
                    abscissa = abscissa_list / sum_amount
                    sum_amount = 1
                    abscissa_list = j[1]
                    last_number = j[1]
                else:
                    sum_amount = 1
                    abscissa_list = j[1]
                    last_number = j[1]
                transverseline_up_sign[0] = j[0]
                transverseline_up_sign[1] = j[2]
        else:
            if abscissa == 0:
                total_length = abs(transverseline_up_sign[1] - transverseline_up_sign[0])
                abscissa = (abscissa_list + j[1]) / sum_amount
            elif abscissa != 0 and j[1] - last_number <= 50:
                if total_length < abs(transverseline_up_sign[1] - transverseline_up_sign[0]):
                    total_length = abs(transverseline_up_sign[1] - transverseline_up_sign[0])
                    abscissa = (abscissa_list + j[1]) / sum_amount
                else:
                    pass
            else:
                pass
    # print abscissa
    # print total_length
    if total_length < sp[1] / 4:
        abscissa = 0

    if abscissa < sp[0] / 4 or abscissa + 2 * sp[0] / 3 > sp[0]:
        abscissa = sp[0] / 4
    # print abscissa

    for i in range(3):
        cv2.line(img, (0, abscissa + i * sp[0] / 3), (sp[1], abscissa + i * sp[0] / 3), (0, 255, 0), 20)
    # for i in verticalline:
    #     cv2.line(img, (i[0], i[1]), (i[2], i[3]), (0, 255, 0), 10)
    # for i in transverseline:
    #     cv2.line(img, (i[0], i[1]), (i[2], i[3]), (0, 255, 0), 10)
        # print i
    for i in verticalline_assumption:
        cv2.line(img, (i, 0), (i, sp[0]), (0, 255, 0), 10)
        # for x1, y1, x2, y2 in i:
        #     cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # print verticalline
    # print transverseline

    # transverseline_assumption = [1455, 2755, 4015]
    # verticalline_assumption = [845, 2700]
    # transverseline_assumption = [370, 530, 800]
    # verticalline_assumption = [168, 564]
    # transverseline_assumption = [200, 370, 545]
    # verticalline_assumption = [168, 461]
    # transverseline_assumption = [900, 1700, 1800, 2800]
    # verticalline_assumption = [500, 1900]
    # transverseline_assumption = [1300, 2450, 2800, 3900]
    # verticalline_assumption = [1100, 2500]
    # cv2.line(img, (845, 1455), (2700,1455), (0, 255, 0), 10)
    # cv2.line(img, (845, 2755), (2700, 2755), (0, 255, 0), 10)
    # cv2.line(img, (845, 4015), (2700, 4015), (0, 255, 0), 10)
    # cv2.line(img, (845, 1455), (845, 4015), (0, 255, 0), 10)
    # cv2.line(img, (2700, 1455), (2700, 4015), (0, 255, 0), 10)
    transverseline_assumption = [abscissa, abscissa + 1 * sp[0] / 3, abscissa + 2 * sp[0] / 3]
    try:
        threads = Feature_Measure(transverseline_assumption, verticalline_assumption, img)
        for t in threads:
            # t.setDaemon(True)
            t.start()
        for t in threads:
            t.join()
    except Exception, e:
        print e

    end = time.clock()
    print('Running time: %s Seconds' % (end - start))
    print Image_up_data
    print Image_down_data
    # plt.subplot(121), plt.imshow(img, cmap='gray')
    # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    # plt.show()
    # plt.imshow(img, cmap='gray')
    # plt.axis('off')
    # plt.show()

# 图片特征匹配多线程准备函数
def Feature_Measure(transverseline, verticalline, img_all):
    threads = []
    for i in range(2):
        img = img_all.copy()
        if i == 0:
            crop_img = img[int(transverseline[0]):int(transverseline[1]), int(verticalline[0]):int(verticalline[1])]
            # Feature_function(crop_img)
            # ThredFeatureMatching(crop_img)
        else:
            crop_img = img[int(transverseline[1]):int(transverseline[2]), int(verticalline[0]):int(verticalline[1])]
            # Feature_function(crop_img)
            # ThredFeatureMatching(crop_img)
        thread = threading.Thread(target=Feature_Multithreading, args=(crop_img, i))
        threads.append(thread)
    return threads

# 图片特征匹配多线程
def Feature_Multithreading(img, sign):
    threads = []
    for order_number, img_path in enumerate(Template_Image):
        thread = threading.Thread(target=Feature_function, args=(img, img_path, sign, order_number))
        threads.append(thread)
    for i in threads:
        i.start()
    for i in threads:
        i.join()

# 图片特征匹配函数
def Feature_function(img1, img2, sign, order_number):
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    # cv2.imshow("123", img1)
    # cv2.waitKey(0)
    # img1 = cv2.imread('image2.jpg', 0)  # queryImage
    img2 = cv2.imread(img2, 0)  # trainImage
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    MIN_MATCH_COUNT = 80
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
        if m.distance < 0.70 * n.distance:
            good.append(m)
    print len(good)
    if len(good) >= MIN_MATCH_COUNT:
        # # 获取关键点的坐标
        # src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        # dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        # # print src_pts
        # # print dst_pts
        # # 第三个参数Method used to computed a homography matrix. The following methods are possible:
        # # 0 - a regular method using all the points
        # # CV_RANSAC - RANSAC-based robust method
        # # CV_LMEDS - Least-Median robust method
        # # 第四个参数取值范围在1 到10，􁲁绝一个点对的阈值。原图像的点经过变换后点与目标图像上对应点的误差
        # # 超过误差就认为是outlier
        # # 返回值中M 为变换矩阵。
        # M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        # matchesMask = mask.ravel().tolist()
        # # 获得原图像的高和宽
        # h, w = img1.shape
        # # 使用得到的变换矩阵对原图像的四个角进行变换，获得在目标图像上对应的坐标。
        # pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        # dst = cv2.perspectiveTransform(pts, M)
        # # 原图像为灰度图
        # cv2.polylines(img2, [np.int32(dst)], True, 255, 1, cv2.LINE_AA)
        if sign == 0:
            Image_up_data.append([sign + 1, order_number + 1, Template_Image_name[order_number], len(good)])
        else:
            Image_down_data.append([sign + 1, order_number + 1, Template_Image_name[order_number], len(good)])
        print "寻找到目标图像"
    else:
        if sign == 0:
            Image_up_data.append([sign + 1, order_number + 1, Template_Image_name[order_number], len(good)])
        else:
            Image_down_data.append([sign + 1, order_number + 1, Template_Image_name[order_number], len(good)])
        print "Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT)
        # matchesMask = None
    # draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
    #                    singlePointColor=None,
    #                    matchesMask=matchesMask,  # draw only inliers
    #                    flags=2)
    # img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
    # # cv2.imshow("test", img3)
    # # cv2.waitKey(0)
    # plt.imshow(img3, 'gray'), plt.show()

# 模板匹配函数
def ThredFeatureMatching(img):
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.waitKey(0)
    template = cv2.imread("../IMG_3859.png", 0)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(img2, template, cv2.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    print max_val
    plt.imshow(img2, 'gray'), plt.show()

# 开启摄像头拍照
def Camera_TakePhoto(str):
    if not cap.isOpened():
        cap.open()
    while(cap.isOpened()):
        ret, img = cap.read()
        if ret == True:
            # cv2.imshow('frame', img)
            # k = cv2.waitKey(10) & 0xff
            # if k == ord('q'):
            cv2.imwrite('%s.jpg' % str, img)
            break
    # cap.release()
    cv2.destroyWindow('frame')
    # cv2.imshow('catch', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # time.sleep(5)

if __name__ == "__main__":
    img_src = "./2017326/IMG_3856.jpg"
    content = raw_input("input:")
    while(content):
        Camera_TakePhoto(content)
        Picture_divide(img_src)
        content = raw_input("input:")
        if content == "0":
            break
    # Picture_divide(img_src)
    # print Image_up_data
    # print Image_down_data