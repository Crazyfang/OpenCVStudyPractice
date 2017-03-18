# -*- coding: utf-8 -*-
"""
图像识别物品摆放架块域识别
实现功能：将物品摆放架的放置物品区域识别出来并计算出点坐标
增加功能：将计算出来的坐标用来划分图像区域并分割来识别颜色块及特定物品
版本：1.20
作者：Crazyfang
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import threading
import Queue

Storage = Queue.Queue()
Image_list = ['../Picture/IMG_20170228_160847.jpg', '../Picture/IMG_20170301_143206.jpg', '../Picture/IMG_20170228_160931.jpg']
Judge_list = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
Template_list = ['1426.jpg', '1533.jpg', '1534.jpg']
Color_materials = []
lock = threading.Lock()

# 图像区域划分函数
def ImagePartition(original_image, number):
    start = time.clock()
    print "正在处理第%d张图" % number
    # img = cv2.imread('picture.jpg')
    img = cv2.imread(original_image)
    sp = img.shape
    # print sp[1]
    # print (2 * sp[0] / 3)
    # edges = cv2.Canny(img, 100, 150)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    # edge = edges.copy()
    # lines = cv2.HoughLines(edges, 1, np.pi/180, 100)

    # 寻找霍夫直线
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, 10, 0)
    # print lines

    # 横线列表
    transverseline = []
    # 竖线列表
    verticalline = []


    # for i in lines:
    #     for rho, theta in i:
    #         a = np.cos(theta)
    #         b = np.sin(theta)
    #         x0 = a*rho
    #         y0 = b*rho
    #         x1 = int(x0 + 1000*(-b))
    #         y1 = int(y0 + 1000*(a))
    #         x2 = int(x0 - 1000*(-b))
    #         y2 = int(y0 - 1000*(a))
    #         cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 将横线和竖线加入各自的列表同时去除不符合纵坐标的噪声点
    for i in lines:
        for x1, y1, x2, y2 in i:
            # if y1 < 500:
            if abs(y1 - y2) < 10:
                if abs(x1 - x2) > 15:
                    if (1 * sp[0] / 3) < y1 < (3 * sp[0] / 4):
                        if sp[1] / 14 < x1:
                            transverseline.append([x1, y1, x2, y2])
            if abs(x1 - x2) < 10:
                if abs(y1 - y2) > 15:
                    if y1 < (2 * sp[0] / 3):
                        verticalline.append([x1, y1, x2, y2])
            # cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # print lines
    print "竖线精简后的列表"
    print verticalline
    # cv2.line(img, (26, 314), (761, 316), (0, 255, 0), 2)
    # print "---------"
    # print transverseline
    # print "---------"
    # print verticalline
    # print "---------"

    sign = 0
    min = 0
    max = 0
    sum = 0
    length = 0

    # 横线通过纵坐标从小到大排序
    transverseline = sorted(transverseline, key=lambda keys: keys[1])
    # 竖线通过横坐标从小到大排序
    verticalline = sorted(verticalline, key=lambda keys: keys[0])
    # print verticalline

    print "排序后的横线列表"
    print transverseline
    print "横线列表的长度"
    print len(transverseline)

    # 基准横线下标列表
    numberlist = []
    # 最大纵坐标差值
    max_differentvalue = 0

    # 寻找三条基准横线改进版
    # 功能：在横线坐标列列表中寻找到三条基准横线的区间下标
    for count, i in enumerate(transverseline):
        if count == 0:
            adjust = i[1]
        else:
            if len(numberlist) == 3:
                break
            else:
                if abs(i[1] - adjust) > max_differentvalue * 0.8 and abs(i[1] - adjust) > 240:  # 高像素图片需要在此更改阈值 > 200   abs(i[1] - adjust) > max_differentvalue * 1.50 and
                    adjust = i[1]
                    numberlist.append(count)
                    max_differentvalue = 0
                if abs(i[1] - adjust) > max_differentvalue:
                    max_differentvalue = abs(i[1] - adjust)
                else:
                    pass
        # print max_differentvalue
    print numberlist
    # print transverseline[200], transverseline[201]
    # numberlist[2] = 250
    # print transverseline[59]
    # print transverseline[60]
    # print transverseline[86]

    # 寻找三条基准横线长度拟差运算
    # 基准横线的组成直线部分的纵坐标的值之和
    count_num = 0
    # 基准横线的最小横坐标
    min = transverseline[0][0]
    # 基准横线的最大横坐标
    max = transverseline[0][2]
    for count, i in enumerate(transverseline):
        if count == numberlist[0]:
            line1 = [min, count_num / count, max, count_num / count]
            count_num = 0
        elif count == numberlist[1]:
            line2 = [min, count_num / (numberlist[1] - numberlist[0]), max, count_num / (numberlist[1] - numberlist[0])]
            # print numberlist[1] - numberlist[0]
            count_num = 0
        elif count == numberlist[2]:
            line3 = [min, count_num / (numberlist[2] - numberlist[1]), max, count_num / (numberlist[2] - numberlist[1])]
            # print numberlist[2] - numberlist[1]
        count_num += i[1]
        if min > i[0]:
            min = i[0]
        if max < i[2]:
            max = i[2]

    # print transverseline[109]
    # print transverseline[108]

    # # 寻找三条基准横线
    # # print transverseline
    # for i in transverseline:
    #     if i[1] < 250:
    #         if min == 0 and max == 0:
    #             min = i[0]
    #             max = i[2]
    #         else:
    #             if min > i[0]:
    #                 min = i[0]
    #             if max < i[2]:
    #                 max = i[2]
    #         sum += i[1]
    #         length += 1
    #     if 250 < i[1] < 350:
    #         if sign == 0:
    #             line1 = [min, sum / length, max, sum / length]
    #             min = i[0]
    #             max = i[2]
    #             sum = 0
    #             length = 0
    #             sign = 1
    #         else:
    #             if min > i[0]:
    #                 min = i[0]
    #             if max < i[2]:
    #                 max = i[2]
    #             sum += i[1]
    #             length += 1
    #     if 350 < i[1] < 400:
    #         if sign == 1:
    #             line2 = [min, sum / length, max, sum / length]
    #             min = i[0]
    #             max = i[2]
    #             sum = 0
    #             length = 0
    #             sign = 0
    #         else:
    #             if min > i[0]:
    #                 min = i[0]
    #             if max < i[2]:
    #                 max = i[2]
    #             sum = i[3]
    # line3 = [min, sum, max, sum]

    print "三条基准横线坐标显示"
    print line1
    print line2
    print line3

    line2[1] += 20
    line2[3] += 20

    line3[1] += 80
    line3[3] += 80

    # 三条基准横线的差值弥补
    if line3[1] - line2[1] + 50 < line2[1] - line1[1]:
        line3[1] = line3[3] = 2 * line2[1] - line1[1] - 50

    # 寻找三条基准横线最小横坐标当做基准最小横坐标
    min = MinNumber(line1[0], line2[0], line3[0])
    # 寻找三条基准横线最大横坐标当做基准最大横坐标
    max = MaxNumber(line1[2], line2[2], line3[2])

    # 寻找基准竖线
    signs = []
    newline = []
    sign = 0
    sum = 0
    length = 1
    rectent = 0
    lens = 1
    for i in verticalline:
        if sign == 0:
            sum += i[0]
            sign = 1
            rectent = i[0]
            # 权重标记
            priority_tagged = 0
        else:
            lens += 1
            if abs(i[0] - rectent) < 10:
                if len(verticalline) == lens:
                    rectent = i[0]
                    sum += i[0]
                    length += 1
                    newline.append(sum / length)
                    signs.append(1)
                else:
                    rectent = i[0]
                    sum += i[0]
                    length += 1
                    priority_tagged = 1
            elif 10 < abs(i[0] - rectent) < 80:
                if len(verticalline) == lens:
                    sum += i[0]
                    length += 1
                    rectent = i[0]
                    newline.append(sum / length)
                    if(priority_tagged == 0):
                        signs.append(2)
                    elif priority_tagged == 1:
                        signs.append(3)
                else:
                    rectent = i[0]
                    sum += i[0]
                    length += 1
                    if priority_tagged == 0:
                        priority_tagged = 2
                    else:
                        priority_tagged = 3
            else:
                if len(verticalline) == lens:
                    # sum += i[0]
                    # length += 1
                    # rectent = i[0]
                    newline.append(sum / length)
                    newline.append(i[0])
                    signs.append(priority_tagged)
                    signs.append(0)
                    # priority_tagged = 0
                else:
                    rectent = i[0]
                    newline.append(sum / length)
                    signs.append(priority_tagged)
                    priority_tagged = 0
                    sum = i[0]
                    length = 1
    print "计算后的竖线列表"
    print newline
    print "权重列表"
    print signs
    # if line1[0] < line2[0]:
    #     min = line1[0]

    # 第二版本竖线识别新增部分
    #
    delsign =[]
    addnumber = []
    subscript = SearchMaxSum(signs)
    breadth = newline[subscript + 1] - newline[subscript]
    print sp[1] / 7 - 50
    print breadth
    while breadth < sp[1] / 7 - 50:
        if subscript != len(newline) - 1:
            if signs[subscript] >= signs[subscript + 1]:
                del newline[subscript + 1]
                del signs[subscript + 1]
            else:
                del newline[subscript]
                del signs[subscript]
            if subscript == len(newline) - 1:
                breadth = newline[subscript] - newline[subscript - 1]
            else:
                breadth = newline[subscript + 1] - newline[subscript]
        else:
            del newline[subscript - 1]
            del signs[subscript - 1]
            subscript -= 1
            breadth = newline[subscript] - newline[subscript - 1]
    while breadth > sp[1] / 7 + 50:
        if subscript != len(signs) - 1:
            if signs[subscript] < signs[subscript + 1]:
                subscript += 1
            else:
                pass
        else:
            pass
        if subscript != 0:
            for i in range(subscript):
                breadth = (newline[subscript] - newline[i]) / (subscript - i)
                if sp[1] / 7 - 50 <= breadth <= sp[1] / 7 + 50:
                    # for x in range(i + 1, subscript):
                    #     del newline[x]
                    #     del signs[x]
                    #     print "数据核查"
                    #     print newline
                    break
                else:
                    if i == subscript - 1 and subscript != len(newline) - 1:
                        for j in range(subscript + 1, len(newline)):
                            breadth = (newline[j] - newline[subscript]) / (j - subscript)
                            if sp[1] / 7 - 50 <= breadth <= sp[1] / 7 + 50:
                                break
                            else:
                                if j == len(newline) - 1:
                                    breadth = sp[1] / 7 + 50
                                    break
                                else:
                                    pass
                    else:
                        breadth = sp[1] / 7 + 50
        else:
            for i in range(subscript + 1, len(newline)):
                breadth = (newline[i] - newline[subscript]) / (i - subscript)
                if sp[1] / 7 - 50 <= breadth <= sp[1] / 7 + 50:
                    break
                else:
                    if i == len(newline) - 1:
                        breadth = sp[1] / 7 + 50
                        break
                    else:
                        pass
    print "区间宽度："
    print breadth
    print "区间下标"
    print subscript
    for i in range(subscript):
        if newline[subscript] - (subscript - i) * breadth > 0:
            newline[i] = newline[subscript] - (subscript - i) * breadth
        else:
            delsign.append(i)
    for i in range(subscript + 1, len(newline)):
        # 第三版本竖线识别修改部分
        # print float((newline[i] - newline[subscript])) / breadth
        # print (i - subscript) + 0.8
        if float((newline[i] - newline[subscript])) / breadth < (i - subscript) + 1.0:
            # 该判断是为了防止新添竖线超过图像最大横坐标
            if newline[subscript] + (i - subscript) * breadth > sp[1]:
                # newline[i] = sp[1] - 5
                delsign.append(i)
            else:
                newline[i] = newline[subscript] + (i - subscript) * breadth
        else:
            if not addnumber:
                adds = ((newline[i] - newline[subscript]) / breadth)
                addnumber.append(i)
            else:
                pass
            newline[i] = newline[subscript] + adds * breadth
            if newline[i] > sp[1]:
                delsign.append(i)
                # del signs[i]
            # if float((newline[i] - newline[subscript])) / breadth < (i - subscript) + 1.0:
            #     adds = ((newline[i] - newline[subscript]) / breadth) + 1
            #     addnumber.append(i)
            #     newline[i] = newline[subscript] + adds * breadth
            #     if newline[i] > sp[1]:
            #         del newline[i]
            #         del signs[i]
            # else:
            #     xx = float((newline[i] - newline[subscript])) / breadth - (i - subscript)
            #     yy = int(xx)
            #     zz = xx - yy
            #     if zz > 0.6:
            #         n = yy
            #     else:
            #         n = yy - 1
            #     for x in range(n):
            #         addnumber.append(i + x * 2)
            #     if zz > 0.6:
            #         adds = ((newline[i] - newline[subscript]) / breadth) + yy + 1
            #         n = yy
            #     else:
            #         adds = ((newline[i] - newline[subscript]) / breadth) + yy
            #     newline[i] = newline[subscript] + adds * breadth
    print "区间下标"
    print subscript
    print "修改宽度后的列表："
    print newline
    print "待删除列表："
    print delsign
    print "待增添列表："
    print addnumber
    if delsign:
        print "列表长度为:%d" % len(newline)
        adjusts = subscript
        for i in range(len(delsign)):
            if delsign[i] < adjusts:
                subscript -= 1
            delsign[i] -= i
        for x in delsign:
            del newline[x]
    print "区间下标"
    print subscript
    print "删除之后的列表："
    print newline



    # 第三部分竖线识别新增部分
    # 功能：添加缺失线
    if addnumber:
        for i in addnumber:
            if newline[subscript] + (i - subscript) * breadth < sp[1]:
                newline.insert(i, newline[subscript] + (i - subscript) * breadth)
    print "增添后的列表："
    print newline
    if len(newline) == 7:
        if newline[len(newline) - 1] - newline[len(newline) - 2] < breadth:
            del newline[len(newline) - 1]
        if newline[0] > breadth:
            newline.insert(0, newline[0] - breadth)
        if newline[0] < sp[1] / 14 -70:
            lens = sp[1] / 14 - newline[0]
            for i in range(0, len(newline)):
                newline[i] += (lens - i * 50)
    elif len(newline) < 7:
        for i in range(newline[0] / breadth):
            # if newline[0] > breadth:
            newline.insert(0, abs(newline[i] - breadth * (i + 1)))
        if newline[0] < sp[1] / 14 - 70:         # 180
            lens = sp[1] / 14 - newline[0]       # 210
            for i in range(len(newline)):
                newline[i] += (lens - i * 50)
        if len(newline) < 7:
            if sp[1] / 14 < newline[0] < sp[1] / 7:
                print newline[0]
                d_value = newline[0] - (newline[0] + sp[1] / 14) / 2
                for i in range(len(newline)):
                    newline[i] -= (i + 1) * d_value - i * 50
                # newline_length = len(newline)
                newline.append(newline[len(newline) - 1] + breadth)
    else:
        if newline[len(newline) - 1] - newline[len(newline) - 2] < breadth:
            del newline[len(newline) - 1]
        else:
            if newline[len(newline) - 1] - sp[1] < sp[1] / 14:
                del newline[len(newline) - 1]
            else:
                pass

    # if newline[0] < 180:
    #     lens = 210 - newline[0]
    #     for i in range(len(newline)):
    #         newline[i] += (lens)
    print "最终列表："
    print newline
    # line1 = map(int, line1)
    # b = map(int, b)
    # c = map(int, c)
    # d = map(int, d)
    # newline[5] = 1120
    try:
        if number == 1:
            threadings = FindColorFools(line1[1], line2[1], line3[1], newline, img)
            # threadings = FeatureMatching(line1[1], line2[1], line3[1], newline, img)
        else:
            threadings = FeatureMatching(line1[1], line2[1], line3[1], newline, img, number)
        for t in threadings:
            # t.setDaemon(True)
            t.start()
        for t in threadings:
            t.join()
    except Exception as e:
        print e
    Color_materials.sort()
    print Color_materials
    # ThredFeatureMatching(img, 1)
    cv2.line(img, (min, line1[1]), (max, line1[3]), (0, 255, 0), 20)
    cv2.line(img, (min, line2[1]), (max, line2[3]), (0, 255, 0), 20)
    cv2.line(img, (min, line3[1]), (max, line3[3]), (0, 255, 0), 20)
    for i in newline:
        cv2.line(img, (i, line1[1]), (i, line3[1]), (0, 255, 0), 20)
    # for i in verticalline:
    #     cv2.line(img, (i[0], i[1]), (i[2], i[3]), (0, 255, 0), 10)
        # for x1, y1, x2, y2 in i:
        #     cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # cv2.line(img, (newline[0], line1[1]), (newline[0], line3[1]), (0, 255, 0), 2)
    # cv2.line(img, (newline[1], line1[1]), (newline[1], line3[1]), (0, 255, 0), 2)
    # cv2.line(img, (newline[2], line1[1]), (newline[2], line3[1]), (0, 255, 0), 2)
    # cv2.line(img, (newline[3], line1[1]), (newline[3], line3[1]), (0, 255, 0), 2)

    # FindColorFools(line2[1], newline[1], 0, line1[1])

    # image, contours, hierarchy = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # imgs = cv2.drawContours(image, contours, 3, (0, 255, 0), 3)
    end = time.clock()
    print('Running time: %s Seconds' % (end - start))
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122), plt.imshow(edges, cmap='gray')
    # plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122), plt.imshow(imgs, cmap='gray')
    # plt.title('Third Image'), plt.xticks([]), plt.yticks([])
    plt.show()
    # cv2.waitKey(0)
    # cv2.imshow("123", img)

# 寻找三个数字中最大的数字
def MinNumber(a, b, c):
    min = a
    if a > b:
        min = b
        if b > c:
            min = c
    else:
        pass
    return min

# 寻找三个数字中最大的数字
def MaxNumber(a, b, c):
    max = a
    if a < b:
        max = b
        if b < c:
            max = c
    else:
        pass
    return max

# 寻找权重相加最大的相邻两个数的第一个数的下标
def SearchMaxSum(a):
    for count, i in enumerate(a):
        if count == 0:
            maxsum = i
        elif count == 1:
            maxsum += i
            sign = count -1
        else:
            if maxsum == 6:
                return sign
            else:
                if maxsum < i + a[count - 1]:
                    maxsum = i + a[count - 1]
                    sign = count - 1
                else:
                    pass
    return sign

# 颜色识别模块
def FindColorFools(line1, line2, line3, d, imgs):
    """
    :param line1: 第一根基准横线的值
    :param line2: 第二根基准横线的值
    :param line3: 第三根基准横线的值
    :param d:基准竖线列表
    :param imgs:图像文件
    :return:显示按既定排序显示的图像
    """
    thread = []
    length = len(d)
    hsvs = cv2.cvtColor(imgs, cv2.COLOR_BGR2HSV)
    # a = map(int, a)
    # b = map(int, b)
    # c = map(int, c)
    # d = map(int, d)
    # line1 = a[1]
    # line2 = b[1]
    # line3 = c[1]
    # print type(x)
    # print type(y)
    # print type(z)
    for subscript, i in enumerate(d):
        if subscript == length - 1:
            break
        else:
            for m in range(2):
                img = imgs.copy()
                # img = cv2.imread("1234.jpg")
                sub = subscript
                if m == 0:
                    # print type(a[1])
                    # print type(b[1])
                    # print type(i)
                    # print type(d[subscript + 1])
                    crop_img = img[line1:line2, i:d[subscript + 1]]
                    hsv = hsvs[line1:line2, i:d[subscript + 1]]
                    sub += 1
                else:
                    crop_img = img[line2:line3, i:d[subscript + 1]]
                    hsv = hsvs[line2:line3, i:d[subscript + 1]]
                    sub += length
                threads = threading.Thread(target=ThredFindColorFools, args=(crop_img, hsv, sub))
                thread.append(threads)
                # img = cv2.imread("1234.png")
                # crop_img = img[187:302, 1:229]
                # hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
                # cimg = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                # 设定黄色的阈值
                # lower_blue=np.array([110,50,50])
                # upper_blue=np.array([130,255,255])
                # lower_blue = np.array([26, 43, 46])
                # upper_blue = np.array([34, 255, 255])
                # lower_yellow = np.array([0, 130, 50])
                # upper_yellow = np.array([34, 255, 255])
                # lower_blue = np.array([30, 4, 219])
                # upper_blue = np.array([51, 150, 255])

                # lower_blue = np.array([20, 241, 129])
                # upper_blue = np.array([25, 255, 157])
                # 根据阈值构建掩模
                # mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
                # res = cv2.bitwise_and(crop_img, crop_img, mask=mask)
                # img, contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

                # if contours:
                #     a = contours[0]
                #     x, y, w, h = cv2.boundingRect(a)
                #
                #     for cnt in contours:
                #         # contour_area_temp = np.fabs(cv2.contourArea(cnt))
                #         x, y, w, h = cv2.boundingRect(cnt)
                #         # print x, y, w, h
                #         if w + h > 50:
                #             cv2.rectangle(crop_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                #             cv2.imshow('%d' % sub, crop_img)
    return thread

# 颜色识别多线程
def ThredFindColorFools(crop_img, hsv, sub):
    lower_yellow = np.array([0, 130, 50])
    upper_yellow = np.array([34, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    img, contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if contours:
        a = contours[0]
        x, y, w, h = cv2.boundingRect(a)

        for cnt in contours:
            # contour_area_temp = np.fabs(cv2.contourArea(cnt))
            x, y, w, h = cv2.boundingRect(cnt)
            # print x, y, w, h
            if w + h > 150:
                # cv2.rectangle(crop_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # cv2.imshow('%d' % sub, crop_img)
                Color_materials.append([sub, "yello materials"])
                # print sub
                # cv2.waitKey(0)
                # print "--------"
                break

# 特征识别模块
def FindFeature(line1, line2, line3, d, imgs):
    length = len(d)
    img2 = cv2.imread("IMG_3781.JPG")
    for subscript, i in enumerate(d):
        if subscript == length - 1:
            break
        else:
            for m in range(2):
                img = imgs.copy()
                # img = cv2.imread("1234.jpg")
                sub = subscript
                if m == 0:
                    # print type(a[1])
                    # print type(b[1])
                    # print type(i)
                    # print type(d[subscript + 1])
                    img1 = img[line1:line2, i:d[subscript + 1]]
                    sub += 1
                else:
                    img1 = img[line2:line3, i:d[subscript + 1]]
                    sub += length
                sift = cv2.xfeatures2d.SURF_create()
                MIN_MATCH_COUNT = 20
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
                print good
                if len(good) > MIN_MATCH_COUNT:
                    # 获取关键点的坐标
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
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
                    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                                       singlePointColor=None,
                                       matchesMask=matchesMask,  # draw only inliers
                                       flags=2)
                    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
                    cv2.imshow("%d" % sub, img3)
                else:
                    print "Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT)
                    matchesMask = None
                # # cv2.drawMatchesKnn expects list of lists as matches.
                # img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good[:10],flags=2)

    cv2.waitKey(0)

# 特征匹配模块多线程
def FeatureMatching(line1, line2, line3, d, imgs, number):
    thread = []
    length = len(d)
    for subscript, i in enumerate(d):
        if subscript == length - 1:
            break
        else:
            for m in range(2):
                img = imgs.copy()
                sub = subscript
                if m == 0:
                    crop_img = img[line1:line2, i:d[subscript + 1]]
                    sub += 1
                else:
                    crop_img = img[line2:line3, i:d[subscript + 1]]
                    sub += length
                threads = threading.Thread(target=ThredFeatureMatching, args=(crop_img, sub, number))
                thread.append(threads)
    return thread

# 特征匹配模块多线程
def ThredFeatureMatching(img, sub, number):
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.waitKey(0)
    for i, filename in enumerate(Template_list):
        template = cv2.imread(filename, 0)
        w, h = template.shape[::-1]
        res = cv2.matchTemplate(img2, template, cv2.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if max_val > Judge_list[i][0]:
            lock.acquire()
            Judge_list[i][0] = max_val
            Judge_list[i][1] = number
            Judge_list[i][2] = sub
            lock.release()
        # top_left = min_loc
        # # print max_val
        # # print min_val
        # if max_val > 0.95:
        #     print "%d-------\n" % sub
        #     print min_val
        #     print "\n"
        #     print max_val

# 总处理函数
def MainDispose(img1, img2, img3, img4):
    threads = []
    items = []
    # Thread1 = threading.Thread(target=ImagePartition, args=(img1, ))
    # Thread1.start()
    Thread2 = threading.Thread(target=ImagePartition, args=(img2, 2))
    Thread3 = threading.Thread(target=ImagePartition, args=(img3, 3))
    Thread4 = threading.Thread(target=ImagePartition, args=(img4, 4))
    #
    threads.append(Thread2)
    threads.append(Thread3)
    threads.append(Thread4)

    for i in threads:
        i.start()
    #
    for i in threads:
        i.join()
    # Thread1.join()
    # while not Storage.empty():
    #     item = Storage.get()
    #     if item[1] == 1:
    #         FindColorFools(item[0][0], item[0][1], item[0][2], item[0][3], img1)
    #     else:
    #         items.append(item)
    #
    # for t in threads:
    #     t.join()
    #
    # while not Storage.empty():
    #     item = Storage.get()
    #     items.append(item)
    # 写到此，积攒思路，留待下次解决

# 程序入口
if __name__ == '__main__':
    # ImagePartition(1)
    # MainDispose('../Picture/IMG_20170301_143213.jpg', Image_list[0], Image_list[1], Image_list[2]) IMG_20170301_143206.jpg
    ImagePartition("../Picture/IMG_20170301_143210.jpg", 1)

    # Color_materials = sorted(Color_materials, key=lambda color_materials:color_materials[0])
    # print Color_materials
    # ImagePartition(Image_list[0], 1)
    # print Judge_list