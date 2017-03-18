# -*- coding: utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

def Main():
    start = time.clock()
    img = cv2.imread('1234.jpg')    # 图片切换
    sp = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # 寻找霍夫直线
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, 10, 0)

    # 横线列表
    transverseline = []
    # 竖线列表
    verticalline = []

    # 将横线和竖线加入各自的列表同时去除不符合纵坐标的噪声点
    for i in lines:
        for x1, y1, x2, y2 in i:
            # if y1 < 500:
            if abs(y1 - y2) < 10:
                transverseline.append([x1, y1, x2, y2])
            if abs(x1 - x2) < 10:
                if abs(y1 - y2) > 10:
                    if y1 < (2 * sp[0] / 3):
                        verticalline.append([x1, y1, x2, y2])

    print "竖线精简后的列表"
    print verticalline

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
                if abs(i[1] - adjust) > max_differentvalue * 1.5 and abs(i[1] - adjust) > 50:
                    adjust = i[1]
                    numberlist.append(count)
                    max_differentvalue = 0
                if abs(i[1] - adjust) > max_differentvalue:
                    max_differentvalue = abs(i[1] - adjust)
                else:
                    pass

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

    print "三条基准横线坐标显示"
    print line1
    print line2
    print line3

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

    # 第二版本竖线识别新增部分
    #
    addnumber = []
    subscript = SearchMaxSum(signs)
    breadth = newline[subscript + 1] - newline[subscript]
    for i in range(subscript):
        newline[i] = abs(newline[subscript] - (subscript - i) * breadth)
    for i in range(subscript + 2, len(newline)):
        # 第三版本竖线识别修改部分
        if float((newline[i] - newline[subscript + 1])) / breadth < (i - subscript - 1) + 0.5:
            # 该判断是为了防止新添竖线超过图像最大横坐标
            if newline[subscript] + (i - subscript) * breadth > sp[1]:
                newline[i] = sp[1] - 5
            else:
                newline[i] = newline[subscript] + (i - subscript) * breadth
        else:
            adds = ((newline[i] - newline[subscript + 1]) / breadth) + 1
            addnumber.append(i)
            newline[i] = newline[subscript + 1] + adds * breadth
    print newline

    # 第三部分竖线识别新增部分
    # 功能：添加缺失线
    if addnumber:
        for i in addnumber:
            newline.insert(i, newline[subscript + 1] + (i - subscript - 1) * breadth)

    Color_Recognition(line1[1], line2[1], line3[1], newline, img)
    cv2.line(img, (min, line1[1]), (max, line1[3]), (0, 255, 0), 2)
    cv2.line(img, (min, line2[1]), (max, line2[3]), (0, 255, 0), 2)
    cv2.line(img, (min, line3[1]), (max, line3[3]), (0, 255, 0), 2)
    for i in newline:
        cv2.line(img, (i, line1[1]), (i, line3[1]), (0, 255, 0), 2)

    end = time.clock()
    print('耗费时间: %s 秒' % (end - start))
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Old Image'), plt.xticks([]), plt.yticks([])

    plt.show()

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

def Color_Recognition(line1, line2, line3, d, imgs):
    length = len(d)
    print "长度"
    print length
    for subscript, i in enumerate(d):
        if subscript == length - 1:
            break
        else:
            for m in range(2):
                img = imgs.copy()
                sub = subscript
                if m == 0:
                    frame = img[line1:line2, i:d[subscript + 1]]
                    sub += 1
                else:
                    frame = img[line2:line3, i:d[subscript + 1]]
                    sub += length
                #设置黄色色域
                # lower = np.array([18, 43, 46])
                # upper = np.array([34, 255, 255])
                lower = np.array([0, 130, 50])
                upper = np.array([34, 255, 255])

                #设定红色阈值
                # lower = np.array([170, 100, 100])
                # upper = np.array([179, 255, 255])

                #设定蓝色阈值
                # lower = np.array([100, 43, 46])
                # upper = np.array([124, 255, 255])

                #转到HSV空间
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                #根据阈值构建掩膜
                mask = cv2.inRange(hsv, lower, upper)
                #腐蚀操作
                mask = cv2.erode(mask, None, iterations=2)
                #膨胀操作，其实先腐蚀再膨胀的效果是开运算，去除噪点
                mask = cv2.dilate(mask, None, iterations=2)

                #获取黄色范围最大外接矩形四角坐标
                x, y, w, h = cv2.boundingRect(mask)
                if w + h > 30:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.imshow('Location %d' % sub, frame)
                # else:
                #     print "No"

def Image_Recognition(frame):
    length = len(d)
    img2 = cv2.imread("IMG_3781.JPG")
    for subscript, i in enumerate(d):
        if subscript == length - 1:
            break
        else:
            for m in range(2):
                img = imgs.copy()
                sub = subscript
                if m == 0:
                    img2 = img[line1:line2, i:d[subscript + 1]]
                    sub += 1
                else:
                    img2 = img[line2:line3, i:d[subscript + 1]]
                    sub += length
                MIN_MATCH_COUNT = 10

                # Initiate SIFT detector
                sift = cv2.xfeatures2d.SIFT_create()

                img1 = cv2.imread('img.png', 0)      # queryImage

                kp1, des1 = sift.detectAndCompute(img1, None)
                kp2, des2 = sift.detectAndCompute(img2, None)

                FLANN_INDEX_KDTREE = 0
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)

                flann = cv2.FlannBasedMatcher(index_params, search_params)

                matches = flann.knnMatch(des1, des2, k=2)

                # store all the good matches as per Lowe's ratio test.
                good = []
                for m, n in matches:
                    if m.distance < 0.7*n.distance:
                        good.append(m)
                if len(good) >= MIN_MATCH_COUNT:
                    print "Image matched,matches number - %d" % len(good)
                else:
                    print "Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT)

if __name__ == '__main__':
    #打开摄像头
    # cap = cv2.VideoCapture(0)
    #从摄像头获取一帧
    # ret, frame = cap.read()
    
    #判断是否成功打开摄像头  
    # if ret:
    #     Image_Recognition(frame)
    # else:
    #     print 'No Camera'
    #
    # #释放摄像头
    # cap.release()
    Main()