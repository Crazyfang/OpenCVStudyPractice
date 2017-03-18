# encoding:utf-8
"""
作者：Crazyfang
日期：2017-3-17
版本：1.0
"""
import cv2
import numpy as np
# from matplotlib import pyplot as plt
import threading
import time

Color_materials = []

# 图像区域划分函数
def ImagePartition(original_image, number):
    start = time.clock()
    # print "正在处理第%d张图" % number
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
    # print "竖线精简后的列表"
    # print verticalline

    sign = 0
    mins = 0
    maxs = 0
    sum = 0
    length = 0

    # 横线通过纵坐标从小到大排序
    transverseline = sorted(transverseline, key=lambda keys: keys[1])
    # 竖线通过横坐标从小到大排序
    verticalline = sorted(verticalline, key=lambda keys: keys[0])

    # print "排序后的横线列表"
    # print transverseline
    # print "横线列表的长度"
    # print len(transverseline)

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
    # print numberlist

    # 寻找三条基准横线长度拟差运算
    # 基准横线的组成直线部分的纵坐标的值之和
    count_num = 0
    # 基准横线的最小横坐标
    mins = transverseline[0][0]
    # 基准横线的最大横坐标
    maxs = transverseline[0][2]
    for count, i in enumerate(transverseline):
        if count == numberlist[0]:
            line1 = [mins, count_num / count, maxs, count_num / count]
            count_num = 0
        elif count == numberlist[1]:
            line2 = [mins, count_num / (numberlist[1] - numberlist[0]), maxs, count_num / (numberlist[1] - numberlist[0])]
            count_num = 0
        elif count == numberlist[2]:
            line3 = [mins, count_num / (numberlist[2] - numberlist[1]), maxs, count_num / (numberlist[2] - numberlist[1])]
        count_num += i[1]
        if mins > i[0]:
            mins = i[0]
        if maxs < i[2]:
            maxs = i[2]

    # print "三条基准横线坐标显示"
    # print line1
    # print line2
    # print line3

    line2[1] += 20
    line2[3] += 20

    line3[1] += 80
    line3[3] += 80

    # 三条基准横线的差值弥补
    if line3[1] - line2[1] + 50 < line2[1] - line1[1]:
        line3[1] = line3[3] = 2 * line2[1] - line1[1] - 50

    # 寻找三条基准横线最小横坐标当做基准最小横坐标
    mins = MinNumber(line1[0], line2[0], line3[0])
    # 寻找三条基准横线最大横坐标当做基准最大横坐标
    maxs = MaxNumber(line1[2], line2[2], line3[2])

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
    # print "计算后的竖线列表"
    # print newline
    # print "权重列表"
    # print signs

    # 第二版本竖线识别新增部分
    delsign =[]
    addnumber = []
    subscript = SearchMaxSum(signs)
    breadth = newline[subscript + 1] - newline[subscript]
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
    # print "区间宽度："
    # print breadth
    # print "区间下标"
    # print subscript
    for i in range(subscript):
        if newline[subscript] - (subscript - i) * breadth > 0:
            newline[i] = newline[subscript] - (subscript - i) * breadth
        else:
            delsign.append(i)
    for i in range(subscript + 1, len(newline)):
        # 第三版本竖线识别修改部分
        if float((newline[i] - newline[subscript])) / breadth < (i - subscript) + 1.0:
            # 该判断是为了防止新添竖线超过图像最大横坐标
            if newline[subscript] + (i - subscript) * breadth > sp[1]:
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

    # print "区间下标"
    # print subscript
    # print "修改宽度后的列表："
    # print newline
    # print "待删除列表："
    # print delsign
    # print "待增添列表："
    # print addnumber
    if delsign:
        # print "列表长度为:%d" % len(newline)
        adjusts = subscript
        for i in range(len(delsign)):
            if delsign[i] < adjusts:
                subscript -= 1
            delsign[i] -= i
        for x in delsign:
            del newline[x]
    # print "区间下标"
    # print subscript
    # print "删除之后的列表："
    # print newline



    # 第三部分竖线识别新增部分
    # 功能：添加缺失线
    if addnumber:
        for i in addnumber:
            if newline[subscript] + (i - subscript) * breadth < sp[1]:
                newline.insert(i, newline[subscript] + (i - subscript) * breadth)
    # print "增添后的列表："
    # print newline
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

    # print "最终列表："
    # print newline

    try:
        if number == 1:
            threadings = FindColorFools(line1[1], line2[1], line3[1], newline, img)
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
    return Color_materials
    # print Color_materials
    #
    # cv2.line(img, (min, line1[1]), (max, line1[3]), (0, 255, 0), 20)
    # cv2.line(img, (min, line2[1]), (max, line2[3]), (0, 255, 0), 20)
    # cv2.line(img, (min, line3[1]), (max, line3[3]), (0, 255, 0), 20)
    # for i in newline:
    #     cv2.line(img, (i, line1[1]), (i, line3[1]), (0, 255, 0), 20)

    end = time.clock()
    # print('Running time: %s Seconds' % (end - start))
    # plt.subplot(121), plt.imshow(img, cmap='gray')
    # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    # plt.show()

# 寻找三个数字中最大的数字
def MinNumber(a, b, c):
    mins = a
    if a > b:
        mins = b
        if b > c:
            mins = c
    else:
        pass
    return mins

# 寻找三个数字中最大的数字
def MaxNumber(a, b, c):
    maxs = a
    if a < b:
        maxs = b
        if b < c:
            maxs = c
    else:
        pass
    return maxs

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
    :return:显示颜色识别多线程列表
    """
    thread = []
    hsvs = cv2.cvtColor(imgs, cv2.COLOR_BGR2HSV)

    for subscript, i in enumerate(d):
        if subscript == len(d) - 1:
            break
        else:
            for m in range(2):
                img = imgs.copy()
                sub = subscript
                if m == 0:
                    crop_img = img[line1:line2, i:int(d[subscript + 1])]
                    hsv = hsvs[line1:line2, i:int(d[subscript + 1])]
                    sub = sub * 2 + 1
                else:
                    crop_img = img[line2:line3, i:int(d[subscript + 1])]
                    hsv = hsvs[line2:line3, i:int(d[subscript + 1])]
                    sub = sub * 2 + 2
                threads = threading.Thread(target=ThredFindColorFools, args=(crop_img, hsv, sub))
                thread.append(threads)
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
            x, y, w, h = cv2.boundingRect(cnt)
            if w + h > 150:
                Color_materials.append([sub, "yellow materials"])
                break

# 测试函数
if __name__ == '__main__':
    ss = ImagePartition("IMG_1145.jpg", 1)
    print ss