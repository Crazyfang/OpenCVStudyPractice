# encoding:utf-8
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

def Picture_divide(img_src):
    start = time.clock()
    img = cv2.imread(img_src)
    sp = img.shape
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
    print verticalline_left
    print verticalline_right

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
    print verticalline_assumption

    if abs(verticalline_assumption[1] - verticalline_assumption[0]) < sp[1] / 3:
        verticalline_assumption[0] = sp[1] / 4
        verticalline_assumption[1] = sp[1] * 3 / 4
    else:
        pass

    print verticalline_assumption

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
    print transverseline_up
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
    print abscissa

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

    end = time.clock()
    print('Running time: %s Seconds' % (end - start))
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.show()

def Picture_divides(img_src):
    start = time.clock()
    img = cv2.imread(img_src)
    sp = img.shape
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

    # for i in lines:
    #     for x1, y1, x2, y2 in i:
    #         if sp[1] / 3 < x1 < 2 * sp[1] / 3:
    #             if abs(y1 - y2) < 10:
    #                 if abs(x1 - x2) > 5:
    #                     transverseline.append([x1, y1, x2, y2])
    #         if abs(x1 - x2) < 10:
    #             verticalline.append([x1, y1, x2, y2])

    # # 横线通过纵坐标从小到大排序
    # transverseline = sorted(transverseline, key=lambda keys: keys[1])
    # # print transverseline
    # # 竖线通过横坐标从小到大排序
    # verticalline = sorted(verticalline, key=lambda keys: keys[0])
    # # print verticalline
    # print len(transverseline)
    # print len(verticalline)
    # print transverseline
    # for i in verticalline:
    #     cv2.line(img, (i[0], i[1]), (i[2], i[3]), (0, 255, 0), 10)
    # for i in transverseline:
    #     cv2.line(img, (i[0], i[1]), (i[2], i[3]), (0, 255, 0), 10)
        # print i
        # for x1, y1, x2, y2 in i:
        #     cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # print verticalline

    # transverseline_assumption = [1455, 2755, 4015]
    # verticalline_assumption = [845, 2700]
    # transverseline_assumption = [370, 530, 800]
    # verticalline_assumption = [168, 564]
    # transverseline_assumption = [200, 370, 545]
    # verticalline_assumption = [168, 461]
    transverseline_assumption = [900, 1700, 1800, 2800]
    verticalline_assumption = [500, 1900]
    # transverseline_assumption = [1300, 2450, 2800, 3900]
    # verticalline_assumption = [1100, 2500]
    # cv2.line(img, (845, 1455), (2700,1455), (0, 255, 0), 10)
    # cv2.line(img, (845, 2755), (2700, 2755), (0, 255, 0), 10)
    # cv2.line(img, (845, 4015), (2700, 4015), (0, 255, 0), 10)
    # cv2.line(img, (845, 1455), (845, 4015), (0, 255, 0), 10)
    # cv2.line(img, (2700, 1455), (2700, 4015), (0, 255, 0), 10)

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
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.show()

if __name__ == "__main__":
    image_path = "./2017324/IMG_20170324_145913.JPG"
    Picture_divide(image_path)