# encoding:utf-8
# import cv2
# img = cv2.imread('1234.jpg',1)
# crop_img = img[1:520, 200:500]
# cv2.imshow("image", crop_img)
# cv2.imshow("images", img)
# cv2.waitKey(0)

# def FindColorFools(a, b, c, d, img):
#     length = len(d)
#     for subscript, i in enumerate(d):
#         for x in range(2):
#             if x == 0:
#                 crop_img = img[a[1]:b[1], i:d[i + 1]]
#                 subscript += 1
#             else:
#                 crop_img = img[b[1]:c[1], i:d[i + 1]]
#                 subscript += length
#             # img = cv2.imread("1234.png")
#             # crop_img = img[187:302, 1:229]
#             hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
#             cimg = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
#             # 设定蓝色的阈值
#             # lower_blue=np.array([110,50,50])
#             # upper_blue=np.array([130,255,255])
#             # lower_blue = np.array([26, 43, 46])
#             # upper_blue = np.array([34, 255, 255])
#             lower_blue = np.array([0, 130, 50])
#             upper_blue = np.array([34, 255, 255])
#             # lower_blue = np.array([30, 4, 219])
#             # upper_blue = np.array([51, 150, 255])
#
#             # lower_blue = np.array([20, 241, 129])
#             # upper_blue = np.array([25, 255, 157])
#             # 根据阈值构建掩模
#             mask = cv2.inRange(hsv, lower_blue, upper_blue)
#             res = cv2.bitwise_and(crop_img, crop_img, mask=mask)
#             img, contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
#
#             area_max_contour = 0
#             # if  contours:
#             #     a = contours[0]
#             #     x,y,w,h = cv2.boundingRect(a)
#             #
#             #     for cnt in contours:
#             #         contour_area_temp = np.fabs(cv2.contourArea(cnt))
#             #         x, y, w, h = cv2.boundingRect(cnt)
#             #         # print x,y,w,h
#             #         # 轮廓外接矩形宽长相差不超过指定像素(根据实际情况调整)
#             #         if np.fabs(w - h) < 80:
#             #             # 找寻满足条件的最大外接矩形面积的轮廓
#             #             if contour_area_temp > area_max_contour:
#             #                 area_max_contour = contour_area_temp
#             #                 a = cnt
#             #     # print a
#             #     # (x, y) 轮廓a的外接矩形左上角的点   w为宽 h为高
#             #     x,y,w, h = cv2.boundingRect(a)
#             # # print x,y,w,h
#             # # print '123455655'
#             # if w+h>100:
#             #     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#
#             if contours:
#                 a = contours[0]
#                 x, y, w, h = cv2.boundingRect(a)
#
#                 for cnt in contours:
#                     # contour_area_temp = np.fabs(cv2.contourArea(cnt))
#                     x, y, w, h = cv2.boundingRect(cnt)
#                     print x, y, w, h
#                     if w + h > 5:
#                         cv2.rectangle(crop_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                         cv2.imshow('%d' % subscript, crop_img)
#                         # print x,y,w,h
#                         # 轮廓外接矩形宽长相差不超过指定像素(根据实际情况调整)
#                         # if np.fabs(w - h) < 80:
#                         #     # 找寻满足条件的最大外接矩形面积的轮廓
#                         #     if contour_area_temp > area_max_contour:
#                         #         area_max_contour = contour_area_temp
#                         #         a = cnt
#                         # print a
#                         # (x, y) 轮廓a的外接矩形左上角的点   w为宽 h为高
#                         # x,y,w, h = cv2.boundingRect(a)
#             # print x,y,w,h
#             # print '123455655'
#
#
#
#             # cv2.HoughCircles(cimg, cv2.HOUGH_STANDARD, 1, 20, circles, param1=50, param2=30, minRadius=0, maxRadius=0)
#             # circles1 = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 1,
#             #                             100, param1=100, param2=30, minRadius=200, maxRadius=300)
#             # print type(circles1)
#             # if circles1 is not None:
#             #     circles = circles1[0, :, :]  # 提取为二维
#             #     circles = np.uint16(np.around(circles))  # 四舍五入，取整
#             #     for i in circles[:]:
#             #         # draw the outer circle
#             #         cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
#             #         # draw the center of the circle
#             #         cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)
#             #     # 对原图像和掩模进行位运算
#
#             # 显示图像
#             # cv2.imshow('frame', crop_img)
#             # cv2.imshow('mask', mask)
#             # cv2.imshow('res', res)
#             # crop_img = photo[a:b, c:d]
#
#     # print crop_img
#     # cv2.imshow("photo", crop_img)
#     cv2.waitKey(0)
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# import threading
# import Queue
# # img = cv2.imread('IMG_3779.JPG', 0)
# # img2 = img.copy()
# # template = cv2.imread('1403.jpg', 0)
# # w, h = template.shape[::-1]
# # print w, h
# # # All the 6 methods for comparison in a list
# # methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
# #             'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
# # # methods = ['cv2.TM_SQDIFF_NORMED']
# # for meth in methods:
# #     img = img2.copy()
# #     #exec 语句用来执行储存在字符串或文件中的Python 语句。
# #     # 例如，我们可以在运行时生成一个包含Python 代码的字符串，然后使用exec 语句执行这些语句。
# #     #eval 语句用来计算存储在字符串中的有效Python 表达式
# #     method = eval(meth)
# #     # Apply template Matching
# #     res = cv2.matchTemplate(img, template, method)
# #     # print res
# #     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
# #     # 使用不同的比较方法，对结果的解释不同
# #     # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
# #     if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
# #         top_left = min_loc
# #     else:
# #         top_left = max_loc
# #     print top_left
# #     # print max_loc
# #     print max_val
# #     print min_val
# #     # cv2.minMaxLoc()
# #     bottom_right = (top_left[0] + w, top_left[1] + h)
# #     cv2.rectangle(img, top_left, bottom_right, 255, 2)
# #     plt.subplot(121),plt.imshow(template, cmap = 'gray')
# #     plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
# #     plt.subplot(122),plt.imshow(img,cmap = 'gray')
# #     plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
# #     plt.suptitle(meth)
# #     plt.show()
# # q = Queue.Queue()
# # def Threads(sub):
# #     print "Thread-%d" % sub
# #     q.put(([1,2,3], sub))
# #
# # def ThreadMain():
# #     threads = []
# #     for i in range(5):
# #         threadings = threading.Thread(target=Threads, args=(i,))
# #         threads.append(threadings)
# #     return threads
# #
# # if __name__ == "__main__":
# #     threads = ThreadMain()
# #     for t in threads:
# #         t.start()
# #     while not q.empty():
# #         print q.get()[0][1]
#
# # str = raw_input()
# #
# # str = str.split(" ")
# #
# # print len(str[len(str) - 1])
# # str = raw_input()
# # st = raw_input()
# #
# # print str.count(st)
# # while 1:
# #     for i in range(10):
# #         if i == 0:
# #             break
# #     print 123
# # counts = raw_input()
# # lists = []
# # for i in range(int(counts)):
# #     str = raw_input()
# #     if str in lists:
# #         pass
# #     else:
# #         for x in range(len(lists))
# #         lists.append(str)
# # print lists
# #
# #
# # for i in range(len(lists)):
# #     print lists[i]
# # delsign = []
# # addnumber = []
# # sp = [300, 5200]
# # newline = [740, 1440, 2140, 2240, 3540, 4240]
# # subscript = 3
# # signs = [1, 0, 1, 3, 1, 1]
# # breadth = 100
# # for i in range(subscript):
# #     newline[i] = abs(newline[subscript] - (subscript - i) * breadth)
# # for i in range(subscript + 1, len(newline)):
# #     # 第三版本竖线识别修改部分
# #     if float((newline[i] - newline[subscript])) / breadth < (i - subscript) + 0.6:
# #         # 该判断是为了防止新添竖线超过图像最大横坐标
# #         if newline[subscript] + (i - subscript) * breadth > sp[1]:
# #             # newline[i] = sp[1] - 5
# #             delsign.append(newline[i])
# #         else:
# #             newline[i] = newline[subscript] + (i - subscript) * breadth
# #     else:
# #         if float((newline[i] - newline[subscript])) / breadth < (i - subscript) + 1.0:
# #             adds = ((newline[i] - newline[subscript]) / breadth) + 1
# #             addnumber.append(i)
# #             newline[i] = newline[subscript] + adds * breadth
# #             if newline[i] > sp[1]:
# #                 del newline[i]
# #                 del signs[i]
# #         else:
# #             xx = float((newline[i] - newline[subscript])) / breadth - (i - subscript)
# #             yy = int(xx)
# #             zz = xx - yy
# #             if zz > 0.6:
# #                 n = yy
# #             else:
# #                 n = yy - 1
# #             for x in range(n):
# #                 addnumber.append(i + x)
# #             if zz > 0.6:
# #                 adds = ((newline[i] - newline[subscript]) / breadth) + yy + 1
# #             else:
# #                 adds = ((newline[i] - newline[subscript]) / breadth) + yy
# #             newline[i] = newline[subscript] + adds * breadth
# #             print newline[i]
# #
# # if addnumber:
# #     for i in addnumber:
# #         if newline[subscript] + (i - subscript) * breadth < sp[1]:
# #             newline.insert(i, newline[subscript] + (i - subscript) * breadth)
# #
# # print newline
#
# # lists = [1, 2, 4, 5]
# # insert = [2, 3]
# # if insert:
# #     for x in range(len(insert)):
# #         insert[x] += x
# #     print insert
# #     for i in insert:
# #         lists.insert(i, i + 1)
# # for x in range(len(insert)):
# #     insert[x] -= x
# # for i in insert:
# #     del lists[i]
# #
# # print lists
#
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import threading
import Queue

def ImagePartition(original_image, number):
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
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, 10, 0)

    # 横线列表
    transverseline = []
    # 竖线列表
    verticalline = []

    for i in lines:
        for x1, y1, x2, y2 in i:
            # if y1 < 500:
            if abs(y1 - y2) < 10:
                if abs(x1 - x2) > 15:
                    if (1 * sp[0] / 3) < y1 < (3 * sp[0] / 4):
                        if 250 < x1:
                            transverseline.append([x1, y1, x2, y2])
            if abs(x1 - x2) < 10:
                if abs(y1 - y2) > 15:
                    if y1 < (2 * sp[0] / 3):
                        verticalline.append([x1, y1, x2, y2])
            # cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    for i in transverseline:
        cv2.line(img, (i[0], i[1]), (i[2], i[3]), (0, 255, 0), 2)

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122), plt.imshow(edges, cmap='gray')
    # plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122), plt.imshow(imgs, cmap='gray')
    # plt.title('Third Image'), plt.xticks([]), plt.yticks([])
    plt.show()

if __name__ == "__main__":
    ImagePartition("IMG_20170310_142125.jpg", 1)