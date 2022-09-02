# 导入需要的包
import os
import random
import shutil
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

path = 'H:\\Pytorch\\face_read-try1\\lfw_funneled'

file = open('H:\\人脸识别\\分类程序\\female_names.txt')
data_female = []
for line in file.readlines():
    line = line.strip('\n')
    data_female.append(line)

file = open('H:\\人脸识别\\分类程序\\male_names.txt')
data_male = []
for line in file.readlines():
    line = line.strip('\n')
    data_male.append(line)

trainmale = 0  # 4105
testmale = 0  # 5132
valmale = 0  # 1026

trainfemale = 0  # 1186
testfemale = 0  # 1483
valfemale = 0  # 297

for filepath, dirnames, filenames in os.walk(r'H:\\人脸识别\\分类程序\\lfw_funneled'):
    for filename in filenames:  # filename就是文件名
        filename2 = os.path.join(filepath, filename)  # filename2是绝对路径
        print(filename2)
        if filename in data_male:
            n = random.randint(0, 2)
            if (n == 0) and (trainmale >= 4105):
                n += 1
            if (n == 1) and (testmale >= 5132):
                n += 1
            if (n == 2) and (valmale >= 1026):
                n -= 1
                if testmale >= 5132:
                    n -= 1
            if n == 0:
                shutil.move(filename2, 'H:\\人脸识别\\分类程序\\data\\train\\male')
                trainmale += 1
            elif n == 1:
                shutil.move(filename2, 'H:\\人脸识别\\分类程序\\data\\test\\male')
                testmale += 1
            else:
                shutil.move(filename2, 'H:\\人脸识别\\分类程序\\data\\val\\male')
                valmale += 1
        if filename in data_female:
            n = random.randint(0, 2)
            if (n == 0) and (trainfemale >= 1186):
                n += 1
            if (n == 1) and (testfemale >= 1483):
                n += 1
            if (n == 2) and (valfemale >= 297):
                n -= 1
                if testfemale >= 1483:
                    n -= 1
            if n == 0:
                shutil.move(filename2, 'H:\\人脸识别\\分类程序\\data\\train\\female')
                trainfemale += 1
            elif n == 1:
                shutil.move(filename2, 'H:\\人脸识别\\分类程序\\data\\test\\female')
                testfemale += 1
            else:
                shutil.move(filename2, 'H:\\人脸识别\\分类程序\\data\\val\\female')
                valfemale += 1
