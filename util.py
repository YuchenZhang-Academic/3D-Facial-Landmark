import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import art3d

def getObj(path):
    with open(path) as file:
        points = []
        while 1:
            line = file.readline()
            if not line:
                break
            strs = line.split(" ")
            if strs[0] == "v":
                points.append((float(strs[1]), float(strs[2]), float(strs[3])))
            if strs[0] == "vt":
                break
        train = np.array(points)
        return train
    
def getLandmark(path):
    with open(path) as file:
        points = []
        while 1:
            line = file.readline()
            if not line or line == "/n":
                break
            strs = line.split("/")
            if strs == ['\n']:
                break
            points.append((float(strs[1]), float(strs[2]), float(strs[3])))
        test = np.array(points)
        return test
    
def norm(train, test):
    train_x = -train[:, 0]
    train_y = -train[:, 2]
    train_z = train[:, 1]
    train = np.vstack((train_x, train_y, train_z))
    test_x = -test[:, 0]
    test_y = -test[:, 2]
    test_z = test[:, 1]
    test = np.vstack((test_x, test_y, test_z))
    # longest = max((train_x.max()- train_x.min()), (train_y.max()- train_y.min()), (train_z.max()- train_z.min()))
    longest = max(np.max(abs(train_x)), np.max(abs(train_y)), np.max(abs(train_z)))
    train = train / longest
    test = test / longest
    return train, test

def findEyeBox(test):
    p_En_r = test[:, 14]
    p_En_l = test[:, 15]
    p_Ex_r = test[:, 16]
    p_Ex_l = test[:, 17]
    p_Ps_r = test[:, 18]
    p_Ps_l = test[:, 19]
    p_Pi_r = test[:, 20]
    p_Pi_l = test[:, 21]
    points = [p_En_r, p_En_l, p_Ex_r, p_Ex_l, p_Ps_r, p_Ps_l, p_Pi_r, p_Pi_l]
    mid = (p_En_r + p_En_l + p_Ex_r + p_Ex_l + p_Ps_r + p_Ps_l + p_Pi_r + p_Pi_l) / 8
    x_length = []
    for i in points:
        x_length.append(abs(mid[0] - i[0]))
    y_length = []
    for i in points:
        y_length.append(abs(mid[1] - i[1]))
    z_length = []
    for i in points:
        z_length.append(abs(mid[2] - i[2]))
    xl = (np.array(x_length).max()) * 1 + 0.02
    yl = (np.array(y_length).max()) * 1 + 0.02
    zl = (np.array(z_length).max()) * 1 + 0.02
    return np.array([mid[0], mid[1], mid[2], xl, yl, zl])

def findNoseBox(test):
    p_G = test[:, 0]
    p_Na = test[:, 1]
    p_Pn = test[:, 2]
    p_Sn = test[:, 3]
    p_A = test[:, 4]
    p_Al_r = test[:, 12]
    p_Al_l = test[:, 13]
    points = [p_G, p_Na, p_Pn, p_Sn, p_A, p_Al_r, p_Al_l]
    mid = (p_G + p_Na + p_Pn + p_Sn + p_A + p_Al_r + p_Al_l) / 7
    x_length = []
    for i in points:
        x_length.append(abs(mid[0] - i[0]))
    y_length = []
    for i in points:
        y_length.append(abs(mid[1] - i[1]))
    z_length = []
    for i in points:
        z_length.append(abs(mid[2] - i[2]))
    xl = (np.array(x_length).max()) * 1 + 0.02
    yl = (np.array(y_length).max()) * 1 + 0.02
    zl = (np.array(z_length).max()) * 1 + 0.02
    return np.array([mid[0], mid[1], mid[2], xl, yl, zl])

def findLipsBox(test):
    p_Ls = test[:, 5]
    p_Sto = test[:, 6]
    p_Li = test[:, 7]
    p_Cph_r = test[:, 26]
    p_Cph_l = test[:, 27]
    p_Ch_r = test[:, 28]
    p_Ch_l = test[:, 29]
    points = [p_Ls, p_Sto, p_Li, p_Cph_r, p_Cph_l, p_Ch_r, p_Ch_l]
    mid = (p_Ls + p_Sto + p_Li + p_Cph_r + p_Cph_l + p_Ch_r + p_Ch_l) / 7
    x_length = []
    for i in points:
        x_length.append(abs(mid[0] - i[0]))
    y_length = []
    for i in points:
        y_length.append(abs(mid[1] - i[1]))
    z_length = []
    for i in points:
        z_length.append(abs(mid[2] - i[2]))
    xl = (np.array(x_length).max()) * 1 + 0.02
    yl = (np.array(y_length).max()) * 1 + 0.02
    zl = (np.array(z_length).max()) * 1 + 0.02
    return np.array([mid[0], mid[1], mid[2], xl, yl, zl])

def findChinBox(test):
    p_B = test[:, 8]
    p_Pg = test[:, 9]
    p_Gn = test[:, 10]
    p_Me = test[:, 11]
    points = [p_B, p_Pg, p_Gn, p_Me]
    mid = (p_B + p_Pg + p_Gn + p_Me) / 4
    x_length = []
    for i in points:
        x_length.append(abs(mid[0] - i[0]))
    y_length = []
    for i in points:
        y_length.append(abs(mid[1] - i[1]))
    z_length = []
    for i in points:
        z_length.append(abs(mid[2] - i[2]))
    xl = (np.array(x_length).max()) * 1 + 0.02
    yl = (np.array(y_length).max()) * 1 + 0.02
    zl = (np.array(z_length).max()) * 1 + 0.02
    return np.array([mid[0], mid[1], mid[2], xl, yl, zl])

def findRightFaceBox(test):
    p_Tra_r = test[:, 22]
    p_Zv_r = test[:, 24]
    p_Go_r = test[:, 30]
    points = [p_Tra_r, p_Zv_r, p_Go_r]
    mid = (p_Tra_r + p_Zv_r + p_Go_r ) / 3
    x_length = []
    for i in points:
        x_length.append(abs(mid[0] - i[0]))
    y_length = []
    for i in points:
        y_length.append(abs(mid[1] - i[1]))
    z_length = []
    for i in points:
        z_length.append(abs(mid[2] - i[2]))
    xl = (np.array(x_length).max()) * 1 + 0.02
    yl = (np.array(y_length).max()) * 1 + 0.02
    zl = (np.array(z_length).max()) * 1 + 0.02
    return np.array([mid[0], mid[1], mid[2], xl, yl, zl])

def findLeftFaceBox(test):
    p_Tra_l = test[:, 23]
    p_Zv_l = test[:, 25]
    p_Go_l = test[:, 31]
    points = [p_Tra_l, p_Zv_l, p_Go_l]
    mid = (p_Tra_l + p_Zv_l + p_Go_l) / 3
    x_length = []
    for i in points:
        x_length.append(abs(mid[0] - i[0]))
    y_length = []
    for i in points:
        y_length.append(abs(mid[1] - i[1]))
    z_length = []
    for i in points:
        z_length.append(abs(mid[2] - i[2]))
    xl = (np.array(x_length).max()) * 1 + 0.02
    yl = (np.array(y_length).max()) * 1 + 0.02
    zl = (np.array(z_length).max()) * 1 + 0.02
    return np.array([mid[0], mid[1], mid[2], xl, yl, zl])

def findEyePoints(test):
    p_En_r = test[:, 14]
    p_En_l = test[:, 15]
    p_Ex_r = test[:, 16]
    p_Ex_l = test[:, 17]
    p_Ps_r = test[:, 18]
    p_Ps_l = test[:, 19]
    p_Pi_r = test[:, 20]
    p_Pi_l = test[:, 21]
    points = [p_En_r, p_En_l, p_Ex_r, p_Ex_l, p_Ps_r, p_Ps_l, p_Pi_r, p_Pi_l]
    return np.array(points)

def findNosePoints(test):
    p_G = test[:, 0]
    p_Na = test[:, 1]
    p_Pn = test[:, 2]
    p_Sn = test[:, 3]
    p_A = test[:, 4]
    p_Al_r = test[:, 12]
    p_Al_l = test[:, 13]
    points = [p_G, p_Na, p_Pn, p_Sn, p_A, p_Al_r, p_Al_l]
    return np.array(points)

def findLipsPoints(test):
    p_Ls = test[:, 5]
    p_Sto = test[:, 6]
    p_Li = test[:, 7]
    p_Cph_r = test[:, 26]
    p_Cph_l = test[:, 27]
    p_Ch_r = test[:, 28]
    p_Ch_l = test[:, 29]
    points = [p_Ls, p_Sto, p_Li, p_Cph_r, p_Cph_l, p_Ch_r, p_Ch_l]
    return np.array(points)

def findChinPoints(test):
    p_B = test[:, 8]
    p_Pg = test[:, 9]
    p_Gn = test[:, 10]
    p_Me = test[:, 11]
    points = [p_B, p_Pg, p_Gn, p_Me]
    return np.array(points)

def findRightFacePoints(test):
    p_Tra_r = test[:, 22]
    p_Zv_r = test[:, 24]
    p_Go_r = test[:, 30]
    points = [p_Tra_r, p_Zv_r, p_Go_r]
    return np.array(points)

def findLeftFacePoints(test):
    p_Tra_l = test[:, 23]
    p_Zv_l = test[:, 25]
    p_Go_l = test[:, 31]
    points = [p_Tra_l, p_Zv_l, p_Go_l]
    return np.array(points)

def getOrganPoints(path:str, organ):
    dirList = os.listdir(path)
    obj_list = []
    mark_list = []
    for i in dirList:
        FilePath = path + i 
        name = os.listdir(FilePath)
        for j in name:
            if '.obj' in j:
                obj_list.append(FilePath + '/' + j)
            else:
                mark_list.append(FilePath + '/' + j)
    obj = []
    landmark = []
    for i in obj_list:
        obj.append([getObj(i), i])
    for i in mark_list:
        landmark.append([getLandmark(i), i])
    funcs = [findEyeBox, findNoseBox, findLipsBox, findChinBox, findRightFaceBox, findLeftFaceBox]
    funcs_mark = [findEyePoints, findNosePoints, findLipsPoints, findChinPoints, findRightFacePoints, findLeftFacePoints]
    num = len(obj)
    train = []
    test = []
    for n in range(num):
        obj_in = []
        a, b = norm(obj[n][0], landmark[n][0])
        location = funcs[organ](b)
        
        x0 = location[0] - location[3]
        x1 = location[0] + location[3]

        y0 = location[1] - location[4]
        y1 = location[1] + location[4]

        z0 = location[2] - location[5]
        z1 = location[2] + location[5]

        for i in range(a.shape[1]):
            if a[0, i] > x0 and a[0, i] < x1 and a[1, i] > y0 and a[1, i] < y1 and a[2, i] > z0 and a[2, i] > z1:
                obj_in.append(a[:, i].tolist())
        if len(obj_in) == 0:
            print(obj[n][1], landmark[n][1])
        train.append(np.array(obj_in))
        test.append(funcs_mark[organ](b))
    temp = []
    k = 0
    for i in train:
        k += 1
        pointsCloud = np.zeros((201, 201, 201))
        longest = max(np.max(abs(i[:, 0])), np.max(abs(i[:, 1])), np.max(abs(i[:, 2])))
        i = i / longest
        np.around(i, 2)
        i = (i + 1) * 100
        i = i.astype(int)
        for j in i:
            pointsCloud[j[0]][j[1]][j[2]] = 1
        temp.append(pointsCloud)
    train = temp
    temp = []
    for i in test:
        # longest = max(np.max(abs(i[:, 0])), np.max(abs(i[:, 1])), np.max(abs(i[:, 2])))
        # i = i / longest
        temp.append(i)
    test = temp
    return train, test

if __name__ == '__main__':
    train, test = getOrganPoints(r'/home/zyc/project/Face/Data/', 0)
    for i in train:
        print(i)