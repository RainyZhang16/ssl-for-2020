import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import torchvision.models as models
'''

model = models.resnet50(pretrained=False)
pre = torch.load("C:\\Users\\HP\\Desktop\\moco_v1_200ep_pretrain.pth.tar")['state_dict']
model.load_state_dict(pre,False)


print(model)
'''
'''
a = [0.9357142857142857,
0.8142857142857143,
0.9196428571428571,
0.85,
0.8696428571428572,
0.9017857142857143,
0.9857142857142858,
0.6017857142857143,
0.9517857142857142,
0.9767857142857143,
0.7714285714285715,
0.7964285714285714,
0.9178571428571428,
0.95,
0.8160714285714286,
0.9375,
0.8928571428571429,
0.9589285714285715,
0.8053571428571429,
0.8803571428571428,
0.9339285714285714,
0.8589285714285714,
0.8928571428571429,
0.7875,
0.8839285714285714,
0.8892857142857142,
0.8321428571428572,
0.6321428571428571,
0.9160714285714285,
0.8642857142857143,
0.7857142857142857,
0.9,
0.7642857142857142,
0.9232142857142858,
0.9071428571428571,
0.9714285714285714,
0.8660714285714286,
0.9785714285714285,
0.8767857142857143,
0.8964285714285715,
0.9482142857142857,
0.85,
0.8946428571428572,
0.8071428571428572,
0.8035714285714286]
print(sum(a)/45)
class_dic = {'airplane': 0,
                     'airport': 1,
                     'baseball_diamond': 2,
                     'basketball_court': 3,
                     'beach': 4,
                     'bridge': 5,
                     'chaparral': 6,
                     'church': 7,
                     'circular_farmland': 8,
                     'cloud': 9,
                     'commercial_area': 10,
                     'dense_residential': 11,
                     'desert': 12,
                     'forest': 13,
                     'freeway': 14,
                     'golf_course': 15,
                     'ground_track_field': 16,
                     'harbor': 17,
                     'industrial_area': 18,
                     'intersection': 19,
                     'island': 20,
                     'lake': 21,
                     'meadow': 22,
                     'medium_residential': 23,
                     'mobile_home_park': 24,
                     'mountain': 25,
                     'overpass': 26,
                     'palace': 27,
                     'parking_lot': 28,
                     'railway': 29,
                     'railway_station': 30,
                     'rectangular_farmland': 31,
                     'river': 32,
                     'roundabout': 33,
                     'runway': 34,
                     'sea_ice': 35,
                     'ship': 36,
                     'snowberg': 37,
                     'sparse_residential': 38,
                     'stadium': 39,
                     'storage_tank': 40,
                     'tennis_court': 41,
                     'terrace': 42,
                     'thermal_power_station': 43,
                     'wetland': 44}
counter = 0
for i in class_dic.keys():
        print(i,end='---')
        print(counter)
        counter += 1
'''



def cal_acc(path):
    f = open(path,'r')
    data = f.readlines()
    f.close()
    res = []
    for i in range(45):
        temp = data[i].split(',')
        totol = 0
        for j in range(45):
            totol += float(temp[j])
        res.append(float(temp[i])/totol)
    return res
moco = 'C:/Users/HP/OneDrive/2020上学期/实验结果/大测试集结果/moco.txt'
unmoco = 'C:/Users/HP/OneDrive/2020上学期/实验结果/大测试集结果/unmoco.txt'
import numpy as np
moco_acc = cal_acc(moco)
unmocoo_acc = cal_acc(unmoco)

res = np.argsort(moco_acc)
print(res)

for i in range(1,6):
    print(1-moco_acc[res[i]],end='----')
    print(res[i])
print('moco准确度',end=':')
print(1-sum(moco_acc)/45)

res = np.argsort(unmocoo_acc)
for i in range(1,6):
    print(1-unmocoo_acc[res[i]],end='----')
    print(res[i])
print('准确度',end=':')
print(1-sum(unmocoo_acc)/45)


