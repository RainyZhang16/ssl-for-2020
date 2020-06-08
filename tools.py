import torch
import torchvision
import os
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

'''
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
'''


# datasets.ImageFolder()

# pretrain与finetuning过程理解：
# imageNet数据训练MoCo的encoder（pretrain） 拿NR数据集一部分finetuning，然后验证精度

class NR_dataset(Dataset):
    def __init__(self, NR_path='./data/train'):
        self.data = []
        self.label = []
        self.transform = transforms.Compose([
            # transforms.CenterCrop((224,224)), # 只能对PIL图片进行裁剪
            transforms.ToTensor(),
        ])
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
        classes = os.listdir(NR_path)
        for i in classes:
            items = os.listdir(os.path.join(NR_path, i))
            item_path = os.path.join(NR_path, i)
            for j in items:
                tmp_img_path = os.path.join(item_path, j)
                # print(tmp_img_path)
                img_PIL = Image.open(tmp_img_path)  # .convert('RGB')
                img_PIL_Tensor = self.transform(img_PIL)
                self.data.append(img_PIL_Tensor)
                self.label.append(class_dic[i])
                # print(coumter)

    def __getitem__(self, idx):
        img, label = self.data[idx], self.label[idx]
        return img, label

    def __len__(self):
        return len(self.data)


def update_res(matrix, res, label):
    for i in range(len(res)):
        if res[i] == label[i]:
            matrix[res[i], res[i]] += 1
        else:
            matrix[label[i], res[i]] += 1
    return matrix


def cal_acc(model, dataset):
    test = DataLoader(dataset, batch_size=64, shuffle=True)

    device = torch.device('cuda')

    res = np.zeros((45, 45))

    model = model.to(device)

    with torch.no_grad():
        for epoch in range(1):
            for i, data in enumerate(test):
                # 将数据从 train_loader 中读出来,一次读取的样本数是32个
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                # 将这些数据转换成Variable类型
                out = model(inputs)

                res_label = torch.argmax(out, dim=1)

                res = update_res(res, res_label, labels)

    #                 print(res_label)
    #                 print(labels)
    #                 print(res)
    return res


def cal_class_acc(res):
    class_acc = []
    for i in range(res.shape[0]):
        class_acc.append(res[i, i] / sum(res[i, :]))
        print(class_acc[i])
    # print(sum(class_acc)/len(class_acc))
    return 1


model = torch.load('./unmoco.pth.tar')
test_data = NR_dataset()
res = cal_acc(model, test_data)
cal_class_acc(res)
f = open('./unmoco.txt','w')
for i in range(45):
    for j in range(45):
        f.write(str(res[i,j]))
        f.write(',')
    f.write('\n')
f.close()