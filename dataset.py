import torch
import torchvision
#import transform
import os
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
import torchvision.models as models
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import time
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
#datasets.ImageFolder()

#pretrain与finetuning过程理解：
#imageNet数据训练MoCo的encoder（pretrain） 拿NR数据集一部分finetuning，然后验证精度


class NR_dataset(Dataset):
    def __init__(self,NR_path = './data/val'):
        self.data = []
        self.label = []
        self.transform = transforms.Compose([
	        #transforms.CenterCrop((224,224)), # 只能对PIL图片进行裁剪
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
        #NR_path = 'D:\\NWPU-RESISC45'
        classes = os.listdir(NR_path)
        coumter = 0
        for i in classes:
            items = os.listdir(os.path.join(NR_path,i))
            item_path = os.path.join(NR_path,i)
            coumter = 0
            for j in items:

                tmp_img_path = os.path.join(item_path,j)
                #print(tmp_img_path)
                img_PIL = Image.open(tmp_img_path)#.convert('RGB')
                img_PIL_Tensor = self.transform(img_PIL)
                self.data.append(img_PIL_Tensor)
                self.label.append(class_dic[i])
                #print(coumter)
                    #pass
    def __getitem__(self, idx):
        img,label = self.data[idx],self.label[idx]
        return img,label
    def __len__(self):
        return len(self.data)

test = NR_dataset()

train = DataLoader(test,batch_size=24,shuffle=True)

test = models.resnet50()
#pre = torch.load("./moco_v1_200ep_pretrain.pth.tar")['state_dict']
#test.load_state_dict(pre,False)


test.fc = nn.Linear(2048,45)



#model = torch.load("C:\\Users\\HP\\Desktop\\moco_v1_200ep_pretrain.pth.tar")

device = torch.device('cuda')
test = test.to(device)
opt = torch.optim.Adam(test.parameters(),lr=1e-5)
cross = torch.nn.CrossEntropyLoss()

for epoch in range(100):
    for i, data in enumerate(train):
        # 将数据从 train_loader 中读出来,一次读取的样本数是32个
        inputs, labels = data
        inputs =inputs.to(device)
        labels = labels.to(device)
        # 将这些数据转换成Variable类型
        out = test(inputs)

        loss = cross(out,labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss)
        # 接下来就是跑模型的环节了，我们这里使用print来代替
        #print("epoch：", epoch, "的第" , i, "个inputs", inputs.data.size(), "labels", labels.data.size())
        #break
    #break
torch.save(test,'./unmoco.pth.tar')
#问题 从dataloader中划分训练集和测试集？


'''
trainLoader = DataLoader(dataset=test,batch_size=24,shuffle=True)


for epoch in range(2):
    for i, data in enumerate(trainLoader):
        # 将数据从 train_loader 中读出来,一次读取的样本数是32个
        inputs, labels = data

        # 将这些数据转换成Variable类型

        # 接下来就是跑模型的环节了，我们这里使用print来代替
        print("epoch：", epoch, "的第" , i, "个inputs", inputs.data.size(), "labels", labels.data.size())
        break
    break
'''



'''
labels = os.listdir('D:/NWPU-RESISC45')
print(labels)
print(os.path.join('D:\\NWPU-RESISC45',labels[0]))
airplane = os.listdir(os.path.join('D:\\NWPU-RESISC45',labels[0]))
print(airplane)
'''


