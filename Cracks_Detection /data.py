from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv
import pandas as pd
from sklearn.model_selection import train_test_split

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):

    # TODO implement the Dataset class according to the description

    def __init__(self, flag, csv_file, transform = tv.transforms.Compose(tv.transforms.ToTensor())):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # It takes a list of torchvision.transforms as a parameter,
        # which should include at least the following: ToPILImage(), ToTensor() and Normalize().
        # 就是读取csv文本文件到DataFrame变量中
        
        super().__init__()
        csv_file='data.csv'
        self.img_data = pd.read_csv(csv_file, sep=';')
        self.flag = flag
        # self.path = path
        self.transform = transform 
        # self.split = split
        #随机划分训练集和测试集
        self.train, self.val = train_test_split(self.img_data, random_state=42, test_size=0.2)#,random_state = split)
        if self.flag == "train" :
            self.img_data = self.train
        elif self.flag == "val" :
            self.img_data = self.val 
        
        # training set or test set
        #The second parameter is a flag “mode” of type String which can be either “val” or “train”.

    def __len__(self):#返回整个数据集的长度:
        return len(self.img_data)

    def __getitem__(self, idx):#每次怎么读数据
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # img_path, label = self.img_data[index].img_path, self.img_data[index].label
        # img_name = os.path.join(self.root_dir,
        #                         self.img_data.iloc[idx, 0])
        img_name = self.img_data.iloc[idx, 0]
        image = imread(img_name)
        # img = skimage.color.gray2rgb(*args)
        image = gray2rgb(image)
        landmarks = self.img_data.iloc[idx, 1:]
        # landmarks = np.array([landmarks])

        # landmarks_frame = pd.read_csv('data/faces/face_landmarks.csv')
        # n = 65
        # img_name = landmarks_frame.iloc[n, 0]
        # landmarks = landmarks_frame.iloc[n, 1:]
        # landmarks = np.asarray(landmarks)
        landmarks = landmarks.astype('float')
        # tensor = torch.from_numpy(np.asarray(PIL.Image.open(path))).permute(2,0,1).float() / 255
        # tensor = torchvision.transforms.functional.to_tensor(PIL.Image.open(path)) # Equivalently way
        # np.ndarray--torch.Tensor 
        image = torch.tensor(self.transform(image))
        landmarks = torch.tensor(landmarks)
        sample = [image,landmarks]
        # if self.transform:
        #     sample = self.transform(sample)

        return sample

def get_train_dataset(): 
    flag = "train"
    csv_file = "train.csv"
    transform = tv.transforms.Compose([

         tv.transforms.ToPILImage(),
         tv.transforms.ToTensor(),
         tv.transforms.Normalize(mean=train_mean, std=train_std)
        ])

   
    return ChallengeDataset(flag,csv_file, transform=transform)

        #ToPILImage(), ToTensor() and Normalize().
def get_validation_dataset():
    flag = "val"
    csv_file = "train.csv"
    transform=tv.transforms.Compose([

         tv.transforms.ToPILImage(),
         tv.transforms.ToTensor(),
         tv.transforms.Normalize(mean=train_mean, std=train_std)
        ])
    return ChallengeDataset(flag,csv_file, transform=transform)

