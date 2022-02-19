# no labels
import os
import os.path as osp
from random import randint

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.core.fromnumeric import squeeze
from PIL import Image
from torch.utils.data import DataLoader, Dataset, dataloader, dataset
from torchvision import datasets, transforms, utils

import event_frame as EF
import re


class Event_dataloader(Dataset):
    def __init__(self,path,transform=None):
        super(Event_dataloader,self).__init__()
        self.path = path
        self.transform = transform
        self.img_seq_names = os.listdir(self.path)
        self.totensor = transforms.ToTensor()
        

    def __getitem__(self, index) :
        img_seq_path = os.path.join(self.path,self.img_seq_names[index])
        event_data = []
        # t0 = 0
        # t1 = 0
        # seq_len = 0
        w0 = 0
        w1 = 0
        w2 = 0
        # events_name = [osp.join(img_seq_path,event) for event in os.listdir(img_seq_path) if event[-3:]=="txt"]
        events_name = [osp.join(img_seq_path,event) for event in os.listdir(img_seq_path) if event[-2:]=="pt"]
        events_name.sort(key=lambda x:int(re.split('[\W+,_]',x)[-3]))
        frame_name = [osp.join(img_seq_path,frame) for frame in os.listdir(img_seq_path) if frame[-3:]=="png"]
        order = 0
        img_data = cv2.imread(frame_name[0])
        frame_size = img_data.shape[:-1]+(1,)
        if self.transform is not None:
            img_data = Image.fromarray(img_data)
            img_data = img_data.convert('L')
            img_data = self.totensor(img_data)
            img_data = self.transform(img_data)
            img_data = img_data.repeat(2,1,1).unsqueeze(0)
        for eventfile in events_name:
            # mark = eventfile.split('\\')[-1][:-4]
            # sub_frame,w = EF.event2frame(eventfile,periodus=1000,frame_size=frame_size,transform=self.transform)
            
            sub_frame = torch.load(eventfile) 
            if self.transform is not None:
                sub_frame = self.transform(sub_frame)
            w = int(re.split('[_.]',eventfile)[2])

            event_data.append(sub_frame)
            if order==0:
                w0 = w
                # t0 = w
                order = order + 1
                # seq_len = seq_len + w
            elif order==1:
                w1 = w
                # t1 = w
                order=order + 1
                # seq_len = seq_len + w
            else:
                w2 = w
                # t0 = t0/(t0+t1+w)
                # t1 = t0+t1/(t0+t1+w)
                # seq_len = seq_len + w
        cut_L = randint(1,w0)
        cut_R = randint(1,w2)
        label = torch.cat((torch.zeros(cut_L),torch.ones(w1),torch.zeros(cut_R)))
        return torch.cat((img_data,torch.cat(event_data)[w0-cut_L:w0+w1+cut_R])),label
    def __len__(self):
        return len(self.img_seq_names)

if __name__=="__main__":
    dataset_path = 'dataset/train'
    mytransform = transforms.Compose([
        # transforms.ToTensor(),
        transforms.Resize((800,800))
    ])
    trainset = Event_dataloader(dataset_path,transform=mytransform)
    # img,event,label = train_loader.__getitem__(0)
    trainloader = DataLoader(trainset,batch_size=1,shuffle=False)
    for step,(data,label) in enumerate(trainloader):
        img = data.squeeze()[0]
        event = data.squeeze()[1:]
        # trans_img = mytransform(Image.fromarray(imgs))
        # img = img.swapaxes(0, 1)
        # img = img.swapaxes(1, 2)
        # plt.imshow(img)
        # plt.show()
        ch3 = torch.zeros((1,800,800))
        for subevent in event[label.bool().squeeze()]:
            # subevent = torch.cat((subevent,ch3))
            # subevent = subevent.swapaxes(0, 1)
            # subevent = subevent.swapaxes(1, 2)
            # plt.imshow(subevent)
            # plt.pause(0.1)
            cv2.imshow('window',np.concatenate((img[0],np.array(subevent[1])),axis=1))
            cv2.waitKey(100)
            # cv2.waitKey()
            # cv2.imshow()
            # plt.show()
    
