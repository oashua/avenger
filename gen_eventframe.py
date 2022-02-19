import os
import event_frame as EF
import os.path as osp
import cv2
import torch
from tqdm import tqdm

data_root = '/home/HuSH/data/time_align374/train'

for img_seq in tqdm(os.listdir(data_root)):
    img_seq_path = osp.join(data_root,img_seq)
    events_name = [osp.join(img_seq_path,event) for event in os.listdir(img_seq_path) if event[-3:]=="txt"]
    # events_name.sort(key=lambda x:int(x.split('/')[-1][:-4]))
    frame_name = [osp.join(img_seq_path,frame) for frame in os.listdir(img_seq_path) if frame[-3:]=="png"]
    img_data = cv2.imread(frame_name[0])
    frame_size = img_data.shape[:-1]+(1,)

    for eventfile in events_name:
        sub_frame,w = EF.event2frame(eventfile,periodus=1000,frame_size=frame_size)

        torch.save(sub_frame.permute(0,3,1,2),"{}_{}.pt".format(eventfile[:-4],w))