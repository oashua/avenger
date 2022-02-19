import cv2
import numpy as np
import os
import os.path as osp
import time
import glob
import torch

# eventfile = "D:/AI/dataset/hsergb/close/test/water_bomb_floor_01/blur3/384/385.txt"
root = "D:/AI/dataset/hsergb/close/test/water_bomb_floor_01/blur3"
def splitevent(x,y,t,p,tstart,period,frame_size):
    sub_idx = np.where((t>=tstart)&(t<tstart+period))
    t = t[sub_idx]
    x = x[sub_idx].astype(int)
    y = y[sub_idx].astype(int)
    p = p[sub_idx]
    p_frame = np.zeros(frame_size)
    n_frame = np.zeros(frame_size)
    for idx in range(len(x)):
        weight = (t[idx]-tstart)/period+0.001
        if p[idx]>0.5 and x[idx]<frame_size[0] and y[idx]<frame_size[1]:
            p_frame[x[idx],y[idx]] = p_frame[x[idx],y[idx]]+weight
        elif p[idx]<0.5 and x[idx]<frame_size[0] and y[idx]<frame_size[1]:
            n_frame[x[idx],y[idx]] = n_frame[x[idx],y[idx]]+weight
    return np.concatenate((p_frame,n_frame),axis=2)
def event2frame(eventfile,periodus,frame_size,transform=None):
    event = np.loadtxt(eventfile)
    if event.size == 0 or len(event.shape)==1:
        return
    y = event[:,0]
    x = event[:,1]
    t = event[:,2]
    p = event[:,3]
    split_length=int((t[-1]-t[0])/periodus)
    if split_length == 0:
        return
    # if flag==1 and split_length <10:
    #     print('this seq: {} is too short.'.format('//'.join(eventfile.split('\\')[:-1])))
    #     return
    w = np.ones(split_length)*periodus+(t[-1]-t[0]-periodus*split_length)/split_length+1
    t_split = np.zeros_like(w)
    t_sum = t[0]
    for n in range(split_length):
        t_split[n] = t_sum
        t_sum = t_split[n]+w[n]
    frame=[]#ch0:pframe,ch1:nframe
    # baseimg = cv2.imread('D:/AI/dataset/hsergb/close/test/water_bomb_floor_01/blur3/384/385.png')
    # baseimg = baseimg/baseimg.max()
    for n in range(len(t_split)):
        sub_frame = splitevent(x,y,t,p,t_split[n],w[n],frame_size=frame_size)
        if transform is not None:
            sub_frame = transform(sub_frame)
        frame.append(torch.from_numpy(sub_frame))
        # global index
        # np.save('{}//{}_{}'.format('//'.join(eventfile.split('\\')[:-1]),index,flag),sub_frame)
        # cv2.imshow('event',np.concatenate((sub_frame,np.zeros(frame_size)),axis=2))
        # cv2.waitKey(0)
        # index = index + 1
    
    return torch.stack(frame,0),split_length
if __name__=="__main__":
    tstart = time.time()
    imggen = glob.iglob('{}/*/*.png'.format(root))
    img = cv2.imread(next(imggen))
    frame_size = img.shape[:-1]+(1,)
    for events_folder in os.listdir(root):
        current_events_dir = osp.join(root,events_folder)
        events_name = [osp.join(current_events_dir,event) for event in os.listdir(current_events_dir) if event[-3:]=="txt"]
        print(events_folder)
        for eventfile in events_name:
            event2frame(eventfile,periodus=1000,frame_size=frame_size)
    print(time.time()-tstart)