import os
import os.path as osp
import numpy as np
import cv2
import random as rd
import shutil
# 图像和事件保存
# e,f,e,e,e,e,f,e,e,e,e,f,e,e,e
w = 3 #blur span
mode = 'close'
sence = 'fountain_schaffhauserplatz_02'
dataset_dir = 'D:/AI/dataset/hsergb'
frame_path = '{}/{}/test/{}/images_corrected'.format(dataset_dir,mode,sence)
# frame_path = 'D:/AI/dataset/hsergb/far/test/bridge_lake_01/images_corrected'
frame_timestamp = osp.join(frame_path,'timestamp.txt')
events_path = '{}/{}/test/{}/events_aligned'.format(dataset_dir,mode,sence)
save_dir = '{}/{}/test/{}/blur{}'.format(dataset_dir,mode,sence,w)

# img_size = (856,800,3)
def fmerge(frame_names,ind,w,img_size):
    sub_fn = frame_names[ind:ind+w]
    # ts = frame_time[ind]
    # te = frame_time[ind+w-1]
    out_img = np.zeros(img_size)
    for fn in sub_fn:
        img = cv2.imread(fn)
        out_img += img
    # return out_img/w,(ts,te)
    return out_img/w

def emerge(events_names,ind,w):
    sub_en = events_names[ind:ind+w]
    p = []
    t=[]
    x=[]
    y=[]
    for en in sub_en:
        e = np.load(en)
        p=np.append(p,e['p'])
        t=np.append(t,e['t'])
        x=np.append(x,e['x'])
        y=np.append(y,e['y'])
    return p,t,x,y
frame_names = [osp.join(frame_path,t) for t in os.listdir(frame_path) if t[-4:]=='.png']
events_names = [osp.join(events_path,t) for t in os.listdir(events_path)]
events_names.sort()
frame_names.sort()
frame_time = np.loadtxt(frame_timestamp)
events_time = np.load(events_names[0])
frame_names = frame_names[:len(frame_time)]




index = 0
# e,fefef,e,fefef,e...->e,fm,em,e,fm,em,e

if not osp.exists(save_dir):
    os.mkdir(save_dir)
img_size = cv2.imread(frame_names[0]).shape
while index+w+1<=len(events_names):
# while index<50:
    current_datapath = osp.join(save_dir,'{}'.format(index))
    if not osp.exists(current_datapath):
        os.mkdir(current_datapath)
    shutil.copyfile(events_names[index],osp.join(current_datapath,'{}.npz'.format(index)))
    # copy e(index) -> index/index.npz

    print(index)
    img = fmerge(frame_names,index,w,img_size=img_size)
    # img,tspan = fmerge(frame_names,index,w)
    p,t,x,y = emerge(events_names,index+1,w-1)
    print(index+1)
    cv2.imwrite(osp.join(current_datapath,'{}.png'.format(index+1)),img)
    np.savez(osp.join(current_datapath,'{}.npz'.format(index+1)),p=p,t=t,x=x,y=y)
    # save fmerge ->index/index+1.png
    # save emerge ->index/index+1.npz
    shutil.copyfile(events_names[index+w],osp.join(current_datapath,'{}.npz'.format(index+w)))
    # copy e(index+w)->index/index+w.npz
    index = index + w




