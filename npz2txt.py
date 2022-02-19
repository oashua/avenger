import numpy as np
import os
import os.path as osp
dataset_dir = "D:/AI/dataset/hsergb/close/test/fountain_schaffhauserplatz_02/blur3"
# dataset_dir = "D:/AI/dataset/hsergb/close/test/water_bomb_floor_01/blur3"

def npz2xytp(npzfile):
    file = np.load(npzfile)
    x = file['x']
    x=np.expand_dims(x,axis=1)
    y = file['y']
    y=np.expand_dims(y,axis=1)
    t = file['t']
    t=np.expand_dims(t,axis=1)
    p = file['p']
    p=np.expand_dims(p,axis=1)
    txtfile = np.concatenate((x,y,t,p),axis=1) 
    np.savetxt('{}.txt'.format(npzfile[:-4]),txtfile)



for fold in os.listdir(dataset_dir):
    print(fold)
    fold_dir = osp.join(dataset_dir,fold)
    for npzfile in os.listdir(fold_dir):
        if npzfile[-3:]=='npz':
            npz2xytp(osp.join(fold_dir,npzfile))