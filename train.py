# eval的批处理
from math import inf
import ml_collections
import argparse
import torch
import os
import os.path as osp
import numpy as np
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from dataset import Event_dataloader
from utils import getKey
from utils import seq2se,iou
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torch.multiprocessing as mp
from event_transformer import Avenger
import time

def get_config():
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size':16})
    config.hidden_size = 512
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 2048
    config.transformer.num_heads = 4
    config.transformer.num_layers = 4
    config.transformer.attention_dropout_rate = 0.2
    config.transformer.dropout_rate = 0.7
    return config
def save_model(args,model,epoch_index):
    output_dir = args.output_dir
    if not osp.exists(output_dir):
        os.mkdir(output_dir)
    torch.save(model,osp.join(output_dir,'avenger_{}_{}.pt'.format(epoch_index,getKey())))

def getAvenger(args):
    config = get_config()
    model = Avenger(config,args.img_size)
    model.to(args.device)
    return model

mytransform = transforms.Compose([
        # transforms.ToTensor(),
        transforms.Resize((800,800))
    ])
def get_loader(args):
    # train_loader = Event_dataloader(args.trainset_dir,transform=mytransform)
    # test_loader = Event_dataloader(args.testset_dir,transform=mytransform)
    train_set = Event_dataloader(path=args.trainset_dir,transform=mytransform)
    test_set = Event_dataloader(path=args.testset_dir,transform=mytransform)
    train_loader = DataLoader(train_set,batch_size=args.train_batch_size,shuffle=True)
    test_loader = DataLoader(test_set,batch_size=args.eval_batch_size,shuffle=False)
    return train_loader,test_loader
def eval(args,model,test_loader):
    eval_loss = 0
    total_acc = 0
    model.eval()
    loss_function = CrossEntropyLoss()
    for i,batch in enumerate(test_loader):
        x,y = batch
        x = x.squeeze()
        y = y.squeeze()
        x,y = Variable(x.float().to(args.device)),Variable(y.to(args.device))
        with torch.no_grad():
            logits,_ = model(x)
            batch_loss = loss_function(logits.transpose(-1,-2),y.unsqueeze(0).softmax(dim=1))#batch input for different input sequence length

            eval_loss += batch_loss.item()
            # align_seq = list(map(lambda x:1 if x>0.5 else 0,logits))
            # s1,e1 = seq2se(align_seq)
            # s2,e2 = seq2se(y)
            # iou_value = iou((s1,e1),(s2,e2))
            # total_acc += iou_value
        if i>args.test_loader_size-1:
            break
    loss = eval_loss/min(args.test_loader_size,len(test_loader))
    # acc = total_acc/len(test_loader)*args.eval_batch_size
    return loss
        
def train(args,model):

    # print('load dataset...')
    train_loader,test_loader = get_loader(args)
    # print('finish.')
    optimizer =torch.optim.Adam(model.parameters(),
                                lr=args.lr,
                                weight_decay=args.weight_decay)
    train_loss_list=[]
    val_loss_list=[]
    val_acc_list=[]
    for i in range(args.total_epoch):
        model.train()
        train_loss = 0
        for step,batch in tqdm(enumerate(train_loader)):
            x,y = batch
            x = x.squeeze()
            y = y.squeeze()
            x,y = Variable(x.float().to(args.device)),Variable(y.to(args.device))
            loss = model(x,y)
            # print(loss.item())
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if step>args.train_loader_size-1:
                break
        if (i+1)%args.check_period==0:
            save_model(args=args,model=model,epoch_index=i)
        train_loss=train_loss/min(args.train_loader_size,len(train_loader))
        train_loss_list.append(train_loss)
        # save_model(args,model,i)
        eval_loss = eval(args,model,test_loader)

        val_loss_list.append(eval_loss)
        # val_acc_list.appned(eval_acc)

        # print("------------\neval loss:{},\teval_acc:{}".format(eval_loss,eval_acc))
        print("Epoch:[{}/{}], loss:{},{}".format(i,args.total_epoch,train_loss,eval_loss))
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainset_dir",default="/home/HuSH/data/time_align374/train")
    parser.add_argument("--testset_dir",default="/home/HuSH/data/time_align374/test")
    parser.add_argument("--output_dir",default="./output/output_temp")
    parser.add_argument("--img_size",default=800,type=int)
    parser.add_argument("--train_batch_size",default=1,type=int)
    parser.add_argument("--eval_batch_size",default=1,type=int)
    parser.add_argument("--lr",default=1e-5,type=float)
    parser.add_argument("--weight_decay",default=0,type=float)
    parser.add_argument("--total_epoch",default=4,type=int)
    parser.add_argument("--check_period",default=2,type=int)
    parser.add_argument("--train_loader_size",default=inf,type=int)
    parser.add_argument("--test_loader_size",default=inf,type=int)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())) 
    print('load model...')
    model = getAvenger(args)
    print('finish.')
    train(args=args,model=model)
    # num_process = 4
    # model.share_memory()
    # processes = []
    # print('create process...')
    
    # mp.set_start_method('spawn')
    # for rank in range(num_process):
    #     p = mp.Process(target=train,args=(args,model,))
    #     p.start()
    #     processes.append(p)
    # print('finish.')
    # for p in processes:
    #     p.join()
if __name__=="__main__":
    main()
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())) 
