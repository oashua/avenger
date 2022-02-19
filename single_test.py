import torch

from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from dataset import Event_dataloader
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

train_data_dir = '/home/HuSH/data/time_align374/train'
test_data_dir = '/home/HuSH/data/time_align374/test'
model = torch.load('/home/HuSH/code/avenger/output/output_0122/avenger_19_Ws8DN3Y1.pt')

mytransform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((800,800))
    ])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_set = Event_dataloader(path=train_data_dir,transform=mytransform)
test_set = Event_dataloader(path=test_data_dir,transform=mytransform)
train_loader = DataLoader(train_set,batch_size=1,shuffle=False)
test_loader = DataLoader(test_set,batch_size=1,shuffle=False)

def single_test(data_loader=train_loader,index=0,model=model):
    for i,batch in enumerate(data_loader):
        if i!=index:
            continue
        x,y = batch
        x = x.squeeze()
        y = y.squeeze()
        x = Variable(x.float().to(device))
        with torch.no_grad():
            logits,_ = model(x)
            return logits,y

def draw(y_hat,y):
    y_hat = y_hat.cpu().detach().numpy()
    y = y.numpy()
    x = range(y.shape[0])
    fig,ax = plt.subplots()
    ax.plot(x,y_hat)
    ax.plot(x,y)
    plt.show()

if __name__ == "__main__":
    y_hat,y = single_test(train_loader,0,model)
    draw(y_hat,y)
    