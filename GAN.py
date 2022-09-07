import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
torch.__version__


transform = transforms.Compose([
    transforms.ToTensor()    #0-1归一化
    ,transforms.Normalize(0.5,0.5)     #-1-1归一化  
])


#加载内置数据集
train_ds = torchvision.datasets.MNIST('data'
                                    ,train=True
                                    ,transform=transform
                                    ,download=True)


dataloader = torch.utils.data.DataLoader(train_ds,batch_size=32,shuffle=True)
imgs,target = next(iter(dataloader))


class G(nn.Module):
    def __init__(self):
        super(G,self).__init__()
        self.main = nn.Sequential(
                                nn.Linear(100,256)
                                ,nn.ReLU()
                                ,nn.Linear(256,512)
                                ,nn.ReLU()
                                ,nn.Linear(512,28*28)
                                ,nn.Tanh()
        )
    
    def forward(self,x):
        img = self.main(x)
        img = img.view(-1,28,28)
        #print(img.shape)
        return img
    
#输入为图片，输出为二分类的概率，sigmoid激活
#BCEloss计算交叉熵损失
class D(nn.Module):
    def __init__(self):
        super(D,self).__init__()
        self.main = nn.Sequential(
                                nn.Linear(28*28,512)
                                ,nn.LeakyReLU()  #判别器推荐使用LeakyRelu，小于0会有一个小的梯度
                                ,nn.Linear(512,256)
                                ,nn.LeakyReLU()
                                ,nn.Linear(256,1)
                                ,nn.Sigmoid()
        )
    
    def forward(self,x):
        x = x.view(-1,28*28)
        x = self.main(x)
        return x
    
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
g = G().to(device)
d = D().to(device)

d_optim = torch.optim.Adam(d.parameters(),lr=0.001)
g_optim = torch.optim.Adam(g.parameters(),lr=0.001)
loss_fn = torch.nn.BCELoss()   #二元交叉熵损失函数

def img_plot(model,test_input):
    prediction = np.squeeze(model(test_input).detach().cpu().numpy())
    fig = plt.figure(figsize=(4,4))
    for i in range(16):
        plt.subplot(4,4,i+1)
        plt.imshow((prediction[i]+1)/2)
        plt.axis('off')
    plt.show()
    
test_input = torch.randn(16,100,device=device)



D_loss = []
G_loss = []
#训练
epoch = 30
for i in range(epoch):
    d_epoch_loss = 0
    g_epoch_loss = 0
    count = len(dataloader)
    for step,(img,label) in enumerate(dataloader):
        img = img.to(device)
        random_noise = torch.randn(img.size(0),100,device =device)  #生成器输入的batch_size要和分类器输入一样
        
        
        #分类器的训练
        d_optim.zero_grad()  #梯度归零
        real_output = d(img)  #对分类器输入真实图片，希望为1（真）
        d_real_loss = loss_fn(real_output
                              ,torch.ones_like(real_output)  #构造全是1的label
                             )
        d_real_loss.backward()
        
        gen_img = g(random_noise)
        fake_output = d(gen_img.detach())  #对分类器输入生成图片，分类器希望为0（假），这时的优化对象是分类器，所以要截断梯度
        d_fake_loss = loss_fn(fake_output
                             ,torch.zeros_like(fake_output)  #构造全是0的label
                             )
        d_fake_loss.backward()
        
        d_loss = d_real_loss + d_fake_loss
        d_optim.step()
        
        #生成器的训练
        g_optim.zero_grad()  #梯度归零
        fake_output = d(gen_img)  #不截断梯度，希望生成器生产的图像被分类为1
        g_loss = loss_fn(fake_output
                             ,torch.ones_like(fake_output)  #构造全是1的label
                             )
        g_loss.backward()
        g_optim.step()
        
        
        with torch.no_grad():
            d_epoch_loss += d_loss
            g_epoch_loss += g_loss  #得到了每一轮epoch总loss
    with torch.no_grad():
        d_epoch_loss /= count #得到了每一轮epoch的平均step_loss
        g_epoch_loss /= count
        D_loss.append(d_epoch_loss)  
        G_loss.append(g_epoch_loss)  
        print('Epoch:',epoch)
        img_plot(g,test_input)
