from turtle import forward
import torch
import torch.nn as nn
import numpy as np

class Con_Token_Emb(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size,stride=stride,padding=padding),
            nn.Flatten(2),
        )
        self.norm=nn.LayerNorm(out_channel)
        
    def forward(self,x):
        x = self.conv(x).transpose(1,2)
        x = self.norm(x)
        return x

class sperate_conv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, padding=1,stride=1):
        super().__init__()
        
        # seperated conventional layer
        self.net = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=kernel_size,stride=stride, padding=padding,groups=in_channel),
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, out_channel, kernel_size=1)
        )
        
    def forward(self,x):
        return self.net(x)
        
        
class Con_Transformer_block(nn.Module):
    def __init__(self, in_channel, out_channel, num_layers=6, nhead=8, flag=False,padding=0,kernel_size=16, stride=10):
        super().__init__()
        self.flag=flag
        self.con_token_emb = Con_Token_Emb(in_channel, in_channel, kernel_size=kernel_size, padding=padding, stride=stride)
        self.sparse_conv = sperate_conv(in_channel, out_channel, kernel_size=3, padding=1, stride=1)
        self.flatten = nn.Flatten(2)
        self.encoderlayer = nn.TransformerEncoderLayer(d_model=out_channel, nhead=nhead)
        self.encoder = nn.TransformerEncoder(self.encoderlayer, num_layers=num_layers)
        if flag:
            self.cls_token = nn.Parameter(torch.zeros(1,1,out_channel))
        
    
    def forward(self,x):
        x = self.con_token_emb(x)
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.reshape(B, H, W, C).permute(0,3,1,2)
        x = self.sparse_conv(x)
        x = self.flatten(x).transpose(1,2)
        if self.flag:
            x = torch.cat((self.cls_token.expand(B,-1,-1), x), dim=1)
            
        # transformer and attention
        x = x.transpose(0,1)      # (B,N,C)->(N,B,C)
        x = self.encoder(x)
        x = x.transpose(0,1)      # (N,B,C)->(B,N,C)
        if self.flag:
            return x
        else:
            x = x.reshape(B, H, W, -1).permute(0,3,1,2) 
            return x
        
        
        

class CvT(nn.Module):
    def __init__(self,in_channel, out_channel, n_class, kernel_size, stride):
        super().__init__()
        
        self.stage1 = Con_Transformer_block(in_channel[0], out_channel[0], kernel_size=kernel_size[0], stride=stride[0])
        self.stage2 = Con_Transformer_block(in_channel[1], out_channel[1], kernel_size=kernel_size[1], stride=stride[1])
        self.stage3 = Con_Transformer_block(in_channel[2], out_channel[2], flag=True, kernel_size=kernel_size[2], stride=stride[2])
        self.mlp = nn.Sequential(
            nn.Linear(out_channel[2], out_channel[2]),
            nn.ReLU(),
            nn.Linear(out_channel[2], n_class)
        )
        
    def forward(self,x):
        
        x = self.stage1(x)
        x = self.stage2(x)        
        x = self.stage3(x)

        x = self.mlp(x[:,0])

        return x
    
    
        
if __name__=='__main__':
    # in_channel, out_channel, kernel_size, padding, stride = 1, 512, 16, 0, 12
    # net = Con_Token_Emb(in_channel, out_channel, kernel_size, padding, stride)        
    # x = torch.rand(2,1,224,224)
    # y = net(x)
    # print(y.shape)
    # block = Con_Transformer_block(in_channel, out_channel)
    # x = torch.rand(2,28*28,1)
    # y1 = block(x)
    # print(y1.shape)
    
    in_channel, out_channel = [3, 64, 128], [64, 128, 256]
    kernel_size, stride = [12, 4, 2], [10, 3, 1]
    n_class =  10
    model = CvT(in_channel, out_channel,n_class,kernel_size, stride)
    x = torch.rand(2,3,224, 224) 
    y = model(x)
    print(y.shape)
