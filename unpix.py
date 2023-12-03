# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 20:08:17 2023

@author: Mateo-drr
"""

from RepMode import Net
import vesuvius_dataloader as dataloader
import os
import torch.nn as nn
import torch
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import wandb
from piq import SSIMLoss, ssim, MultiScaleSSIMLoss
from PIL import Image
import numpy as np
from main import opts

#CONFIG
WANDB=False
lr=1e-5
batch=10
num_epochs=10

class UnPix(nn.Module):
    def __init__(self):
        super(UnPix, self).__init__()

        #ENCODER
        self.enc = nn.Sequential(nn.Conv2d(1, 32, 3, stride=1, padding=1, padding_mode='reflect'),
                                 nn.Mish(inplace=True),
                                 nn.Conv2d(32, 64, 3, stride=1, padding=1, padding_mode='reflect'),
                                 nn.Mish(inplace=True),
                                 nn.Conv2d(64, 128, 3, stride=1, padding=1, padding_mode='reflect'),
                                 )
        
        self.dec = nn.Sequential(nn.Conv2d(128, 64, 3, stride=1, padding=1, padding_mode='reflect'),
                                 nn.Mish(inplace=True),
                                 nn.Conv2d(64, 32, 3, stride=1, padding=1, padding_mode='reflect'),
                                 nn.Mish(inplace=True),
                                 nn.Conv2d(32, 1, 3, stride=1, padding=1, padding_mode='reflect'),
                                 )

    def forward(self, x):
        # Encoder
        x = self.enc(x)
        x = self.dec(x)
        
        return x.clip(0,1)
    

#INIT WANDB
if WANDB:
    wandb.init(name='unpix',project="SSP", entity="unitnais")
    config = {
        "learning_rate": lr,
        "batch_size": batch,
        "num_epochs": num_epochs,
    }
    wandb.config.update(config)

#LOAD MODEL
model = Net(opts).to(opts.device)
model.load_state_dict(torch.load(opts.path+'epochs/E50.pth'))
unpix = UnPix().to(opts.device)

#LOAD DATA
train_ds = dataloader.train_ds
valid_ds = dataloader.valid_ds
#CREATE DATALOADER
train_dl = DataLoader(train_ds, batch_size=batch, shuffle=False, pin_memory=True)
valid_dl = DataLoader(valid_ds, batch_size=batch, shuffle=False)

#DEFINE OPTIMIZER
optimizer = torch.optim.AdamW(unpix.parameters(), lr=1e-5)
l1 = MultiScaleSSIMLoss()

for epoch in range(0,num_epochs):
    #process 64x64x64 chunks and rebuilt a 1x640x640 image
    img=[]
    lbl=[]
    model.eval()
    t_loss=0
    for i,data in enumerate(tqdm(train_dl, total=int(len(train_ds)/train_dl.batch_size))):
        signal, target, task = data
        signal = signal.to(opts.device)
        target = target.to(opts.device).to(torch.float)
        task = task.to(opts.device)
        
        with torch.no_grad():
            outputs = model(signal,task)
        outputs,target = outputs, target.unsqueeze(1)
        
        target = target.clip(0.1,0.9)

        #iterate the batch
        for i,t in enumerate(outputs):
            lbl.append(target[i])
            img.append(t)
            
        #IF IMAGE HAS BEEN PROCESSED START UNPIX         
        if i%99 == 0:
            unpix.train()
            image = torch.stack(img,dim=0)
            target = torch.stack(lbl,dim=0)
            
            optimizer.zero_grad()
            outputs = unpix(image)
            
            loss = l1(outputs,target)
            loss.backward()
            optimizer.step()
            t_loss +=loss.item()  

            img=[]
            lbl=[]              
                
    print('E',epoch+1,'L',t_loss/(i%99))    
            
    tl = []
    lbl=[]
    v_loss=0
    model.eval()
    unpix.eval()
    with torch.no_grad():
        for data in valid_dl:
            signal, target, task = data
            signal = signal.to(opts.device)
            target = target.to(opts.device)
            task = task.to(opts.device)
            
            outputs = model(signal,task)
            outputs,target = outputs, target.unsqueeze(1)
            target = target.clip(0.1,0.9)
            
            for i,t in enumerate(outputs):
                tl.append(t)
                lbl.append(target[i])       
                    
                unpix.train()
                image = torch.stack(img,dim=0)
                target = torch.stack(lbl,dim=0)        
                        
                outputs = unpix(image)        
                        
                loss = l1(outputs,target)        
                v_loss +=loss.item()           
                pssim = ssim(outputs, target)
      
    print('E',epoch+1,'L',v_loss, 'SSIM', pssim)





              