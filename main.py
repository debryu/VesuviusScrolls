# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 18:31:55 2023

@author: Mateo-drr
"""

from RepMode import Net
import chunkDS as dataloader
import os
import torch.nn as nn
import torch
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import wandb
from piq import SSIMLoss, ssim
from PIL import Image
import numpy as np
from chunkDS import reconstruct_image
import matplotlib.pyplot as plt


path = 'D:/Universidades/Trento/3S/AdvCV/CSTMRepMode/'
class Options:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
opts = Options(
    path_exp_dir='exps/test',
    gpu_ids=[0],
    path_load_dataset='data/all_data',
    num_epochs=50,
    batch_size=8,#dataloader.BATCH_SIZE,
    lr=1e-6,
    numit = 200,
    criterion=nn.MSELoss(),
    device='cuda',
    gclip = 1,
    interval_val=2,
    seed=0,
    debugging=False,
    interval_checkpoint=None,
    epoch_checkpoint = 20,
    run_name = '"Second"',
    id = None,
    tags = [],
    nn_module = 'RepMode',
    adopted_datasets = ['Normal'],#,'Infrared'],
    path_load_model = None, #"exps/test/checkpoints/model_test_0012.p"
    monitor_model = False,
    path=path
)

#def main():
if True:

    torch.backends.cudnn.benchmark = True  
    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)
    
    # STOPPING_EPOCH = 2000 # Must be multiple of 5 to match with the gradient accumulation steps
    # RUN_EVAL = False
    # EVAL_WINDOW = 150
    WANDB = True
    wimg = False
    # plot_prev = False
    #prev_image_folder = "G:/VS_CODE/CV/Vesuvius Challenge/Fragments/Frag1/previews/"
    
    
    
    
    #LOAD DATA
    train_ds = dataloader.train_ds
    valid_ds = dataloader.valid_ds
    #CREATE DATALOADER
    train_dl = DataLoader(train_ds, batch_size=opts.batch_size, shuffle=True, pin_memory=True)
    valid_dl = DataLoader(valid_ds, batch_size=opts.batch_size, shuffle=False, pin_memory=True)
    
    #LOAD MODEL
    model = Net(opts, mult_chan=16).to(opts.device)
    if opts.path_load_model is not None and os.path.exists(opts.path_load_model):
        model.load_state(opts.path_load_model)
    
    #DEFINE OPTIMIZER
    optimizer = torch.optim.AdamW(model.parameters(), lr=opts.lr)
    scaler = GradScaler()
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,5)
    #ReduceLROnPlateau(optimizer, 'min')
    #INIT WANDB
    if WANDB:
        wandb.init(name='R2',project="SSP", entity="unitnais")
        config = {
            "learning_rate": opts.lr,
            "batch_size": opts.batch_size,
            
            "num_epochs": opts.num_epochs,
        }
        wandb.config.update(config)
    
    #TRAIN & EVAL LOOP
    l2 = nn.MSELoss()#reduction='sum')
    l1 = nn.L1Loss()#reduction='sum')
    #model.load_state_dict(torch.load(path+'epochs/E50.pth'))
    
    def vfragcheck(tg,tl,lbl,lblname='T'):
        grid_array = reconstruct_image(tl)
        image = Image.fromarray((grid_array * 255).astype(np.uint8), 'L')
        image.save(path + f'pred/E{epoch+1}{lblname}.png')
        
        #Save model
        torch.save(model.state_dict(), path+f'epochs/E{epoch+1}.pth')
        
        if tg:
            grid_array_l = reconstruct_image(lbl)
            image = Image.fromarray((grid_array_l * 255).astype(np.uint8), 'L')
            image.save(path + f'pred/0{lblname}.png')
            if WANDB and wimg:
                images = wandb.Image(grid_array_l, caption="BIN MASK, IR IMG")
                wandb.log({"Target": images})
            
            tg = False
        
        pssim = ssim(torch.tensor(grid_array).unsqueeze(0).unsqueeze(0),
                     torch.tensor(grid_array_l).unsqueeze(0).unsqueeze(0))
        return pssim, grid_array
    
    tg =  True
    tgir = True
    for epoch in range(0,opts.num_epochs):
        print('\nEPOCH', epoch+1)
        #gc.collect()
        t_loss = 0
        v_loss1 = 0
        v_loss0 = 0
        
        print('\tTRAIN:')
        model.train()
        for i,data in enumerate(tqdm(train_dl, total=int(len(train_ds)/train_dl.batch_size))):
            signal, target, task = data['chunk'], data['tlbl'], data['task']
            signal = signal.to(opts.device)
            target = target.to(opts.device).to(torch.float)
            task = task.to(opts.device)
            
            optimizer.zero_grad()
            with autocast():
                outputs = model(signal,task)
                outputs,target = outputs, target.unsqueeze(1)
                
                target = target.clip(0.1,0.9)
                
                loss = opts.criterion(outputs, target)
                #loss = loss + l2(outputs, target)#*l1(outputs, target) + (0.9 - torch.max(outputs))**2 + (0.1 - torch.min(outputs))**2
                
            if WANDB:
                wandb.log({'tloss': loss})#, 'lr': scheduler.get_last_lr()[0]})
                
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), opts.gclip)
            scaler.step(optimizer)
            scaler.update()
            #scheduler.step(epoch + i / opts.numit)
    
            t_loss +=loss.item()
            #break
            if i == opts.numit-1:
                break
    
        print('\tLoss:',round(t_loss/((i+1)*opts.batch_size),6))
        
        tl = []
        if tg:
            lbl=[]
        print('\tEVAL:')
        model.eval()
        with torch.no_grad():
            #RUN on bw labels
            print('\t\tMASK')
            for data in tqdm(valid_dl, total=int(len(valid_ds)/valid_dl.batch_size)):
                signal, target, task, coord = data['chunk'], data['lbl'], data['task'], data['coord']
                signal = signal.to(opts.device)
                target = target.to(opts.device)
                task = task.to(opts.device)
                
                outputs = model(signal,task)
                outputs,target = outputs, target.unsqueeze(1)
                
                target = target.clip(0.1,0.9)
                
                loss = opts.criterion(outputs, target) 
                #loss = loss + l2(outputs, target) #*l1(outputs, target) + (0.9 - torch.max(outputs))**2 + (0.1 - torch.min(outputs))**2
                
                #outputs = outputs.clip(0,1)
                
                v_loss0 +=loss.item()
                
                for i,t in enumerate(outputs):
                    tl.append({'chunk':t.squeeze().to('cpu').detach().numpy(),
                               'coord':(coord[0][i],coord[1][i])})
                    if tg:
                        lbl.append({'chunk':target[i].squeeze().to('cpu').detach().numpy(),
                                   'coord':(coord[0][i],coord[1][i])}) 
    
            pssim0, pred0 = vfragcheck(tg, tl, lbl)
            print('\t\tLoss:',round(v_loss0/len(valid_ds),6), 'SSIM', round(pssim0.item(),4))
            
            #RUN on IR lables
            # tl = []
            # if tgir:
            #     lbl=[]
            # print('\t\tINFRARED INV')
            # for data in tqdm(valid_dl, total=int(len(valid_ds)/valid_dl.batch_size)):
            #     signal, target, task, coord = data['chunk'], data['ir'], data['task'], data['coord']
            #     signal = signal.to(opts.device)
            #     target = target.to(opts.device)
            #     task = torch.ones_like(task).to(opts.device)
                
            #     outputs = model(signal,task)
            #     outputs,target = outputs, target.unsqueeze(1)
                
            #     target = target.clip(0.1,0.9)
                
            #     loss = opts.criterion(outputs, target) 
            #     #loss = loss + l2(outputs, target) #*l1(outputs, target) + (0.9 - torch.max(outputs))**2 + (0.1 - torch.min(outputs))**2
                
            #     v_loss1 +=loss.item()
                
            #     for i,t in enumerate(outputs):
            #         tl.append({'chunk':t.squeeze().to('cpu').detach().numpy(),
            #                     'coord':(coord[0][i],coord[1][i])})
            #         if tgir:
            #             lbl.append({'chunk':target[i].squeeze().to('cpu').detach().numpy(),
            #                         'coord':(coord[0][i],coord[1][i])}) 
        
            # pssim1, pred1 = vfragcheck(tgir, tl, lbl,'I')
            # print('\t\tLoss:',round(v_loss1/len(valid_ds),6), 'SSIM', round(pssim1.item(),4))
            
        if WANDB:
            #if wimg:
                # images = wandb.Image(np.vstack((pred0,pred1)), caption="Top: Mask, Bottom: IR")
                # wandb.log({"Pred": images})
            #wandb.log({'Tloss': t_loss, 'Vloss': v_loss0, 'Vloss_ir':v_loss1, 'SSIM':pssim0, 'SSIM_ir': pssim1})
            wandb.log({'Tloss': t_loss, 'Vloss': v_loss0, 'SSIM':pssim0})

# if __name__ == "__main__":
#     main()