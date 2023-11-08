# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 18:31:55 2023

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

STOPPING_EPOCH = 2000 # Must be multiple of 5 to match with the gradient accumulation steps
RUN_EVAL = False
EVAL_WINDOW = 150
WANDB = False
plot_prev = False
prev_image_folder = "G:/VS_CODE/CV/Vesuvius Challenge/Fragments/Frag1/previews/"


class Options:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
opts = Options(
    path_exp_dir='exps/test',
    gpu_ids=[0],
    path_load_dataset='data/all_data',
    num_epochs=1,
    batch_size=2,#dataloader.BATCH_SIZE,
    lr=0.0000002,
    criterion=nn.MSELoss(reduction='mean'),
    device='cuda',
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
)

#LOAD DATA
train_ds = dataloader.train_ds
valid_ds = dataloader.valid_ds
#CREATE DATALOADER
train_dl = DataLoader(train_ds, batch_size=opts.batch_size, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=opts.batch_size, shuffle=False)


#LOAD MODEL
model = Net(opts).to(opts.device)
if opts.path_load_model is not None and os.path.exists(opts.path_load_model):
    model.load_state(opts.path_load_model)

#DEFINE OPTIMIZER
optimizer = torch.optim.AdamW(model.parameters(), lr=opts.lr)
scaler = GradScaler()

#TRAIN & EVAL LOOP
for epoch in range(opts.num_epochs):
    t_loss = 0
    v_loss = 0
    
    model.train()
    for data in tqdm(train_dl, total=int(len(train_ds)/train_dl.batch_size)):
        signal, target, task = data
        signal = signal.to(opts.device)
        target = target.to(opts.device)
        task = task.to(opts.device)
        
        optimizer.zero_grad()
        with autocast():
            outputs = model(signal,task)
            loss = opts.criterion(outputs.unsqueeze(1), target.unsqueeze(1)).squeeze()
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        t_loss +=loss.item()


    print('E',epoch+1,'L',t_loss/len(train_ds))
    
    model.eval()
    with torch.no_grad():
        for data in valid_dl:
            signal, target, task = data
            signal = signal.to(opts.device)
            target = target.to(opts.device)
            task = task.to(opts.device)
            
            outputs = model(signal,task)
            loss = opts.criterion(outputs,target)
            
            v_loss +=loss.item()

    print('E',epoch+1,'L',v_loss/len(valid_ds))