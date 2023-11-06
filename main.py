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
    num_epochs=1000,
    batch_size=dataloader.BATCH_SIZE,
    lr=0.0000002,
    criterion=nn.MSELoss,
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
    adopted_datasets = None,
    path_load_model = None, #"exps/test/checkpoints/model_test_0012.p"
    monitor_model = False,
)

#LOAD DATA
train_ds = dataloader.train_dataset
valid_ds = dataloader.val_dataset
train_dl = dataloader.train_loader
valid_dl = dataloader.val_loader

#LOAD MODEL
model = Net()
if opts.path_load_model is not None and os.path.exists(opts.path_load_model):
    model.load_state(opts.path_load_model)

#DEFINE OPTIMIZER
optimizer = torch.optim.AdamW(model.parameters(), lr=opts.lr)

#TRAIN LOOP
for epoch in range(opts.num_epochs):
    t_loss = 0
    v_loss = 0
    
    model.train()
    for data in tqdm(train_dl, total=int(len(train_ds)/train_dl.batch_size)):
        inputs, targets = data
        inputs = inputs.to(opts.device)
        targets = targets.to(opts.device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss = opts.criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        t_loss +=loss.item()

    print('E',epoch+1,'L',t_loss/len(train_ds))
    
    model.eval()
    with torch.no_grad():
        for data in valid_dl:
            inputs, targets = data
            inputs = inputs.to(opts.device)
            targets = targets.to(opts.device)
            
            outputs = model(inputs)
            loss = opts.criterion(outputs,targets)
            
            v_loss +=loss.item()

    print('E',epoch+1,'L',v_loss/len(valid_ds))