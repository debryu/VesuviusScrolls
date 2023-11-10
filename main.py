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
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.utils.data as data
import gc
import utils.losses as losses


STOPPING_EPOCH = 200 # Must be multiple of 8 to match with the gradient accumulation steps
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
    num_epochs=10,
    batch_size = 4,
    eval_batch_size = 2,
    lr=0.0001,
    criterion=nn.MSELoss(reduction='none'),
    #criterion = losses.dice_loss(),
    interval_val=1,

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
    device='cuda',
)

def train(model, dataloader, opts, epoch, optimizer, scaler, grad_acc_steps=8, early_stopping=STOPPING_EPOCH):
    model.train()
    train_loss = []
    for i,data in enumerate(tqdm(dataloader)):
        signal, target, task, id = data
        signal = signal.to(opts.device)
        target = target.to(opts.device)
        task = task.to(opts.device)
        with autocast():
            outputs = model(signal,task)
            # Outputs shape: [2,1,64,64,64]
            # Smash the 3D output into a 2D output on Z axis
            outputs = torch.mean(outputs, dim=2).to(torch.float64)
            targets = target.unsqueeze(1).to(torch.float64)
            # Target shape: [2,1,64,64]
            loss = opts.criterion(outputs, targets).squeeze()
            #print(loss)
            loss = torch.mean(loss)
            train_loss.append(loss.item())
        loss = loss/grad_acc_steps
        scaler.scale(loss).backward()
        if (i+1) % grad_acc_steps == 0:
            #print("Step")
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        if (i+1) >= early_stopping:
            break
    print('[TRAIN] epoch ',epoch+1,' - Loss: ',np.mean(train_loss))
    return train_loss


def evaluate(model, dataloader, opts, epoch, prev_image_folder = "G:/VS_CODE/CV/Vesuvius Challenge/Fragments/Frag1/previews/",EWW = 64*14, EWH = 64*14, stride_W = 32, stride_H = 32):
    model.eval()
    eval_loss = []
    subsection_predictions = []
    squares = len(dataloader)
    with torch.no_grad():
        for data in tqdm(dataloader):
            signal, target, task, id = data
            signal = signal.to(opts.device)
            target = target.to(opts.device)
            task = task.to(opts.device)
            outputs = model(signal,task)
            outputs = torch.mean(outputs, dim=2).to(torch.float64)
            #print(outputs.shape)
            start = int((64-stride_W)/2)
            end = int(start + stride_W)
            # Add to the list of predictions
            subsection_predictions.append(outputs[:,:,start:end,start:end])
            target = target.unsqueeze(1).to(torch.float64)
            loss = opts.criterion(outputs,target)
            loss = torch.mean(loss)
            eval_loss.append(loss.item())
    # Concatenate all the pixels
    subsection_predictions = torch.cat(subsection_predictions, dim=0).squeeze(1)
    # [EWW*EWH,16,16]
    subsection_predictions = subsection_predictions.reshape(int(EWW/stride_W), int(EWH/stride_H), stride_W,stride_H).permute(0,2,1,3).reshape(EWW, EWH)
    
    # subs_predictions shape: [EWW*EWH,64,64]
    #subsection_predictions = subsection_predictions.reshape(int(EWW/64), int(EWH/64), 64,64).permute(0,2,1,3).reshape(EWW, EWH)

    #print(pixels)
    bw_output = subsection_predictions.clip(0,1).cpu().numpy() * 255
    bw_output = bw_output.astype(np.uint8)
    #print(bw_output)
    # Save the image
    plt.imsave(prev_image_folder + f"epoch_{epoch}.png", bw_output)
    print('[EVAL]',epoch+1,' - Loss: ',np.mean(eval_loss))
    return eval_loss


def main():
    train_ds = dataloader.train_ds
    valid_ds = dataloader.validation_frag1
    #LOAD DATA
    train_dl = data.DataLoader(train_ds, batch_size = opts.batch_size, shuffle=True)
    validation_dl = data.DataLoader(valid_ds, batch_size = opts.eval_batch_size, shuffle=False)

    #LOAD MODEL
    model = Net(opts).to(opts.device)
    if opts.path_load_model is not None and os.path.exists(opts.path_load_model):
        model.load_state(opts.path_load_model)

    #DEFINE OPTIMIZER
    optimizer = torch.optim.AdamW(model.parameters(), lr=opts.lr)
    scaler = GradScaler()

    #TRAIN & EVAL LOOP
    for epoch in range(opts.num_epochs):    
        train(model, train_dl, opts, epoch, optimizer, scaler)
        gc.collect()
        if (epoch+1) % opts.interval_val == 0:
            evaluate(model, validation_dl, opts, epoch)
            gc.collect()





if __name__ == "__main__":
    main()