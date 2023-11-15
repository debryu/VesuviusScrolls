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
import wandb
from piq import SSIMLoss
import math

LOAD_MODEL = True
STOPPING_EPOCH = 8000000 # Must be multiple of 8 to match with the gradient accumulation steps
RUN_EVAL = False
EVAL_WINDOW = 150
WANDB = True
plot_prev = False
prev_image_folder = "G:/VS_CODE/CV/Vesuvius Challenge/Fragments/Frag1/previews/"
models_folder = "G:/VS_CODE/CV/Vesuvius Challenge/models/"

class Options:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
opts = Options(
    path_exp_dir='exps/test',
    gpu_ids=[0],
    path_load_dataset='data/all_data',
    num_epochs=100000,
    batch_size = 6,
    eval_batch_size = 2,
    lr=0.00001,
    criterion=nn.MSELoss(reduction='none'),
    #criterion = losses.dice_loss(),
    interval_val=1,

    seed=0,
    debugging=False,
    interval_checkpoint=None,
    epoch_checkpoint = 20,
    run_name = "DeBData",
    id = None,
    tags = [],
    nn_module = 'RepMode',
    adopted_datasets = ['Normal'],#,'Infrared'],
    path_load_model = None, #"exps/test/checkpoints/model_test_0012.p"
    monitor_model = False,
    device='cuda',
)

def train(model, dataloader, opts, epoch, optimizer, scaler, use_wandb,  grad_acc_steps=8, early_stopping=STOPPING_EPOCH):
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
            #outputs = torch.mean(outputs, dim=2).to(torch.float64)
            outputs = outputs.to(torch.float64)
            targets = target.unsqueeze(1).to(torch.float64)
            # Target shape: [2,1,64,64]
            loss = opts.criterion(outputs, targets).squeeze()
            loss_reference = torch.mean(loss).detach()
            # Apply a gaussian filter to the output
            center = ((64-1)/2,(64-1)/2)
            weight = losses.gaussian_kernel(64,center, sigma = losses.SIGMA)
            weight = torch.tensor(weight).to(opts.device).to(torch.float64)
            #plt.imshow(weight.cpu().numpy())
            #plt.show()
            # Just to visualize that is working
            #wei = self.gaussian_kernel(4,(1.5,1.5),1.5)
            #wei = torch.tensor(wei).to(self.device)
            #tensor = torch.tensor(np.ones((4, 4))).to(self.device)
            #print(tensor*wei)
            loss = loss * weight
            focus_loss = torch.mean(loss)*22222
            image_loss = SSIMLoss(data_range=1.0)
            img_rec_loss = image_loss(outputs, targets)/2
            class_loss = losses.dice_loss_weight_noMask(outputs, targets)*4
            if i == 0 and epoch % 5 == 0 and False:
                images = [outputs[0].squeeze(0).cpu().detach().numpy(), target[0].cpu().numpy(), outputs[1].squeeze(0).cpu().detach().numpy(), target[1].cpu().numpy()]
                fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
                # Loop through the images and plot them on the subplots
                for i, ax in enumerate(axes.flatten()):
                    ax.imshow(images[i], cmap='gray')  # You may need to specify the colormap based on your images
                    ax.axis('off')  # Turn off axis labels
                    ax.set_title(f"CL: {class_loss:.2f}, RL: {img_rec_loss:.2f}, FL: {focus_loss:.2f}") 
                # Adjust layout to prevent overlap of subplots
                plt.tight_layout()

                # Show the plot
                plt.show()

            if use_wandb and False:
                wandb.log({
                            '[TRAIN] focus_loss/iter': focus_loss.item(),
                            '[TRAIN] rec_loss/iter': img_rec_loss.item(),
                            '[TRAIN] class_loss/iter': class_loss.item(),
                           })

        loss = (img_rec_loss + class_loss + focus_loss)/3
        train_loss.append(loss.item())
        #print(loss)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        if (i+1) == early_stopping:
            break
    print('[TRAIN] epoch ',epoch+1,' - Loss: ',np.mean(train_loss))
    return train_loss


def evaluate(model, dataloader, opts, epoch, prev_image_folder = "G:/VS_CODE/CV/Vesuvius Challenge/Fragments/Frag1/previews/",EWW = 64*14, EWH = 64*14, stride_W = 64, stride_H = 64):
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
            #outputs = torch.mean(outputs, dim=2).to(torch.float64)
            outputs = outputs.to(torch.float64)
            #print(outputs.shape)
            start = int((64-stride_W)/2)
            end = int(start + stride_W)
            # Add to the list of predictions
            subsection_predictions.append(outputs[:,:,start:end,start:end])
            target = target.unsqueeze(1).to(torch.float64)
            loss = opts.criterion(outputs, target).squeeze()
            # Apply a gaussian filter to the output
            center = ((64-1)/2,(64-1)/2)
            weight = losses.gaussian_kernel(64,center, sigma = losses.SIGMA)
            weight = torch.tensor(weight).to(opts.device).to(torch.float64)
            loss = loss * weight
            focus_loss = torch.mean(loss)*22222
            class_loss = losses.dice_loss_weight_noMask(outputs, target)*4
            image_loss = SSIMLoss(data_range=1.0)
            img_rec_loss = image_loss(outputs, target)/2
            loss = (img_rec_loss + class_loss + focus_loss)/3
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
    plt.imsave(prev_image_folder + f"epoch_{epoch+1}.png", bw_output)
    print('[EVAL]',epoch+1,' - Loss: ',np.mean(eval_loss))
    return eval_loss


def main():

    if WANDB:
        wandb.init(
            settings=wandb.Settings(start_method="thread"),
            project='SSP',
            name= opts.run_name,
            tags=opts.tags,
            config=opts,
            id=None,
            save_code=True,
            allow_val_change=True,
        )
    train_ds = dataloader.train_ds
    valid_ds = dataloader.validation_frag1
    #LOAD DATA
    train_dl = data.DataLoader(train_ds, batch_size = opts.batch_size, shuffle=True)
    validation_dl = data.DataLoader(valid_ds, batch_size = opts.eval_batch_size, shuffle=False)

    #LOAD MODEL
    model = Net(opts).to(opts.device)
    
    if LOAD_MODEL:
        s_epoch = 7
        model.load_state_dict(torch.load(models_folder + f"model_DeBData_{s_epoch}.p"))
        training_range = range(s_epoch+1,opts.num_epochs)
    else:
        training_range = range(opts.num_epochs)

    #DEFINE OPTIMIZER
    optimizer = torch.optim.AdamW(model.parameters(), lr=opts.lr)
    #optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr)
    #optimizer = torch.optim.SGD(model.parameters(), lr=opts.lr)
    scaler = GradScaler()


    best_model_loss = 100000000
    patience = 5
    #TRAIN & EVAL LOOP
    for epoch in training_range:  
        losses = train(model, train_dl, opts, epoch, optimizer, scaler, use_wandb=WANDB)
        final_loss = np.mean(losses)
        stats = {
                    '[TRAIN] loss/epoch': final_loss,
                 }
        if WANDB:
            wandb.log(stats)
        if final_loss < best_model_loss:
            best_model_loss = final_loss
            # Save the model
            torch.save(model.state_dict(), models_folder + f"model_{opts.run_name}_{epoch+1}.p")
            patience = 10
        else:
            patience -= 1

        if patience == 0:
            break
        gc.collect()
        if (epoch+1) % opts.interval_val == 0:
            losses = evaluate(model, validation_dl, opts, epoch)
            stats = {
                        '[VAL] loss/epoch': np.mean(losses),
                    }
            if WANDB:
                wandb.log(stats)
            gc.collect()





if __name__ == "__main__":
    main()