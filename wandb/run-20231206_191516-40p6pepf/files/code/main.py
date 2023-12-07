from RepMode import Net
#import vesuvius_dataloader as dataloader
import bucket_loader as dataloader
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
from utils.various import Options, get_scheduler, scheduler_step
import time

WANDB = True
prev_image_folder = "G:/VS_CODE/CV/Vesuvius Challenge/Fragments/Frag1/previews/"
models_folder = "G:/VS_CODE/CV/Vesuvius Challenge/models/"

opts = Options(
    num_epochs=100000,
    batch_size = 7,
    eval_batch_size = 2,
    lr=0.00001,#0.0000001, 
    criterion = losses.loss_func,
    patience = 300000,
    early_stopping_epoch = 50000,
    warmup_stopping_epoch = 400,
    warmup_epochs = 9,
    multiplier = 10000,
    cosine_epochs = 10, 
    max_grad_norm = 3.0,
    interval_val=1,
    model_checkpoint=1,
    seed=0,
    run_name = "DeBData",
    id = None,
    nn_module = 'RepMode',
    adopted_datasets = dataloader.dataset,
    resume_epoch = None,
    path_load_model = f"G:/VS_CODE/CV/Vesuvius Challenge/models/",
    path_exp_dir='exps/test',
    device='cuda',
    gpu_ids=[0],
    skipping_samples_every = dataloader.step,
    skipping_blackSamples_every = dataloader.allBlack_step,
)

#plt.ion()

def train(model, dataloader, opts, epoch, optimizer, scaler, use_wandb):
    model.train()
    train_loss = []
    tloss = []
    for i,data in enumerate(tqdm(dataloader)):
        signal, target, task, id = data
        signal = signal.to(opts.device)
        target = target.to(opts.device)
        task = task.to(opts.device)
        with autocast():
            outputs = model(signal,task).to(torch.float64)
            targets = target.unsqueeze(1).to(torch.float64)
            loss = opts.criterion(outputs, targets)

            # Get the MateoLoss-er
            m_loss = losses.mateo_loss(outputs.detach(), targets.detach())
            # Target shape: [2,1,64,64]
            # Apply a gaussian filter to the output
            #center = ((64-1)/2,(64-1)/2)
            #weight = losses.gaussian_kernel(64,center, sigma = losses.SIGMA)
            #weight = torch.tensor(weight).to(opts.device).to(torch.float64)
            #plt.imshow(weight.cpu().numpy())
            #plt.show()
            # Just to visualize that is working
            #wei = self.gaussian_kernel(4,(1.5,1.5),1.5)
            #wei = torch.tensor(wei).to(self.device)
            #tensor = torch.tensor(np.ones((4, 4))).to(self.device)
            #print(tensor*wei)
            #loss = loss * weight
            #focus_loss = torch.mean(loss)*22222
            #image_loss = SSIMLoss(data_range=1.0)
            #img_rec_loss = image_loss(outputs, targets)/2
            #class_loss = losses.dice_loss_weight_noMask(outputs, targets)*4
            if i == 0 and epoch % 1 == 0 and False:
                images = [outputs[0].squeeze(0).cpu().detach().numpy(), target[0].cpu().numpy(), outputs[1].squeeze(0).cpu().detach().numpy(), target[1].cpu().numpy()]
                #fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
                # Loop through the images and plot them on the subplots
                #for i, ax in enumerate(axes.flatten()):
                #    ax.imshow(images[i], cmap='gray')  # You may need to specify the colormap based on your images
                #    ax.axis('off')  # Turn off axis labels
                #    ax.set_title(f"CL: {loss:.2f}") 
                plt.imsave(prev_image_folder + f"train_previews/a_{epoch+1}.png", images[0])
                plt.imsave(prev_image_folder + f"train_previews/aGroundTruth_{epoch+1}.png", images[1])
                plt.imsave(prev_image_folder + f"train_previews/b_{epoch+1}.png", images[2])
                plt.imsave(prev_image_folder + f"train_previews/bGroundTruth_{epoch+1}.png", images[3])
                # Adjust layout to prevent overlap of subplots
                #plt.tight_layout()
                # Show the plot
                #plt.show(block = False)
                # Record the start time
                #start_time = time.time()
                # Wait until 5 minutes have passed or user closes the plot
                #while time.time() - start_time < 300:
                    #if plt.get_fignums():
                        # Plot window is still open
                        #plt.pause(1)
                    #else:
                        # Plot window is closed, exit the loop
                        #break

                # Close the plot window
                #plt.close('all')
            if use_wandb and False:
                wandb.log({
                            '[TRAIN] loss/iter': loss.item(),
                            '[TRAIN] tloss/iter': m_loss.item(),
                           })
        train_loss.append(loss.item())
        tloss.append(m_loss.item())
        #print(loss)
        scaler.scale(loss).backward()
        nn.utils.clip_grad_norm_(model.parameters(), opts.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        if opts.resume_epoch == None:
            re = 0
        else:
            re = opts.resume_epoch
        if epoch-1 <= opts.warmup_epochs + re:
            if (i+1) == opts.warmup_stopping_epoch:
                print("Early stopping warmup")
                break
        if opts.early_stopping_epoch is not None:
            if (i+1) == opts.early_stopping_epoch:
                break
    print('[TRAIN] epoch ',epoch,' - Loss: ',np.mean(train_loss))
    return train_loss, tloss

def evaluate_only_chunks(model, dataloader, opts):
    model.eval()
    with torch.no_grad():
        for data in tqdm(dataloader):
            signal, target, task, id = data
            signal = signal.to(opts.device)
            target = target.to(opts.device)
            task = task.to(opts.device)
            outputs = model(signal,task)
            images = [  outputs[0].squeeze(0).cpu().detach().numpy(), target[0].cpu().numpy(), outputs[1].squeeze(0).cpu().detach().numpy(), target[1].cpu().numpy(),
                        outputs[2].squeeze(0).cpu().detach().numpy(), target[2].cpu().numpy(), outputs[3].squeeze(0).cpu().detach().numpy(), target[3].cpu().numpy()]
            plt.imsave(prev_image_folder + f"train_previews/a.png", images[0])
            plt.imsave(prev_image_folder + f"train_previews/aGroundTruth.png", images[1])
            plt.imsave(prev_image_folder + f"train_previews/b.png", images[2])
            plt.imsave(prev_image_folder + f"train_previews/bGroundTruth.png", images[3])
            plt.imsave(prev_image_folder + f"train_previews/c.png", images[4])
            plt.imsave(prev_image_folder + f"train_previews/cGroundTruth.png", images[5])
            plt.imsave(prev_image_folder + f"train_previews/d.png", images[6])
            plt.imsave(prev_image_folder + f"train_previews/dGroundTruth.png", images[7])
            
    return images

def evaluate(model, dataloader, opts, epoch, prev_image_folder = "G:/VS_CODE/CV/Vesuvius Challenge/Fragments/Frag1/previews/",EWW = 64*14, EWH = 64*14, stride_W = 32, stride_H = 32):
    model.eval()
    eval_loss = []
    vloss = []
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
            loss = opts.criterion(outputs, target)
            m_loss = losses.mateo_loss(outputs.detach(), target.detach())
            # Apply a gaussian filter to the output
            #center = ((64-1)/2,(64-1)/2)
            #weight = losses.gaussian_kernel(64,center, sigma = losses.SIGMA)
            #weight = torch.tensor(weight).to(opts.device).to(torch.float64)
            #loss = loss * weight
            #focus_loss = torch.mean(loss)*22222
            #class_loss = losses.dice_loss_weight_noMask(outputs, target)*4
            #image_loss = SSIMLoss(data_range=1.0)
            #img_rec_loss = image_loss(outputs, target)/2
            #loss = (img_rec_loss + class_loss + focus_loss)/3
            eval_loss.append(loss.item())
            vloss.append(m_loss.item())
    # Concatenate all the pixels
    subsection_predictions = torch.cat(subsection_predictions, dim=0).squeeze(1)
    # [EWW*EWH,16,16]
    subsection_predictions = subsection_predictions.reshape(int(EWW/stride_W), int(EWH/stride_H), stride_W,stride_H).permute(0,2,1,3).reshape(EWW, EWH)
    
    # subs_predictions shape: [EWW*EWH,64,64]
    #subsection_predictions = subsection_predictions.reshape(int(EWW/64), int(EWH/64), 64,64).permute(0,2,1,3).reshape(EWW, EWH)

    #print(pixels)
    #bw_output = subsection_predictions.clip(0,1).cpu().numpy() * 255
    bw_output = subsection_predictions.cpu().numpy()
    #bw_output = bw_output.astype(np.uint8)
    #print(bw_output)
    # Save the image
    plt.imsave(prev_image_folder + f"epoch_{epoch}.png", bw_output)
    print('[EVAL]',epoch,' - Loss: ',np.mean(eval_loss))
    return eval_loss, vloss


def main():

    if WANDB:
        wandb.init(
            settings=wandb.Settings(start_method="thread"),
            project='SSP',
            name= opts.run_name,
            config=opts,
            id=None,
            save_code=True,
            allow_val_change=True,
        )

    '''----------------------------------------------------------------------------------------------------'''
    # Load the dataset
    train_ds = dataloader.train_ds
    valid_ds = dataloader.validation_frag1
    
    sanity_check = data.DataLoader(dataloader.train_sanity_check, batch_size = 4, shuffle=False)
    #train_dl = data.DataLoader(train_ds, batch_size = opts.batch_size, shuffle=True)
    train_dl = data.DataLoader(train_ds, batch_size = opts.batch_size, shuffle=True)
    validation_dl = data.DataLoader(valid_ds, batch_size = opts.eval_batch_size, shuffle=False)

    '''----------------------------------------------------------------------------------------------------'''
    # Load the model
    model = Net(opts).to(opts.device)
    
    if opts.resume_epoch is not None:
        model.load_state_dict(torch.load(opts.path_load_model + f"model_DeBData_{opts.resume_epoch}.p"))
        training_range = range(opts.resume_epoch+1,opts.num_epochs)
    else:
        training_range = range(opts.num_epochs)

    '''----------------------------------------------------------------------------------------------------'''
    # Optimizer, scheduler and scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=opts.lr/opts.multiplier)
    scheduler = get_scheduler(optimizer, multiplier=opts.multiplier, warmup_epoch=opts.warmup_epochs, cosine_epoch=opts.cosine_epochs)
    scaler = GradScaler()
    '''----------------------------------------------------------------------------------------------------'''


    best_model_loss = 100000000
    pat = opts.patience
    #TRAIN & EVAL LOOP
    for epoch in training_range:  
        print("LR: ",scheduler.get_lr()[0])
        losses, m_losses = train(model, train_dl, opts, epoch+1, optimizer, scaler, use_wandb=WANDB)
        final_loss = np.mean(losses)
        if WANDB:
            wandb.log({
                        '[TRAIN] loss/epoch': final_loss,
                        '[TRAIN] tloss/epoch': np.mean(m_losses),
                        '[TRAIN] lr': scheduler.get_lr()[0],
                        '[TRAIN] epoch': epoch+1,
                      })
        
        gc.collect()
        imgs = evaluate_only_chunks(model, sanity_check, opts)
        if WANDB:
            line_1 = np.concatenate((imgs[0],imgs[1]), axis=1)
            line_2 = np.concatenate((imgs[2],imgs[3]), axis=1)
            line_3 = np.concatenate((imgs[4],imgs[5]), axis=1)
            line_4 = np.concatenate((imgs[6],imgs[7]), axis=1)
            grid_array_l = np.concatenate((line_1,line_2,line_3,line_4), axis=0)
            imgs = wandb.Image(grid_array_l, caption="PREDICTION | LABEL")
            wandb.log({"Training predictions": imgs})

        if (epoch+1) % opts.interval_val == 0:
            losses, m_losses = evaluate(model, validation_dl, opts, epoch+1)
            final_loss = np.mean(losses)
            if WANDB:
                wandb.log({
                            '[VAL] loss/epoch': final_loss,
                            '[VAL] vloss/epoch': np.mean(m_losses),
                        })
            if final_loss < best_model_loss:
                best_model_loss = final_loss
                # Save the model
                torch.save(model.state_dict(), models_folder + f"best_model_{opts.run_name}_{epoch+1}.p")
                pat = opts.patience
            else:
                pat -= 1
                gc.collect()
        
        if (epoch+1) % opts.model_checkpoint == 0:
            # Save the model
            torch.save(model.state_dict(), models_folder + f"model_checkpoint_{opts.run_name}_{epoch+1}.p")
        if pat == 0:
            break
        scheduler_step(scheduler, epoch+1)
    # Save the model
    torch.save(model.state_dict(), models_folder + f"finish_{opts.run_name}_{epoch+1}.p")





if __name__ == "__main__":
    main()