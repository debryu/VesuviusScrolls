# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 21:09:52 2023

@author: Mateo-drr
"""

import torch
import numpy as np
import PIL.Image as Image
from torch.utils.data import Dataset
import os
import gc
import cv2 
import matplotlib.pyplot as plt
from itertools import chain
import torchvision.transforms.functional as tf
import torchvision.transforms as tt
from torchvision.transforms import v2
import random
import torch.nn.functional as F

random.seed(8)
np.random.seed(7)
torch.manual_seed(6)

prob = 0.1

#Get the size of each fragment and get the fragments from storage
base_folder = 'D:/MachineLearning/datasets/VesuviusDS/'
names = os.listdir('C:/numpy/')
#names = [names[0]] + names[2:]
dataset_names = {}
fragments=[]
for i,n in enumerate(names):
    dataset_names[i] = n
    size = tuple(map(int, n.split('_')[1].split('.')[0].split('-')))
    temp = np.memmap('C:/'+'numpy/'+n, dtype=np.float32, mode='r', shape=size)
    fragments.append(temp)

#LOAD THE LABELS
#Load masks
masks=[]
lbls=[]
irs=[]
for i in range(0,4):
    if i != 1:
        masks.append(np.array(Image.open( base_folder + f"Fragments/Frag{i+1}/mask.png")))
        lbls.append(np.array(Image.open(base_folder + f"Fragments/Frag{i+1}/inklabels.png")))
        irs.append(np.array(tf.invert(Image.open(base_folder + f"Fragments/Frag{i+1}/ir.png"))))
    else:
        for j in range(0,2):
            masks.append(np.array(Image.open( base_folder + f"Fragments/Frag{i+1}/part{j+1}_mask.png")))
            lbls.append(np.array(Image.open(base_folder + f"Fragments/Frag{i+1}/part{j+1}_inklabels.png")))
            irs.append(np.array(tf.invert(Image.open(base_folder + f"Fragments/Frag{i+1}/part{j+1}_ir.png").convert('L'))))

def chunk(array, mask, lbl, ir, fragid, chunk_shape=(64, 64), valid=False):
    """
    Split a 3D array into chunks of the specified shape.
    Also splits the respective mask and label, removing chunks that fall outside the mask

    Parameters:
    - array: 3D NumPy array
    - chunk_shape: Tuple specifying the shape of each chunk

    Returns:
    - List of 3D chunks
    """
    
    if array.shape[1] % chunk_shape[0] != 0:
        array = array[:, :-(array.shape[1] % chunk_shape[0]), :]
        mask = mask[ :-(mask.shape[0] % chunk_shape[0]), :]
        lbl = lbl[ :-(lbl.shape[0] % chunk_shape[0]), :]
        ir = ir[ :-(ir.shape[0] % chunk_shape[0]), :]
    if array.shape[2] % chunk_shape[1] != 0:
        array = array[:, :, :-(array.shape[2] % chunk_shape[1])]
        mask = mask[ :, :-(mask.shape[1] % chunk_shape[1])]
        lbl = lbl[:, :-(lbl.shape[1] % chunk_shape[1])]
        ir = ir[:, :-(ir.shape[1] % chunk_shape[1])]
        
    cmask = []
    for x in np.split(mask, array.shape[1] // chunk_shape[0], axis=0):
        for xy in np.split(x, array.shape[2] // chunk_shape[1], axis=1):
            cmask.append(xy)
            
    clbl = []
    for x in np.split(lbl, array.shape[1] // chunk_shape[0], axis=0):
        for xy in np.split(x, array.shape[2] // chunk_shape[1], axis=1):
            clbl.append(xy)
            
    cir = []
    for x in np.split(ir, array.shape[1] // chunk_shape[0], axis=0):
        for xy in np.split(x, array.shape[2] // chunk_shape[1], axis=1):
            cir.append(xy)
            
    clblbin=[]
    for x in np.split(lbl, array.shape[1] // chunk_shape[0], axis=0):
        for xy in np.split(x, array.shape[2] // chunk_shape[1], axis=1):
            if np.any(xy): #
                #print('hahah')
                clblbin.append(np.ones_like(xy))
            else:
                #print('bbbb')
                clblbin.append(np.zeros_like(xy))
    
    
    chunks = []
    i = 0
    # Split along the first axis
    for a,x_chunk in enumerate(np.split(array, array.shape[1] // chunk_shape[0], axis=1)):
        # Split along the second axis
        for b,xy_chunk in enumerate(np.split(x_chunk, array.shape[2] // chunk_shape[1], axis=2)):
            #chunks.append(xy_chunk)
            # Check if the chunk has any non-zero values in the mask
            if np.any(cmask[i]) :#
                if (((np.random.random() < prob)  or np.any(clbl[i])) and not valid) or valid:
                    # Append both the image chunk and the mask chunk
                    
                    chunks.append({'chunk':xy_chunk,
                                   'lbl':clbl[i],
                                   'lblbin':clblbin[i],
                                   'ir':cir[i],
                                   'mask':cmask[i],
                                   'fragid':fragid,
                                   'coord':(a * chunk_shape[0], b * chunk_shape[1])})
            i+=1

    return chunks

#Remove validation 
s1,s2 = 3540,4600
vfrag = fragments[0][:,s1:s2,:]
vlbl = lbls[0][s1:s2,:]
vir = irs[0][s1:s2,:]
vmask = masks[0][s1:s2,:]
#colums
fragments[0] = np.concatenate([fragments[0][:, :s1, :], fragments[0][:, s2:, :]], axis=1)
lbls[0] = np.concatenate([lbls[0][ :s1, :], lbls[0][ s2:, :]], axis=0)
irs[0] = np.concatenate([irs[0][ :s1, :], irs[0][ s2:, :]], axis=0)
masks[0] = np.concatenate([masks[0][ :s1, :], masks[0][ s2:, :]], axis=0)


chunk_frag = []
for i in range(0,len(fragments)):
    chunk_frag.append(chunk(fragments[i], masks[i], lbls[i], irs[i], i+1))
    
#Separate valid data
vdata = chunk(vfrag, vmask, vlbl, vir, 1, valid=True)

#Flatten the data into a single list
tdata = list(chain(*chunk_frag))
#vdata = list(chain(*vdata))



class chunkDS(Dataset):
    def __init__(self, data,valid=False):
        self.data = data
        self.valid = valid
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        chunk = torch.tensor(self.data[idx]['chunk']).unsqueeze(0) #[1,64,64,64]
        lbl = torch.tensor(self.data[idx]['lbl']) #[64,64]
        ir = torch.tensor(self.data[idx]['ir'])/255
        
        #Select task
        task = 0
        tlbl = lbl
        # if np.random.random() > 0.5 and not self.valid:
        #     task = 1
        #     tlbl = ir 
        
        chunk, tlbl = rolling(chunk, tlbl, self.valid)
        chunk, tlbl = flipping(chunk, tlbl, self.valid)
        chunk, tlbl = jitter(chunk, tlbl, self.valid)
        chunk = masking(chunk, tlbl, self.valid)
        
        return {'chunk':chunk,
                'tlbl': tlbl,
                'lbl':lbl,
                'ir':ir,
                'coord': self.data[idx]['coord'],
                'task': task}
        
train_ds = chunkDS(tdata)
valid_ds = chunkDS(vdata, valid=True)

def reconstruct_image(chunks,key='chunk'):
    # Find the maximum coordinates to determine the shape of the reconstructed image
    max_x = max(chunk_data['coord'][0] + chunk_data[key].shape[0] for chunk_data in chunks)
    max_y = max(chunk_data['coord'][1] + chunk_data[key].shape[1] for chunk_data in chunks)

    # Create an empty array for the reconstructed image
    reconstructed_image = np.ones((max_x, max_y))

    # Iterate over the chunks
    for chunk_data in chunks:
        # Get the chunk, label, mask, and coordinates
        chunk = chunk_data[key]
        coordinates = chunk_data['coord']

        # Get the coordinates for placing the chunk in the reconstructed image
        start_x, start_y = coordinates

        # Place the chunk in the reconstructed image
        reconstructed_image[start_x:start_x+chunk.shape[0], start_y:start_y+chunk.shape[1]] = chunk

    return reconstructed_image

del fragments, vdata, tdata, chunk_frag, vfrag        
gc.collect()
        


def rolling(chunk,tlbl,valid):
    if random.random() < 0.33 and not valid:
        # Get the width of the input tensor
        width = chunk.size(2)
        
        shift_amount = random.randint(1, width - 1)
        
        # Apply X-axis translation using torch.roll
        chunk = torch.roll(chunk, shifts=shift_amount, dims=2)
        tlbl = torch.roll(tlbl, shifts=shift_amount, dims=0)
        
    if random.random() < 0.33 and not valid:
        # Get the width of the input tensor
        width = chunk.size(3)
        
        # Randomly select the amount of translation
        shift_amount = random.randint(1, width - 1)
        
        # Apply X-axis translation using torch.roll
        chunk = torch.roll(chunk, shifts=shift_amount, dims=3)
        tlbl = torch.roll(tlbl, shifts=shift_amount, dims=1)
    return chunk,tlbl
            
def flipping(chunk,tlbl,valid):
    if random.random() < 0.33 and not valid:
        chunk = torch.flip(chunk, dims=[2])
        tlbl = torch.flip(tlbl, dims=[0])
        
    if random.random() < 0.33 and not valid:
        chunk = torch.flip(chunk, dims=[3])
        tlbl = torch.flip(tlbl, dims=[1])
    return chunk,tlbl
        
def jitter(chunk,tlbl,valid):
    if random.random() < 0.5 and not valid:
        r = 0.8+random.random()*0.4
        chunk = adjust_contrast(adjust_intensity(chunk,r),r)
        tlbl = adjust_contrast(adjust_intensity(tlbl,r),r)
    return chunk,tlbl
        
def adjust_intensity(image, factor):
    return torch.clamp(image * factor, 0, 1)

def adjust_contrast(image, factor):
    return torch.clamp((image - 0.5) * factor + 0.5, 0, 1)
# for i in range(0,100):
#     print(train_ds[i]['task'])        
        
def masking(chunk,tlbl,valid, mask_prob=0.5, mask_size=16):
    """
    Randomly masks a 3D section of the input image.

    Parameters:
    - image: 3D torch tensor (C x D x H x W)
    - mask_prob: Probability of applying the mask
    - mask_size: Size of the random mask

    Returns:
    - masked_image: Image with a randomly masked section
    """
    if random.random() < mask_prob and not valid:
        # print('masked')
        mask_size = (mask_size,mask_size,mask_size)
        # Randomly choose a position for the mask
        depth, height, width = chunk.shape[1:]
        d_start = random.randint(0, depth - mask_size[0])
        h_start = random.randint(0, height - mask_size[1])
        w_start = random.randint(0, width - mask_size[2])

        # Create a binary mask with ones in the selected section
        mask = torch.ones_like(chunk)
        mask[:, d_start:d_start + mask_size[0], h_start:h_start + mask_size[1], w_start:w_start + mask_size[2]] = 0

        # Apply the mask to the image
        mchunk = chunk * mask
        
        # Create mask for 2d lbl
        # mask = torch.ones_like(tlbl)
        # mask[:, h_start:h_start + mask_size[1], w_start:w_start + mask_size[2]] = 0
        
    else:
        mchunk = chunk

    return mchunk        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        