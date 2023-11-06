import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import glob
import PIL.Image as Image
import torch.utils.data as data
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import os
import time
import gc

startTime = time.time()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

base_folder = "G:/VS_CODE/CV/Vesuvius Challenge/"
output_folder = base_folder + "Fragments_dataset/3d_surface/"
scroll_number = 1
labels_folder = base_folder + f"Fragments/Frag{scroll_number}/"


# Load the labels
mask = np.array(Image.open( labels_folder +"mask.png").convert('1'))
label = torch.from_numpy(np.array(Image.open(labels_folder+"inklabels.png"))).gt(0).float().to(device)

# Plot the mask and the labels
'''
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.set_title("mask.png")
ax1.imshow(mask, cmap='gray')
ax2.set_title("inklabels.png")
ax2.imshow(label.cpu(), cmap='gray')
plt.show()
'''

print("Generating pixel lists...")
# Split our dataset into train and val. The pixels inside the rect are the 
# val set, and the pixels outside the rect are the train set.
# Adapted from https://www.kaggle.com/code/jamesdavey/100x-faster-pixel-coordinate-generator-1s-runtime
# Create a Boolean array of the same shape as the bitmask, initially all True

FROM = 25
WINDOW = 8
# Set the window size to be odd by using 1, otherwise 0
ODD_WINDOW = 0
# The dimension of the scan subset will be 2*WINDOW + ODD_WINDOW

BATCH_SIZE = 6
BATCH_SIZE_EVAL = 6
SCAN_DEPTH = 2*WINDOW + ODD_WINDOW
SCAN_TO_LOAD = f"scan_1_{FROM}d{SCAN_DEPTH}.pt"

# Rectangle to use as validation set
rect = (1100, 3500, 700, 950)
small_rect = (1175, 3602, 149, 149)
rect = small_rect

label_as_img = Image.open(labels_folder+"inklabels.png")
# Crop the rectangle from the original image
val_label = label_as_img.crop((1175, 3602, 1175+149, 3602+149))
# Save the cropped image
val_label.save("g:/VS_CODE/CV/Vesuvius Challenge/Fragments/Frag1/previews/label.png")

not_border = np.zeros(mask.shape, dtype=bool)
not_border[WINDOW:mask.shape[0]-WINDOW, WINDOW:mask.shape[1]-WINDOW] = True
arr_mask = np.array(mask) * not_border
inside_rect = np.zeros(mask.shape, dtype=bool) * arr_mask
# Sets all indexes with inside_rect array to True
inside_rect[rect[1]:rect[1]+rect[3]+1, rect[0]:rect[0]+rect[2]+1] = True
# Set the pixels within the inside_rect to False
outside_rect = np.ones(mask.shape, dtype=bool) * arr_mask
outside_rect[rect[1]:rect[1]+rect[3]+1, rect[0]:rect[0]+rect[2]+1] = False
pixels_inside_rect = np.argwhere(inside_rect)
print(len(pixels_inside_rect))
pixels_outside_rect = np.argwhere(outside_rect)


class SubvolumeDataset(data.Dataset):
    def __init__(self, image_stack, label, pixels, task):
        
        self.image_stack = image_stack
        self.label = label
        self.pixels = pixels
        self.task = task
    def __len__(self):
        return len(self.pixels)
    def __getitem__(self, index):
        y, x = self.pixels[index]
        subvolume = self.image_stack[:, y-WINDOW:y+WINDOW+ODD_WINDOW, x-WINDOW:x+WINDOW+ODD_WINDOW].view(1, SCAN_DEPTH, WINDOW*2+ODD_WINDOW, WINDOW*2+ODD_WINDOW)
        inklabel = self.label[y, x].view(1)
        inkpatch = self.label[y-WINDOW:y+WINDOW+ODD_WINDOW, x-WINDOW:x+WINDOW+ODD_WINDOW]
        #print(inkpatch.shape)
        return subvolume, inkpatch, self.task
        return subvolume, inklabel, self.task


print("Loading the scan...")
# Load the image stack
scan = torch.load(output_folder + SCAN_TO_LOAD).to(device)
print(scan.shape)
endTime = time.time()
print(f"Time elapsed for loading the scan: {endTime - startTime} seconds")

datasets = []
datasets.append("Frag1")
datasets.append("No task")

task = torch.eye(len(datasets), dtype=torch.long)[0]

train_dataset = SubvolumeDataset(scan, label, pixels_outside_rect,0)
train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

EVAL_WINDOW = 150
do_skip = False
skip = EVAL_WINDOW
val_pixels = []
for i,pixel in enumerate(pixels_inside_rect):
    #if i == 301:
    #    asd
    if skip == 0:
        do_skip = False
        skip = EVAL_WINDOW
    if (i % 2 != 0) and (not do_skip):
        if (i+1) % EVAL_WINDOW == 0:
            #print("ASDASDASDASDDSA\n\n\n\n ASDASDASDAS")
            do_skip = True
        #print("Skipping", i, do_skip, skip, len(val_pixels))
        continue
    if do_skip:
        skip -= 1
        #print("DO SKIP SSSSkipping", i, do_skip, skip, len(val_pixels))
        continue
    val_pixels.append(pixel)

#print("Val pixels", len(val_pixels))
#print("inside pixs", len(pixels_inside_rect))
#asd

#val_dataset = SubvolumeDataset(scan, label, pixels_inside_rect,0)
val_dataset = SubvolumeDataset(scan, label, val_pixels,0)
val_loader = data.DataLoader(val_dataset, batch_size=BATCH_SIZE_EVAL, shuffle=False)



del scan
gc.collect()