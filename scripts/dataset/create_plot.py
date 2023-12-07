import sys
sys.path.append('C:/Users/debryu/Desktop/VS_CODE/HOME/CV/Vesuvius Challenge/VesuviusScrolls')  # Add the parent directory to the Python path
from utils.plotting import *
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import glob
import PIL.Image as Image
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import os
import time
import gc
import cv2 as cv
from utils.dataloader_fn import create_edge_mask, extract_training_points, normalize_training_points, extract_random_points, extract_training_and_val_points, extract_test_points, extract_render_points
from utils.augmentations import *

'''
'''
COORD = (1310,3835)
FRAG = 0
''''''
base_folder = 'C:/Users/debryu/Desktop/VS_CODE/HOME/CV/Vesuvius_ds/complete dataset/' #"G:/VS_CODE/CV/Vesuvius Challenge/"
###############################################################################
#Get the size of each fragment and get the fragments from storage
names = os.listdir(base_folder)
frag_names = [name for name in names if name.endswith('.npy')]
dataset_names = {}
fragments=[]
for i,n in enumerate(frag_names):
    dataset_names[i] = n
    size = tuple(map(int, n.split('_')[1].split('.')[0].split('-')))
    temp = np.memmap(base_folder+n, dtype=np.float32, mode='r', shape=size)
    fragments.append(temp)
###############################################################################
FRAG1_LABEL_PNG = Image.open(base_folder + f"f1_inklabels.png")
FRAG2_LABEL_PNGa = Image.open(base_folder + f"f2a_inklabels.png")
FRAG2_LABEL_PNGb = Image.open(base_folder + f"f2b_inklabels.png")
FRAG3_LABEL_PNG = Image.open(base_folder + f"f3_inklabels.png")
FRAG4_LABEL_PNG = Image.open(base_folder + f"f4_inklabels.png")
FRAG1_LABEL = torch.from_numpy(np.array(FRAG1_LABEL_PNG))
FRAG2_LABELa = torch.from_numpy(np.array(FRAG2_LABEL_PNGa))
FRAG2_LABELb = torch.from_numpy(np.array(FRAG2_LABEL_PNGb))
FRAG3_LABEL = torch.from_numpy(np.array(FRAG3_LABEL_PNG))
FRAG4_LABEL = torch.from_numpy(np.array(FRAG4_LABEL_PNG))
LABELS = [FRAG1_LABEL, FRAG2_LABELa, FRAG2_LABELb, FRAG3_LABEL, FRAG4_LABEL]

x,y = COORD
WINDOW = 32
image_stack = fragments[FRAG]
label = LABELS[FRAG]
sublabel = label[y-WINDOW:y+WINDOW, x-WINDOW:x+WINDOW]
plt.imshow(sublabel)
plt.show()
subvolume = torch.tensor(image_stack[:, y-WINDOW:y+WINDOW, x-WINDOW:x+WINDOW])
plot3D(subvolume, f"{FRAG}_{x}_{y}")
plot2D(sublabel, f"LABEL{FRAG}_{x}_{y}")