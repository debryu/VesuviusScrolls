import sys
sys.path.append('C:/Users/debryu/Desktop/VS_CODE/HOME/CV/Vesuvius Challenge/VesuviusScrolls')  # Add the parent directory to the Python path
import vesuvius_dataloader as dataloader
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import torch.utils.data as data
import math

WINDOW = 32
train_ds = dataloader.train_ds
n_samples = len(train_ds)
batch_size = 8
iters = math.floor(n_samples/batch_size)
#print(n_samples)
#LOAD DATA
'''----------
CHANGE THESE'''
points_to_analyze = dataloader.training_points_2a
label = dataloader.FRAG2_LABELa
'''----------'''
black_coords = []
for i,coord in enumerate(tqdm(points_to_analyze)):
    #Read the label
    y,x = coord
    if (y >7000):
      continue
    #print(coord)
    inkpatch = label[int(y-WINDOW):int(y+WINDOW), int(x-WINDOW):int(x+WINDOW)]
    white_pixels = torch.sum(inkpatch)
    total_pixels = 4*WINDOW*WINDOW
    if(white_pixels/total_pixels < 0.1):
        black_coords.append(coord)

# Save the black coords

'''----------
CHANGE THE NAME
'''
np.save("G:/VS_CODE/CV/Vesuvius Challenge/Fragments_dataset/special_dataset/black01/black_coords_f2a.npy", black_coords)
