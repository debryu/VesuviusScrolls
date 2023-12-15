import zarr
from torch.utils import data
import torch
from PIL import Image
import numpy as np
from utils.augmentations import *
import os
import pickle


'''
_____________________________________________________________
The segment folder has to have the following structure:
- segments/
    - zarrs/
    - zarr_group/
        - group.zarr
    - masks/
    - labels/
    - buckets/
    - edge_points/ 
_____________________________________________________________
'''
SEGMENTS_FOLDER = "G:/VS_CODE/CV/Vesuvius Challenge/Fragments/segments/"
# Define the train and validation segments
train_segments = ['20230522181603']
# train_segments = ['20230522181603','20230702185752','20230827161847','20230905134255','20230909121925']
val_segments = ['20230904135535']
'''
_____________________________________________________________
'''

# Load the zarr group
root = zarr.open_group(SEGMENTS_FOLDER + 'zarr_group/group.zarr', mode='r')
train_masks = {}    # Store the masks for the training segments
train_labels = {}   # Store the labels for the training segments
val_masks = {}      # Store the masks for the validation segments
val_labels = {}     # Store the labels for the validation segments
train_ds = []       # Store the training samples
for seg in train_segments:
    mask = np.array(Image.open(SEGMENTS_FOLDER + 'masks/' + seg + '_mask.png').convert('1'))
    x_size = mask.shape[1]
    y_size = mask.shape[0]
    train_masks[seg] = mask
    full_label = np.array(Image.open(SEGMENTS_FOLDER + 'labels/' + seg + '_inklabels.png').convert('1'))
    label = full_label[0:y_size, 0:x_size]
    train_labels[seg] = label
    '''
    Load the points from the buckets
    '''
    for file in os.listdir(SEGMENTS_FOLDER + 'buckets/'):
        name = file.split('_')[1]
        if name not in train_segments:
            print("[INFO] Segment", file, "present in the buckets but not in current train segments")
            continue
        buckets = pickle.load(open(SEGMENTS_FOLDER + 'buckets/' + file, "rb"))
        buckets = np.concatenate(buckets)
        for coord in buckets:
          y,x = coord
          sample = {
              "coordinates": (y,x),
              "segment": name
          }
          train_ds.append(sample)
val_ds = []        # Store the validation samples   
for seg in val_segments:
    mask = np.array(Image.open(SEGMENTS_FOLDER + 'masks/' + seg + '_mask.png').convert('1'))
    x_size = mask.shape[1]
    y_size = mask.shape[0]
    val_masks[seg] = mask
    full_label = np.array(Image.open(SEGMENTS_FOLDER + 'labels/' + seg + '_inklabels.png').convert('1'))
    label = full_label[0:y_size, 0:x_size]
    val_labels[seg] = label
    '''
    _______________________________________________________________________________
    TEMPORARY !!!!
    This will take the firts 1/3 of the segment on the y axis and all the x axis
    _____               _____
    |  k|               |  k|
    | i |    ------>    ----- 
    |a  |
    -----
    _______________________________________________________________________________
    '''
    x_points = x_size//64
    y_points = (y_size//64)//3
    val_segment_data = (x_points, y_points)
    for i in range(y_points):
      for j in range(x_points):
          y = i*64 + 32
          x = j*64 + 32
          sample = {
              "coordinates": (y,x),
              "segment": seg
          }
          val_ds.append(sample)

# Define the pytorch dataset
WINDOW = 32
class SubvolumeDataset(data.Dataset):
    def __init__(self, dataset, train = True, task = None):
        self.dataset = dataset
        self.task = task
        self.epoch = 0
        self.masks = []
        self.isTraining = train
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        data = self.dataset[index]
        y,x = data['coordinates']
        segment_name = data['segment']
        segment_data = root[segment_name]
        subvolume = segment_data[0:64, y-WINDOW:y+WINDOW, x-WINDOW:x+WINDOW]/65535.0    # From float16 to float32
        if self.isTraining:
            ink_label = train_labels[segment_name]
            ink_label = ink_label[y-WINDOW:y+WINDOW, x-WINDOW:x+WINDOW]
            #Add augmentations
            data = train_transformations(image = subvolume)
        else:
            ink_label = val_labels[segment_name]
            ink_label = ink_label[y-WINDOW:y+WINDOW, x-WINDOW:x+WINDOW]
            #Add augmentations
            data = valid_transformations(image = subvolume)
        subvolume = data['image']
        #ink_label = data['mask']
        ink_label = torch.tensor(ink_label, dtype=torch.float32)
        return subvolume, ink_label, 0, segment_name
    
#print("Total training samples:", len(train_ds))
samples_to_show = ((len(train_ds)-1)//len(train_segments))//4
sanity_check_items = [samples_to_show*1,samples_to_show*2,samples_to_show*3,samples_to_show*4]
train_sanity_check = SubvolumeDataset([train_ds[i] for i in sanity_check_items])
train_dataset = SubvolumeDataset(train_ds)
eval_dataset = SubvolumeDataset(val_ds, train = False)

dataset = {
    'ink': 'ink'
}
tasks = ['ink']