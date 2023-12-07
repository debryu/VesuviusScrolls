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
from utils.dataloader_fn import create_edge_mask, extract_training_points, normalize_training_points, extract_random_points, extract_training_and_val_points
from utils.augmentations import *
import pickle
import math

FROM = 0
WINDOW = 32
# Set the window size to be odd by using 1, otherwise 0
ODD_WINDOW = 0
# The dimension of the scan subset will be 2*WINDOW + ODD_WINDOW

bucket_path = "G:/VS_CODE/CV/Vesuvius Challenge/Fragments_dataset/special_dataset/buckets/buckets_11_1000.pkl"
d1 = "G:/VS_CODE/CV/Vesuvius Challenge/Fragments_dataset/special_dataset/buckets/buckets_edge1_400_10_4000.pkl"
d2 = "G:/VS_CODE/CV/Vesuvius Challenge/Fragments_dataset/special_dataset/buckets/buckets_edge2a_400_15_6000.pkl"
d3 = "G:/VS_CODE/CV/Vesuvius Challenge/Fragments_dataset/special_dataset/buckets/buckets_edge2b_400_15_6000.pkl"
d4 = "G:/VS_CODE/CV/Vesuvius Challenge/Fragments_dataset/special_dataset/buckets/buckets_edge3_400_10_4000.pkl"
d5 = "G:/VS_CODE/CV/Vesuvius Challenge/Fragments_dataset/special_dataset/buckets/buckets_edge4_400_10_4000.pkl"
step = None
allBlack_step = None

SCAN_DEPTH = 2*WINDOW + ODD_WINDOW
SCAN_TO_LOAD = f"scan_1_{FROM}d{SCAN_DEPTH}.pt"
EVAL_WINDOW = 640 + 64*4 # Must be multiple of 64
base_folder = 'C:/Users/debryu/Desktop/VS_CODE/HOME/CV/Vesuvius_ds/complete dataset/' #"G:/VS_CODE/CV/Vesuvius Challenge/"
obf = "G:/VS_CODE/CV/Vesuvius Challenge/"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
datasets = ["Normal", "Infrared"]

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

'''
DEFINE MANUALLY ALL THE LABEL FILES

'''

FRAG1_MASK = np.array(Image.open( base_folder + f"f1_mask.png").convert('1'))
FRAG2_MASKa = np.array(Image.open( base_folder + f"f2a_mask.png").convert('1'))
FRAG2_MASKb = np.array(Image.open( base_folder + f"f2b_mask.png").convert('1'))
FRAG3_MASK = np.array(Image.open( base_folder + f"f3_mask.png").convert('1'))
FRAG4_MASK = np.array(Image.open( base_folder + f"f4_mask.png").convert('1'))
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
FRAG1_EDGES_LABEL, FRAG1_EDGES_LABEL_PT = create_edge_mask(base_folder + f"f1_inklabels.png")
FRAG2_EDGES_LABELa, FRAG2_EDGES_LABEL_PTa = create_edge_mask(base_folder + f"f2a_inklabels.png")
FRAG2_EDGES_LABELb, FRAG2_EDGES_LABEL_PTb = create_edge_mask(base_folder + f"f2b_inklabels.png")
FRAG3_EDGES_LABEL, FRAG3_EDGES_LABEL_PT = create_edge_mask(base_folder + f"f3_inklabels.png")
FRAG4_EDGES_LABEL, FRAG4_EDGES_LABEL_PT = create_edge_mask(base_folder + f"f4_inklabels.png")
'''
-------------------------------------------------------------------------------------------------------------------
'''
#FRAG1_LABEL = torch.from_numpy(np.array(FRAG1_LABEL_PNG)).gt(0).float().to(device)
#plt.imshow(FRAG1_LABEL.cpu().numpy())
#plt.show()

# Get the label as an image
label_as_img = Image.open(obf + f"Fragments/Frag1/inklabels.png")

# Crop the rectangle from the original image
#(1175,3602,63,63)
print("Evaluation window: ", EVAL_WINDOW)
val_label = label_as_img.crop((1175, 3602, 1175+EVAL_WINDOW, 3602+EVAL_WINDOW))

# Save the cropped image
val_label.save(obf + 'Fragments/Frag1/label.png')#"g:/VS_CODE/CV/Vesuvius Challenge/Fragments/Frag1/previews/label.png")



''' 
Cut out a small window for validation during training
'''
small_rect = (1175, 3602, EVAL_WINDOW-1, EVAL_WINDOW-1) #H and W bust be a multiple of 64

'''
Generate all the training coordinates (points) for each segment
'''
training_points_1, validation_points_1 = extract_training_and_val_points(FRAG1_EDGES_LABEL, small_rect)



buckets = pickle.load(open(bucket_path, "rb"))
bucks1 = pickle.load(open(d1, "rb"))
bucks2a = pickle.load(open(d2, "rb"))
bucks2b = pickle.load(open(d3, "rb"))
bucks3 = pickle.load(open(d4, "rb"))
bucks4 = pickle.load(open(d5, "rb"))

void_info = pickle.load(open("G:/VS_CODE/CV/Vesuvius Challenge/Fragments_dataset/special_dataset/buckets/void_allFrags_12000.pkl", "rb"))
void_coords = void_info['coords']
void_frags = void_info['index']

num_buckets = len(buckets)
num_elements = len(buckets[0])
all_coordinates = np.concatenate(buckets)
all_coord1 = np.concatenate(bucks1)
all_coord2a = np.concatenate(bucks2a)
all_coord2b = np.concatenate(bucks2b)
all_coord3 = np.concatenate(bucks3)
all_coord4 = np.concatenate(bucks4)
all_coords = [all_coord1, all_coord2a, all_coord2b, all_coord3, all_coord4, void_coords]
all_coords = np.concatenate(all_coords)

class_labels = np.array(  [0] * len(all_coord1)
                          + [1] * len(all_coord2a)
                           + [2] * len(all_coord2b)
                            + [3] * len(all_coord3)
                              + [4] * len(all_coord4)
                                + void_frags)
# Store all the labels
all_labels = [FRAG1_LABEL, FRAG2_LABELa, FRAG2_LABELb, FRAG3_LABEL, FRAG4_LABEL]
all_masks = [FRAG1_MASK, FRAG2_MASKa, FRAG2_MASKb, FRAG3_MASK, FRAG4_MASK]

'''
# Only frag1
all_coordinates = np.concatenate([training_points_3[0::64]])
class_labels = np.array([0] * len(training_points_3[0::64]))
all_labels = [FRAG3_LABEL]
'''
class SubvolumeDatasetEval(data.Dataset):
    def __init__(self, surfaces, labels, class_labels, coordinates,task = None):
        self.surfaces = surfaces
        self.labels = labels
        self.class_labels = class_labels
        self.coordinates = coordinates
        self.task = task
        self.epoch = 0
        self.masks = []
    def __len__(self):
        #print(self.coordinates)
        return len(self.coordinates)
    def __getitem__(self, index):
        task = 0
        scroll_id = self.class_labels[index]
        label = self.labels[scroll_id]
        image_stack = self.surfaces[scroll_id]
        y, x = self.coordinates[index]
        subvolume = image_stack[:, y-WINDOW:y+WINDOW+ODD_WINDOW, x-WINDOW:x+WINDOW+ODD_WINDOW]
        inkpatch = label[y-WINDOW:y+WINDOW+ODD_WINDOW, x-WINDOW:x+WINDOW+ODD_WINDOW]
        inkpatch = inkpatch.numpy().astype(np.uint8)
        #subvolume = torch.tensor(subvolume)
        #print(torch.max(subvolume), torch.min(subvolume))
        #jhgfjh
        data = valid_transformations(image = subvolume, mask = inkpatch)
        subvolume = data['image'].unsqueeze(0)
        inkpatch = data['mask']
        #subvolume = subvolume.view(1, SCAN_DEPTH, WINDOW*2+ODD_WINDOW, WINDOW*2+ODD_WINDOW)
        return subvolume, inkpatch, task, scroll_id

class SubvolumeDataset(data.Dataset):
    def __init__(self, surfaces, labels, class_labels, coordinates,task = None):
        self.surfaces = surfaces
        self.labels = labels
        self.class_labels = class_labels
        self.coordinates = coordinates
        self.task = task
        self.epoch = 0
        self.masks = []
    def __len__(self):
        return len(self.coordinates)
    def __getitem__(self, index):
        if index == (num_buckets*num_elements - 1):
            task = num_buckets - 1
        else:
            task = math.floor(index/num_elements)
        task = 1
        scroll_id = self.class_labels[index]
        label = self.labels[scroll_id]
        image_stack = self.surfaces[scroll_id]
        y, x = self.coordinates[index]
        subvolume = image_stack[:, y-WINDOW:y+WINDOW+ODD_WINDOW, x-WINDOW:x+WINDOW+ODD_WINDOW]
        inkpatch = label[y-WINDOW:y+WINDOW+ODD_WINDOW, x-WINDOW:x+WINDOW+ODD_WINDOW]
        inkpatch = inkpatch.numpy().astype(np.uint8)
        #Add augmentations
        data = train_transformations(image = subvolume, mask = inkpatch)
        subvolume = data['image'].unsqueeze(0)
        inkpatch = data['mask']
        return subvolume, inkpatch, task, scroll_id




train_ds = SubvolumeDataset(fragments, all_labels, class_labels, all_coords)
train_sanity_check = SubvolumeDatasetEval(fragments, all_labels, class_labels[[8,1508,2888,3888]], all_coords[[8,1508,2888,3888]])
total_train_iters = len(all_coordinates)
validation_frag1 = SubvolumeDatasetEval(fragments, all_labels, np.array([0] * len(validation_points_1)), validation_points_1, task=0)
total_val_iters = len(validation_points_1)

# Store additional information that can be useful for when we add many fragments
dataset_index = {
            0: 'Fragment 1',
            1: 'Fragment 2a (splitted in 2 by height)',
            2: 'Fragment 2b (splitted in 2 by height)',
            3: 'Fragment 3',
            4: 'Fragment 4',
}
dataset = {
            "object": dataset_index,
            "names": dataset_names,
}