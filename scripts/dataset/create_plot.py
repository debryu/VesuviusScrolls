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
from utils.various import Options, get_scheduler, scheduler_step
import bucket_loader as dataloader
from RepMode import Net

'''
'''
COORD = (1310,3835)
#COORD = (2289,3841)
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

# Load the model
opts = Options(
    num_epochs=100000,
    batch_size = 7,
    eval_batch_size = 2,
    lr=0.00001,#0.0000001, 
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
    resume_epoch = 13,
    path_load_model = f"G:/VS_CODE/CV/Vesuvius Challenge/models/",
    path_exp_dir='exps/test',
    device='cuda',
    gpu_ids=[0],
    skipping_samples_every = dataloader.step,
    skipping_blackSamples_every = dataloader.allBlack_step,
)
model = Net(opts).to(opts.device)
model.load_state_dict(torch.load(opts.path_load_model + f"model_DeBData_25.p"))
model.eval()

subvolume = subvolume.unsqueeze(0).unsqueeze(0).to(opts.device)
task = torch.tensor(0).unsqueeze(0).to(opts.device)
print(task)

prediction, latent = model(subvolume.to(opts.device),task)
latent -= torch.min(latent)
latent /= torch.max(latent)
print(torch.max(latent), torch.min(latent))
plot3D(latent, f"latent_{FRAG}_{x}_{y}")
plot3D(subvolume, f"{FRAG}_{x}_{y}")
plot2D(prediction, f"prediction_{FRAG}_{x}_{y}")
plot2D(sublabel, f"LABEL{FRAG}_{x}_{y}")