import sys
sys.path.append('C:/Users/debryu/Desktop/VS_CODE/HOME/CV/Vesuvius Challenge/VesuviusScrolls')  # Add the parent directory to the Python path
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import torch.utils.data as data
import math
import pickle
import os

NUM_BUCKETS = 11
BUCKET_LIMIT = 1000


path = "G:/VS_CODE/CV/Vesuvius Challenge/Fragments_dataset/special_dataset/buckets/"
spath = "G:/VS_CODE/CV/Vesuvius Challenge/Fragments_dataset/special_dataset/"

base_folder = 'C:/Users/debryu/Desktop/VS_CODE/HOME/CV/Vesuvius_ds/complete dataset/' #"G:/VS_CODE/CV/Vesuvius Challenge/"
obf = "G:/VS_CODE/CV/Vesuvius Challenge/"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
#np.save("G:/VS_CODE/CV/Vesuvius Challenge/Fragments_dataset/special_dataset/buckets/black_coords_f2a.npy", black_coords)
buckets = pickle.load(open(path + f"buckets_{NUM_BUCKETS}_{BUCKET_LIMIT}.pkl", "rb"))


fragment = fragments[2]
for i,bucket in enumerate(buckets):
    bucket_values = []
    for coord in tqdm(bucket):
        y,x = coord
        subvolume = fragment[:, y-32:y+32, x-32:x+32]
        subvolume = subvolume.reshape(-1)
        bucket_values.append(subvolume)
    bucket_values = np.concatenate(bucket_values)
    print("plotting")
    plt.hist(bucket_values, bins = 1000)
    plt.xlim(0.02,0.98)
    plt.ylim(0,2.5e6)
    plt.title(f"Mean: {np.mean(bucket_values)} - Std: {np.std(bucket_values)}")
    plt.savefig(spath + f'bucket_{i}.png')