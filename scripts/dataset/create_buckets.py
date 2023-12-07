import sys
sys.path.append('C:/Users/debryu/Desktop/VS_CODE/HOME/CV/Vesuvius Challenge/VesuviusScrolls')  # Add the parent directory to the Python path
import vesuvius_dataloader as dataloader
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import torch.utils.data as data
import math
import pickle

'''----------
CHANGE THESE'''
NUM_BUCKETS = 400
BUCKET_LIMIT = 15
NAME = "edge2b"
points_to_analyze = dataloader.training_points_2b
label = dataloader.FRAG2_LABELb
'''----------'''

buckets = [[] for _ in range(NUM_BUCKETS)]
WINDOW = 32

random_points = np.random.permutation(points_to_analyze)
print(random_points)
n_elements=0
for i,coord in enumerate(tqdm(random_points)):
    #Read the label
    y,x = coord
    if (y >7000):
      continue
    #print(coord)
    inkpatch = label[int(y-WINDOW):int(y+WINDOW), int(x-WINDOW):int(x+WINDOW)]
    white_pixels = torch.sum(inkpatch)
    total_pixels = 4*WINDOW*WINDOW
    percentage_of_white = (NUM_BUCKETS**2-1)*white_pixels/total_pixels
    
   #if white_pixels == 0:
   #     bucket_index = 0
    if percentage_of_white == 100:
        bucket_index = -1
    else:
        bucket_index = math.floor(percentage_of_white/NUM_BUCKETS)
    if len(buckets[bucket_index]) < BUCKET_LIMIT:
        buckets[bucket_index].append(coord)
        n_elements += 1
    if i%10000 == 0:
        text = f"Buckets: "
        finish = True
        for b in buckets:
            if len(b) < BUCKET_LIMIT:
                finish = False
            text += f" {len(b)}| "

        if finish:
            break  
        print(text)


step = len(buckets)//10
# show elements of the buckets
for i,b in enumerate(buckets[0::step]):
    print(f"Bucket {i}:")
    mean = np.zeros((64,64))
    for j,coord in enumerate(b):
        y,x = coord
        inkpatch = label[int(y-WINDOW):int(y+WINDOW), int(x-WINDOW):int(x+WINDOW)]
        mean = np.add(mean,inkpatch)
    mean = mean/BUCKET_LIMIT
    plt.imshow(mean)
    plt.show()
# Save the black coords

'''----------
CHANGE THE NAME
'''
path = "G:/VS_CODE/CV/Vesuvius Challenge/Fragments_dataset/special_dataset/buckets/"
#np.save("G:/VS_CODE/CV/Vesuvius Challenge/Fragments_dataset/special_dataset/buckets/black_coords_f2a.npy", black_coords)
pickle.dump(buckets, open(path + f"edge_{NAME}_{NUM_BUCKETS}_{BUCKET_LIMIT}_{n_elements}.pkl", "wb"))