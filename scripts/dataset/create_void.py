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
from PIL import Image

'''----------
CHANGE THESE'''
POINTS_TO_COLLECT =  12000 #Half of the total which is 24000
NAME = "allFrags"
'''----------'''


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
WINDOW = 32

def return_black_p(label,mask, frag_index):
    label = label.numpy()
    border = np.zeros(label.shape, dtype=bool)
    border[WINDOW:label.shape[0]-WINDOW-1, WINDOW:label.shape[1]-WINDOW-1] = True
    black_p = np.logical_not(label)
    valid_points = black_p*border*mask
    return np.argwhere(valid_points)


labels = [FRAG1_LABEL, FRAG2_LABELa, FRAG2_LABELb, FRAG3_LABEL, FRAG4_LABEL]
all_points2a = return_black_p(FRAG2_LABELa,FRAG2_MASKa,1)
all_points2b = return_black_p(FRAG2_LABELb,FRAG2_MASKb,2)
all_points3 = return_black_p(FRAG3_LABEL,FRAG3_MASK,3)
all_points4 = return_black_p(FRAG4_LABEL,FRAG4_MASK,4)
all_sources = [1]*len(all_points2a)+[2]*len(all_points2b)+[3]*len(all_points3)+[4]*len(all_points4)

all_points = np.concatenate([all_points2a,all_points2b,all_points3,all_points4])
print("Make sure the numbers stay the same")
np.random.seed(42)
print(np.random.permutation([1,2,3,4,5]))
np.random.seed(42)
all_points_rnd = np.random.permutation(all_points)
np.random.seed(42)
print(np.random.permutation([1,2,3,4,5]))
np.random.seed(42)
all_sources_rnd = np.random.permutation(all_sources)
np.random.seed(42)
print(np.random.permutation([1,2,3,4,5]))
WINDOW = 32
void_coords = []
frag_indexes = []
for i,coord in enumerate(tqdm(all_points_rnd)):
    #Read the label
    y,x = coord
    #print(coord)
    frag_index = all_sources_rnd[i]
    label = labels[frag_index]
    inkpatch = label[int(y-WINDOW):int(y+WINDOW), int(x-WINDOW):int(x+WINDOW)]
    
    if torch.sum(inkpatch) == 0:
        void_coords.append(coord)
        frag_indexes.append(frag_index)
        if len(void_coords) == POINTS_TO_COLLECT:
            plt.imshow(inkpatch)
            plt.show()
            break
    if i%100000 == 0:
        print(len(void_coords))

print(len(void_coords))

void = {"coords": void_coords, "index":frag_indexes}

# Save the black coords
'''----------
CHANGE THE NAME
'''
path = "G:/VS_CODE/CV/Vesuvius Challenge/Fragments_dataset/special_dataset/buckets/"
#np.save("G:/VS_CODE/CV/Vesuvius Challenge/Fragments_dataset/special_dataset/buckets/black_coords_f2a.npy", black_coords)
pickle.dump(void, open(path + f"void_{NAME}_{POINTS_TO_COLLECT}.pkl", "wb"))