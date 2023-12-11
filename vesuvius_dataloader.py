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

FROM = 0
WINDOW = 32
# Set the window size to be odd by using 1, otherwise 0
ODD_WINDOW = 0
# The dimension of the scan subset will be 2*WINDOW + ODD_WINDOW

TEST = False
CREATE_EVAL = True
BATCH_SIZE = 2
BATCH_SIZE_EVAL = 2
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
val_label = label_as_img.crop((1000, 3500, 1000+64*40, 3500+64*15))

# Save the cropped image
val_label.save(obf + 'Fragments/Frag1/label.png')#"g:/VS_CODE/CV/Vesuvius Challenge/Fragments/Frag1/previews/label.png")



''' 
Cut out a small window for validation during training
'''
small_rect = (1175, 3602, EVAL_WINDOW-1, EVAL_WINDOW-1) #H and W bust be a multiple of 64
small_rect = (1000, 3500, 999+64*38, 3499+64*15)
'''
Generate all the training coordinates (points) for each segment
'''
training_points_1, validation_points_1 = extract_training_and_val_points(FRAG1_EDGES_LABEL, small_rect)
training_points_2a = extract_training_points(FRAG2_EDGES_LABELa)
training_points_2b = extract_training_points(FRAG2_EDGES_LABELb)
training_points_3 = extract_training_points(FRAG3_EDGES_LABEL)
training_points_4 = extract_training_points(FRAG4_EDGES_LABEL)

val_points_1 = extract_test_points(FRAG1_LABEL, small_rect)
print(val_points_1[0], val_points_1[-1])

val_points_1 = extract_render_points(val_points_1, 64*40, 64*15)
print(len(val_points_1))
print(val_points_1[0], val_points_1[-1])

# Normalize the training points
training_points_1 = normalize_training_points(training_points_1)
training_points_2a = normalize_training_points(training_points_2a)
training_points_2b = normalize_training_points(training_points_2b)
training_points_3 = normalize_training_points(training_points_3)
training_points_4 = normalize_training_points(training_points_4)

random_points_1 = extract_random_points(FRAG1_LABEL, 60)
random_points_2a = extract_random_points(FRAG2_LABELa, 50)
random_points_2b = extract_random_points(FRAG2_LABELb, 50)
random_points_3 = extract_random_points(FRAG3_LABEL, 60)
random_points_4 = extract_random_points(FRAG4_LABEL, 60)

all_training_1, _ = extract_training_and_val_points(FRAG1_LABEL)
all_training_2a = extract_training_points(FRAG2_LABELa)
all_training_2b = extract_training_points(FRAG2_LABELb)
all_training_3 = extract_training_points(FRAG3_LABEL)
all_training_4 = extract_training_points(FRAG4_LABEL)

hpp_F1_BLACK = [
    (2383,1818),
    (2483,3237),
    (3746,2594),
    (5620,1430),
    (5553,2649),
    (6606,2749),
    (3658,3680),
    (3004,3469),
    (753,3070),
]
hpp_F1_WHITE = [
    (4168,2837),
    (764,4001),
    (6296,2527),
    (4855,1995),
    (1962,2738),
    (7294,3070),
    (5110,3602),
    (787,4811),
    (6329,1341),
]
hpp_F2a_BLACK = [
    (3476,2016),
    (3488,2116),
    (3477,2216),
    (3481,2316),
    (3478,2416),
    (3465,2516),
    (3470,2616),
    (3479,2717),
    (3477,2818),
    (3481,2919),
    (3478,3030),
    (3465,3100),
    (3470,3200),
    (3479,3300),
    (3480,3384),
]

black_squaresf102 = np.load(obf + 'Fragments_dataset/special_dataset/black02/black_coords_f1.npy')
black_squaresf2a01 = np.load(obf + 'Fragments_dataset/special_dataset/black01/black_coords_f2a.npy')
black_squaresf301_more = np.load(obf + 'Fragments_dataset/special_dataset/black01/black_coords_f3.npy') 



step = 16
allBlack_step = 64
# Concatenate coordinates
all_coordinates = np.concatenate([training_points_1[0::step], 
                                  training_points_2a[0::step], 
                                  training_points_2b[0::step], 
                                  training_points_3[0::step], 
                                  training_points_4[0::step], 
                                  hpp_F1_BLACK, 
                                  hpp_F1_WHITE,
                                  hpp_F2a_BLACK,
                                  #random_points_1,
                                  #random_points_2a,
                                  #random_points_2b,
                                  #random_points_3,
                                  #random_points_4,
                                  black_squaresf102[0::allBlack_step],
                                  black_squaresf2a01[0::allBlack_step],
                                  black_squaresf301_more[0::allBlack_step],
                                  ])

small_coordinates = np.concatenate([training_points_2b[0::step],black_squaresf102[0::allBlack_step]])

print(len(training_points_1[0::step]))
print(len(training_points_2a[0::step]))
print(len(training_points_2b[0::step]))
print(len(training_points_3[0::step]))
print(len(training_points_4[0::step]))
print(len(hpp_F1_BLACK))
print(len(hpp_F1_WHITE))
print(len(hpp_F2a_BLACK))
#print(len(random_points_1))
#print(len(random_points_2a))
#print(len(random_points_2b))
#print(len(random_points_3))
#print(len(random_points_4))
print(len(black_squaresf102))
print(len(black_squaresf2a01))
print(len(black_squaresf301_more))


# Create an array for class labels
class_labels = np.array(  [0] * len(training_points_1[0::step]) 
                        + [1] * len(training_points_2a[0::step]) 
                        + [2] * len(training_points_2b[0::step]) 
                        + [3] * len(training_points_3[0::step]) 
                        + [4] * len(training_points_4[0::step])
                        + [0] * len(hpp_F1_BLACK) 
                        + [0] * len(hpp_F1_WHITE)    
                        + [1] * len(hpp_F2a_BLACK) 
                        #+ [0] * len(random_points_1)     
                        #+ [1] * len(random_points_2a)
                        #+ [2] * len(random_points_2b)
                        #+ [3] * len(random_points_3)   
                        #+ [4] * len(random_points_4)   
                        + [0] * len(black_squaresf102[0::allBlack_step])
                        + [1] * len(black_squaresf2a01[0::allBlack_step])
                        + [3] * len(black_squaresf301_more[0::allBlack_step])                                                                                                                                                                                                                                             
                       )

# Create an array for class labels
small_labels = np.array( [2] * len(training_points_2b[0::step]) 
                        + [0] * len(black_squaresf102[0::allBlack_step])
                        )
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
        #print(x,y)
        #print(x,y)
        subvolume = image_stack[:, y-WINDOW:y+WINDOW+ODD_WINDOW, x-WINDOW:x+WINDOW+ODD_WINDOW]
        inkpatch = label[y-WINDOW:y+WINDOW+ODD_WINDOW, x-WINDOW:x+WINDOW+ODD_WINDOW]
        inkpatch = inkpatch.numpy().astype(np.uint8)
        #subvolume = torch.tensor(subvolume)
        #print(torch.max(subvolume), torch.min(subvolume))
        #jhgfjh
        #print(subvolume.shape)
        #print(inkpatch.shape)
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
        #print(self.coordinates)
        return len(self.coordinates)
    def __getitem__(self, index):
        task = 0
        scroll_id = self.class_labels[index]
        #print(index)
        #print(scroll_id)
        label = self.labels[scroll_id]
        #print('label shape: ', label.shape)
        #print(len(self.surfaces))
        image_stack = self.surfaces[scroll_id]
        #print(image_stack.shape)
        y, x = self.coordinates[index]
        #print(y, x)
        #print(self.image_stack[:, y-WINDOW:y+WINDOW+ODD_WINDOW, x-WINDOW:x+WINDOW+ODD_WINDOW].shape)
        

        subvolume = image_stack[:, y-WINDOW:y+WINDOW+ODD_WINDOW, x-WINDOW:x+WINDOW+ODD_WINDOW]
        inkpatch = label[y-WINDOW:y+WINDOW+ODD_WINDOW, x-WINDOW:x+WINDOW+ODD_WINDOW]
        inkpatch = inkpatch.numpy().astype(np.uint8)
        #Add augmentations
        data = train_transformations(image = subvolume, mask = inkpatch)
        subvolume = data['image'].unsqueeze(0)
        #print(subvolume.shape)
        inkpatch = data['mask']
        
        #subvolume = torch.tensor(subvolume)
        #subvolume = subvolume.view(1, SCAN_DEPTH, WINDOW*2+ODD_WINDOW, WINDOW*2+ODD_WINDOW)
        #inklabel = label[y, x].view(1)
        #inkpatch = label[y-WINDOW:y+WINDOW+ODD_WINDOW, x-WINDOW:x+WINDOW+ODD_WINDOW]
       
        '''
        # Generate a random value from 0 to 1
        rand = np.random.rand()
        if self.task == None:
            if rand > 0.5:
                task = 1
                inkpatch = inkpatch*-1 + 1
            else:
                task = 0
        else:
            task = self.task
        #print(inkpatch)
        #print(current_task)
        #plt.imshow(inkpatch.cpu().numpy())
        #plt.show()
        '''
        # Flipping
        '''
        if rand > 0.5:
            if rand > 0.75:
                subvolume = torch.flip(subvolume, [3])
                inkpatch = torch.flip(inkpatch, [1])
            else:
                subvolume = torch.flip(subvolume, [2])
                inkpatch = torch.flip(inkpatch, [0])
            #print(inkpatch.shape)
        '''
        #print(inkpatch.shape)
        return subvolume, inkpatch, task, scroll_id
        #return subvolume, inklabel, self.task



train_ds = SubvolumeDataset(fragments, all_labels, class_labels, all_coordinates)
small_train = SubvolumeDataset(fragments, all_labels, small_labels, small_coordinates)
train_sanity_check = SubvolumeDatasetEval(fragments, all_labels, small_labels[[8,10,24,28]], small_coordinates[[8,10,24,28]])
print('sanity ch',len(train_sanity_check))
total_train_iters = len(all_coordinates)
#print(validation_points_1)
validation_frag1 = SubvolumeDatasetEval(fragments, all_labels, np.array([0] * len(val_points_1)), val_points_1, task=0)
total_val_iters = len(validation_points_1)

#test2a = SubvolumeDatasetEval(fragments, all_labels, np.array([1] * len(test_points_1)), test_points_1, task=0)


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

tasks = [0,1,2,3,4,5,6,7,8,9]