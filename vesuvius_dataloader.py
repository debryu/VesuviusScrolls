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
import cv2 as cv
import random


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
EVAL_WINDOW = 640 # Must be multiple of 64
base_folder = 'D:/MachineLearning/datasets/VesuviusDS/'#"G:/VS_CODE/CV/Vesuvius Challenge/"
output_folder = base_folder + "Fragments_dataset/3d_surface/"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
datasets = ["Normal", "Infrared"]

###############################################################################
#Get the size of each fragment and get the fragments from storage
names = os.listdir(base_folder+'numpy/')
fragments=[]
for n in names:
    size = tuple(map(int, n.split('_')[1].split('.')[0].split('-')))
    temp = np.memmap(base_folder+'numpy/'+n, dtype=np.float32, mode='r', shape=size)
    fragments.append(temp)
###############################################################################

class SubvolumeDataset(data.Dataset):
    
    def __init__(self, image_stack, label, edge, pixels, task = None):
        self.edge = edge
        self.image_stack = image_stack
        self.label = label
        self.pixels = pixels
        self.task = task
        
    def __len__(self):
        return len(np.concatenate(self.pixels,axis=0))
    
    def __getitem__(self, idx):
        current_task = self.task
        # If there is no task, assign one randomly
        if current_task == None:
            T = np.random.rand()
            if T < 0.667:
                current_task= 0
            else:
                current_task = 1

        #randomly pick one of the fragments
        randfrag = random.choice([0,1,2])
        y, x = self.pixels[randfrag][idx]
        subvolume = self.image_stack[randfrag]
        t1 = self.label[randfrag]
        t2 = self.edge[randfrag]

        #Generate a random number between 0 and 1
        rand = np.random.rand()
        subvolume = subvolume[:, y-WINDOW:y+WINDOW+ODD_WINDOW, x-WINDOW:x+WINDOW+ODD_WINDOW]
        subvolume = torch.tensor(subvolume)
        subvolume = subvolume.view(1, SCAN_DEPTH, WINDOW*2+ODD_WINDOW, WINDOW*2+ODD_WINDOW)
        if current_task == 0:
            inklabel = t1[y, x].view(1)
            inkpatch = t1[y-WINDOW:y+WINDOW+ODD_WINDOW, x-WINDOW:x+WINDOW+ODD_WINDOW]
        if current_task == 1:
            inklabel = t2[y, x].view(1)/255
            inkpatch = t2[y-WINDOW:y+WINDOW+ODD_WINDOW, x-WINDOW:x+WINDOW+ODD_WINDOW]/255


        
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
        return subvolume, inkpatch, current_task
        #return subvolume, inklabel, self.task


'''

'''
def extract_training_points(FRAG_MASK, validation_rect= (1175,3602,63,63)):
    not_border = np.zeros(FRAG_MASK.shape, dtype=bool)
    not_border[WINDOW:FRAG_MASK.shape[0]-WINDOW, WINDOW:FRAG_MASK.shape[1]-WINDOW] = True
    arr_mask = np.array(FRAG_MASK) * not_border
    
    # Initialize the validation patch as big as the whole mask
    validation = np.zeros(FRAG_MASK.shape, dtype=bool) * arr_mask
    # and then set the inner rectangle to True
    validation[validation_rect[1]:validation_rect[1]+validation_rect[3]+1, validation_rect[0]:validation_rect[0]+validation_rect[2]+1] = True
    # Sets all the pixels inside the mask to True
    train = np.ones(FRAG_MASK.shape, dtype=bool) * arr_mask
    # and then set the pixels within the validation rectangle to False
    train[validation_rect[1]:validation_rect[1]+validation_rect[3]+1, validation_rect[0]:validation_rect[0]+validation_rect[2]+1] = False
    train = np.argwhere(train)
    validation = np.argwhere(validation)
    # Create a subset for faster rendering
    val = extract_render_points(validation, validation_rect[2]+1, validation_rect[3]+1)
    return train, val

def extract_render_points(pixels, original_width, original_height, FOV = 64):
    W = original_width
    H = original_height
    pixels_to_render = []
    points_per_row = int(W/FOV)
    points_per_col = int(H/FOV)
    pix_count_row = 32
    pix_count_col = 31
    print("start")
    val_pixels_0overlap = []
    for i in range(0, points_per_col):
        for j in range(0, points_per_row):
            #print(f'i:{i}/122 - j:{j}/82  -- Row: {pix_count_row}, Col: {pix_count_col}')
            pixels_to_render.append(pixels[pix_count_row + W*pix_count_col])
            #print(len(pixels_to_render))
            pix_count_row += 64
        pix_count_col += 64
        pix_count_row = 32
    return pixels_to_render

def create_edge_mask(image_path):
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    edges = cv.Canny(img, 100, 200)
    # Define a kernel for dilation
    kernel = np.ones((64, 64), np.uint8)  # You can adjust the size of the kernel
    # Dilate the edges to increase the thickness
    dilated_edges = cv.dilate(edges, kernel, iterations=1)
    edge_mask = np.array(torch.from_numpy(np.array(dilated_edges)).gt(155).float().to('cpu'))
    edge_mask_pt = torch.from_numpy(np.array(dilated_edges)).gt(155).float().to(device)
    #plt.imshow(edge_mask)
    #plt.show()
    return edge_mask, edge_mask_pt

FRAG1_MASK = np.array(Image.open( base_folder + f"Fragments/Frag1/mask.png").convert('1'))
FRAG3_MASK = np.array(Image.open( base_folder + f"Fragments/Frag3/mask.png").convert('1'))
FRAG4_MASK = np.array(Image.open( base_folder + f"Fragments/Frag4/mask.png").convert('1'))


FRAG1_EDGES_LABEL, FRAG1_EDGES_LABEL_PT = create_edge_mask(base_folder + f"Fragments/Frag1/inklabels.png")
FRAG3_EDGES_LABEL, FRAG3_EDGES_LABEL_PT = create_edge_mask(base_folder + f"Fragments/Frag3/inklabels.png")
FRAG4_EDGES_LABEL, FRAG4_EDGES_LABEL_PT = create_edge_mask(base_folder + f"Fragments/Frag4/inklabels.png")

FRAG1_IR_PNG = Image.open(base_folder + f"Fragments/Frag1/irnc.png")



FRAG1_LABEL_PNG = Image.open(base_folder + f"Fragments/Frag1/inklabels.png")
FRAG1_LABEL = torch.from_numpy(np.array(FRAG1_LABEL_PNG)).gt(0).float().to(device)
FRAG3_LABEL_PNG = Image.open(base_folder + f"Fragments/Frag3/inklabels.png")
FRAG3_LABEL = torch.from_numpy(np.array(FRAG3_LABEL_PNG)).gt(0).float().to(device)
FRAG4_LABEL_PNG = Image.open(base_folder + f"Fragments/Frag4/inklabels.png")
FRAG4_LABEL = torch.from_numpy(np.array(FRAG4_LABEL_PNG)).gt(0).float().to(device)

FRAG1_IR = torch.from_numpy(np.array(FRAG1_IR_PNG)).float().to(device)

# Get the label as an image
label_as_img = Image.open(base_folder + f"Fragments/Frag1/inklabels.png")
ir_as_img = Image.open(base_folder + f"Fragments/Frag1/irnc.png")
# Crop the rectangle from the original image
val_label = label_as_img.crop((1175, 3602, 1175+EVAL_WINDOW, 3602+EVAL_WINDOW))
ir_label = ir_as_img.crop((1175, 3602, 1175+EVAL_WINDOW, 3602+EVAL_WINDOW))
# Save the cropped image
val_label.save(base_folder + 'Fragments/Frag1/label.png')#"g:/VS_CODE/CV/Vesuvius Challenge/Fragments/Frag1/previews/label.png")
ir_label.save(base_folder + 'Fragments/Frag1/irn_label.png')#"g:/VS_CODE/CV/Vesuvius Challenge/Fragments/Frag1/previews/irn_label.png")


''' 
Cut out a small window for validation during training
'''
small_rect = (1175, 3602, EVAL_WINDOW-1, EVAL_WINDOW-1) #H and W bust be a multiple of 64
'''
'''

#LOAD THE DATA
training_points_1, validation_points_1 = extract_training_points(FRAG1_EDGES_LABEL, small_rect)
training_points_3, validation_points_3 = extract_training_points(FRAG1_EDGES_LABEL, small_rect)
training_points_4, validation_points_4 = extract_training_points(FRAG1_EDGES_LABEL, small_rect)

tpoints=[training_points_1,training_points_3,training_points_4]
vpoints=[validation_points_1,validation_points_3,validation_points_4]
fraglbl=[FRAG1_LABEL,FRAG3_LABEL,FRAG4_LABEL]
fragedg=[FRAG1_EDGES_LABEL, FRAG3_EDGES_LABEL,FRAG4_EDGES_LABEL]

train_ds = SubvolumeDataset(fragments, fraglbl, fragedg, tpoints,0)
valid_ds = SubvolumeDataset(fragments, fraglbl, fragedg, vpoints,0)
#train_ds = SubvolumeDataset(frag1_scan, FRAG1_LABEL, FRAG1_EDGES_LABEL, training_points_1,0)
#valid_ds = SubvolumeDataset(frag1_scan, FRAG1_LABEL, FRAG1_EDGES_LABEL, validation_points_1,0)



