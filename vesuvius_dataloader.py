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
base_folder = 'G:/VS_CODE/CV/Vesuvius Challenge/' #"G:/VS_CODE/CV/Vesuvius Challenge/"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
datasets = ["Normal", "Infrared"]

###############################################################################
#Get the size of each fragment and get the fragments from storage
names = os.listdir(base_folder+'Fragments_dataset/numpy/')
dataset_names = {}
fragments=[]
for i,n in enumerate(names):
    dataset_names[i] = n
    size = tuple(map(int, n.split('_')[1].split('.')[0].split('-')))
    temp = np.memmap(base_folder+'Fragments_dataset/numpy/'+n, dtype=np.float32, mode='r', shape=size)
    fragments.append(temp)
###############################################################################

class SubvolumeDataset(data.Dataset):
    def __init__(self, surfaces, labels, class_labels, coordinates,task = 0):
        self.surfaces = surfaces
        self.labels = labels
        self.class_labels = class_labels
        self.coordinates = coordinates
        self.task = task
    def __len__(self):
        #print(self.coordinates)
        return len(self.coordinates)
    def __getitem__(self, index):
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
        subvolume = torch.tensor(subvolume)
        subvolume = subvolume.view(1, SCAN_DEPTH, WINDOW*2+ODD_WINDOW, WINDOW*2+ODD_WINDOW)
        
        inklabel = label[y, x].view(1)
        inkpatch = label[y-WINDOW:y+WINDOW+ODD_WINDOW, x-WINDOW:x+WINDOW+ODD_WINDOW]
       
        #print(inkpatch)
        #print(current_task)
        #plt.imshow(inkpatch.cpu().numpy())
        #plt.show()
        
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
        return subvolume, inkpatch, self.task, scroll_id
        #return subvolume, inklabel, self.task


'''

'''
def extract_random_points(FRAG_MASK, num_points):
    not_border = np.zeros(FRAG_MASK.shape, dtype=bool)
    not_border[WINDOW:FRAG_MASK.shape[0]-WINDOW, WINDOW:FRAG_MASK.shape[1]-WINDOW] = True
    arr_mask = np.array(FRAG_MASK) * not_border
    arr_mask = np.argwhere(arr_mask)
    random_points = np.random.choice(arr_mask.shape[0], num_points, replace=False)
    return arr_mask[random_points]


def extract_test_points(FRAG_MASK):
    not_border = np.zeros(FRAG_MASK.shape, dtype=bool)
    not_border[WINDOW:FRAG_MASK.shape[0]-WINDOW, WINDOW:FRAG_MASK.shape[1]-WINDOW] = True
    arr_mask = np.array(FRAG_MASK) * not_border
    test = np.ones(FRAG_MASK.shape, dtype=bool) * arr_mask
    test = np.argwhere(test)
    return test

def extract_training_and_val_points(FRAG_MASK, validation_rect= (1175,3602,63,63)):
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

def extract_training_points(FRAG_MASK):
    not_border = np.zeros(FRAG_MASK.shape, dtype=bool)
    not_border[WINDOW:FRAG_MASK.shape[0]-WINDOW, WINDOW:FRAG_MASK.shape[1]-WINDOW] = True
    arr_mask = np.array(FRAG_MASK) * not_border
    train = np.ones(FRAG_MASK.shape, dtype=bool) * arr_mask
    train = np.argwhere(train)
    return train


def extract_render_points(pixels, original_width, original_height, stride_H = 64, stride_W = 64):
    W = original_width
    H = original_height
    pixels_to_render = []
    
    points_per_row = int(W/stride_W)
    points_per_col = int(H/stride_H)
    
    offset = (stride_H/2 - 1)*W + stride_W/2
    for i in range(0, points_per_col):
        for j in range(0, points_per_row):
            #print(f'i:{i}/122 - j:{j}/82  -- Row: {pix_count_row}, Col: {pix_count_col}')
            index = j*stride_W + stride_W*(i)*W
            pixels_to_render.append(pixels[index + int(offset)])
            #print(len(pixels_to_render))
    
    return pixels_to_render

def old_extract_render_points(pixels, original_width, original_height, FOV = 64):
    W = original_width
    H = original_height
    pixels_to_render = []
    points_per_row = int(W/FOV)
    points_per_col = int(H/FOV)
    pix_count_row = 32
    pix_count_col = 31
    #print("start")
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
    kernel = np.ones((1, 1), np.uint8)  # You can adjust the size of the kernel
    # Dilate the edges to increase the thickness
    dilated_edges = cv.dilate(edges, kernel, iterations=1)
    edge_mask = np.array(torch.from_numpy(np.array(dilated_edges)).gt(155).float().to('cpu'))
    edge_mask_pt = torch.from_numpy(np.array(dilated_edges)).gt(155).float().to(device)
    #plt.imshow(edge_mask)
    #plt.show()
    return edge_mask, edge_mask_pt



'''
DEFINE MANUALLY ALL THE LABEL FILES

'''

FRAG1_MASK = np.array(Image.open( base_folder + f"Fragments/Frag1/mask.png").convert('1'))
FRAG2_MASK1 = np.array(Image.open( base_folder + f"Fragments/Frag2/part1_mask.png").convert('1'))
FRAG2_MASK2 = np.array(Image.open( base_folder + f"Fragments/Frag2/part2_mask.png").convert('1'))
FRAG3_MASK = np.array(Image.open( base_folder + f"Fragments/Frag3/mask.png").convert('1'))
FRAG4_MASK = np.array(Image.open( base_folder + f"Fragments/Frag4/mask.png").convert('1'))
FRAG1_LABEL_PNG = Image.open(base_folder + f"Fragments/Frag1/inklabels.png")
FRAG2_LABEL_PNGa = Image.open(base_folder + f"Fragments/Frag2/part1_inklabels.png")
FRAG2_LABEL_PNGb = Image.open(base_folder + f"Fragments/Frag2/part2_inklabels.png")
FRAG3_LABEL_PNG = Image.open(base_folder + f"Fragments/Frag3/inklabels.png")
FRAG4_LABEL_PNG = Image.open(base_folder + f"Fragments/Frag4/inklabels.png")
FRAG1_LABEL = torch.from_numpy(np.array(FRAG1_LABEL_PNG))
FRAG2_LABELa = torch.from_numpy(np.array(FRAG2_LABEL_PNGa))
FRAG2_LABELb = torch.from_numpy(np.array(FRAG2_LABEL_PNGb))
FRAG3_LABEL = torch.from_numpy(np.array(FRAG3_LABEL_PNG))
FRAG4_LABEL = torch.from_numpy(np.array(FRAG4_LABEL_PNG))
FRAG1_EDGES_LABEL, FRAG1_EDGES_LABEL_PT = create_edge_mask(base_folder + f"Fragments/Frag1/inklabels.png")
FRAG2_EDGES_LABELa, FRAG2_EDGES_LABEL_PTa = create_edge_mask(base_folder + f"Fragments/Frag2/part1_inklabels.png")
FRAG2_EDGES_LABELb, FRAG2_EDGES_LABEL_PTb = create_edge_mask(base_folder + f"Fragments/Frag2/part2_inklabels.png")
FRAG3_EDGES_LABEL, FRAG3_EDGES_LABEL_PT = create_edge_mask(base_folder + f"Fragments/Frag3/inklabels.png")
FRAG4_EDGES_LABEL, FRAG4_EDGES_LABEL_PT = create_edge_mask(base_folder + f"Fragments/Frag4/inklabels.png")
'''
-------------------------------------------------------------------------------------------------------------------
'''
#FRAG1_LABEL = torch.from_numpy(np.array(FRAG1_LABEL_PNG)).gt(0).float().to(device)
#plt.imshow(FRAG1_LABEL.cpu().numpy())
#plt.show()

# Get the label as an image
label_as_img = Image.open(base_folder + f"Fragments/Frag1/inklabels.png")

# Crop the rectangle from the original image
val_label = label_as_img.crop((1175, 3602, 1175+EVAL_WINDOW, 3602+EVAL_WINDOW))

# Save the cropped image
val_label.save(base_folder + 'Fragments/Frag1/label.png')#"g:/VS_CODE/CV/Vesuvius Challenge/Fragments/Frag1/previews/label.png")



''' 
Cut out a small window for validation during training
'''
small_rect = (1175, 3602, EVAL_WINDOW-1, EVAL_WINDOW-1) #H and W bust be a multiple of 64

'''
Generate all the training coordinates (points) for each segment
'''
training_points_1, validation_points_1 = extract_training_and_val_points(FRAG1_EDGES_LABEL, small_rect)
training_points_2a = extract_training_points(FRAG2_EDGES_LABELa)
training_points_2b = extract_training_points(FRAG2_EDGES_LABELb)
training_points_3 = extract_training_points(FRAG3_EDGES_LABEL)
training_points_4 = extract_training_points(FRAG4_EDGES_LABEL)
random_points_2a = extract_random_points(FRAG2_EDGES_LABELa, 100)
random_points_3 = extract_random_points(FRAG3_EDGES_LABEL, 363)
random_points_4 = extract_random_points(FRAG4_EDGES_LABEL, 100)
#563

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

# Concatenate coordinates
all_coordinates = np.concatenate([training_points_1[0::64], 
                                  training_points_2a[0::64], 
                                  training_points_2b[0::64], 
                                  training_points_3[0::64], 
                                  training_points_4[0::64], 
                                  hpp_F1_BLACK, 
                                  hpp_F1_WHITE,
                                  hpp_F2a_BLACK,
                                  random_points_2a,
                                  random_points_3,
                                  random_points_4,
                                  ])
# Create an array for class labels
class_labels = np.array(  [0] * len(training_points_1[0::64]) 
                        + [1] * len(training_points_2a[0::64]) 
                        + [2] * len(training_points_2b[0::64]) 
                        + [3] * len(training_points_3[0::64]) 
                        + [4] * len(training_points_4[0::64])
                        + [0] * len(hpp_F1_BLACK) 
                        + [0] * len(hpp_F1_WHITE)    
                        + [1] * len(hpp_F2a_BLACK)  
                        + [1] * len(random_points_2a)
                        + [3] * len(random_points_3)   
                        + [4] * len(random_points_4)                                                                                                                                                                                                                                                
                       )
# Store all the labels
all_labels = [FRAG1_LABEL, FRAG2_LABELa, FRAG2_LABELb, FRAG3_LABEL, FRAG4_LABEL]


'''
# Only frag1
all_coordinates = np.concatenate([training_points_3[0::64]])
class_labels = np.array([0] * len(training_points_3[0::64]))
all_labels = [FRAG3_LABEL]
'''

train_ds = SubvolumeDataset(fragments, all_labels, class_labels, all_coordinates)
total_train_iters = len(all_coordinates)
#print(validation_points_1)
validation_frag1 = SubvolumeDataset(fragments, all_labels, np.array([0] * len(validation_points_1)), validation_points_1,0)
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