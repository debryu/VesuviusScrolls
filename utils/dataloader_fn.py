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

def normalize_training_points(list_of_points):
    for coord in list_of_points:
        # Randomize the coordinates
        x_offset = np.random.randint(0, 1.5*WINDOW) - int(1*WINDOW)
        y_offset = np.random.randint(0, 1.5*WINDOW) - int(1*WINDOW)
        if coord[0] > 129 and coord[0] < 7200:
            coord[0] += y_offset
        if coord[1] > 129:
            coord[1] += x_offset
    return list_of_points
