import torch
import numpy as np
import os
import PIL.Image as Image
from tqdm import tqdm
import matplotlib.pyplot as plt 

#Run once for each fragment
def create_3D_volume_from_tif(base_folder = "G:/VS_CODE/CV/Vesuvius Challenge/", scroll_number = 0, FROM = 0, HEIGHT = 64, split = False):
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  scroll_type = f"Fragments/Frag{scroll_number}"
  PREFIX = base_folder + scroll_type + "/surface_volume/"
  output_folder = base_folder + "Fragments_dataset/3d_surface/"
  files = [os.path.join(PREFIX, filename) for filename in os.listdir(PREFIX) if filename.endswith('.tif')]
  images = []
  for filename in tqdm(files[FROM:FROM+HEIGHT]):
      image = np.array(Image.open(filename), dtype=np.float32)/65535.0
      y_size = image.shape[0]
      x_size = image.shape[1]
      if split:
        image = image[:y_size//2,:]
      images.append(image)
  
  image_stack = torch.stack([torch.from_numpy(image) for image in images], dim=0)
  print("Saving the image stack...")
  # Save the image stack
  torch.save(image_stack, output_folder + f"scan_{scroll_number}_{FROM}d{HEIGHT}.pt")

  if split:
     for filename in tqdm(files[FROM:FROM+HEIGHT]):
        image = np.array(Image.open(filename), dtype=np.float32)/65535.0
        y_size = image.shape[0]
        x_size = image.shape[1]
        image = image[y_size//2:,:]
        images.append(image)
        torch.save(image_stack, output_folder + f"partial_scan_{scroll_number}_{FROM}d{HEIGHT}.pt")
  print("Done!")


def save_all_layers_to_np(in_folder, output_folder = "G:/VS_CODE/CV/Vesuvius Challenge/Fragments_dataset/numpy_layers/"):
  files = [os.path.join(in_folder, filename) for filename in os.listdir(in_folder) if filename.endswith('.tif')]
  for i,filename in enumerate(tqdm(files[0:64])):
      image = np.array(Image.open(filename), dtype=np.float32)/65535.0
      np.save(output_folder + f'layer_{i}.numpy', image)
  return image.shape

def create_numpy_dataset_from_tif(scroll_number,tifs_folder = "G:/VS_CODE/CV/Vesuvius Challenge/Fragments/Frag2/surface_volume/", output_folder = "G:/VS_CODE/CV/Vesuvius Challenge/Fragments_dataset/numpy/", numpy_layers_folder = "G:/VS_CODE/CV/Vesuvius Challenge/Fragments_dataset/numpy_layers/"):
  size = save_all_layers_to_np(tifs_folder)
  files = [os.path.join(numpy_layers_folder, filename) for filename in os.listdir(numpy_layers_folder) if filename.endswith('.npy')]
  images = []
  for i,filename in enumerate(tqdm(files)):
    image = np.memmap(os.path.join(numpy_layers_folder, filename), dtype=np.float32, mode='r', shape=size)
    images.append(image)
  image_stack = np.stack(images, axis=0)
  shp = np.shape(image_stack)
  np.save(output_folder + f'F{scroll_number}_{shp[0]}-{shp[1]}-{shp[2]}', image_stack)

def create_numpy_dataset_from_layers(scroll_number,numpy_layers_folder = "G:/VS_CODE/CV/Vesuvius Challenge/Fragments_dataset/numpy_layers/",output_folder = "G:/VS_CODE/CV/Vesuvius Challenge/Fragments_dataset/numpy/"):
  files = [os.path.join(numpy_layers_folder, filename) for filename in os.listdir(numpy_layers_folder) if filename.endswith('.npy')]
  size = (14830,9506)
  images = []
  for i,filename in enumerate(tqdm(files)):
    image = np.memmap(os.path.join(numpy_layers_folder, filename), dtype=np.float32, mode='r', shape=size)
    images.append(image)
  image_stack = np.stack(images, axis=0)
  shp = np.shape(image_stack)
  np.save(output_folder + f'F{scroll_number}_{shp[0]}-{shp[1]}-{shp[2]}', image_stack)


#Run once for each fragment
def create_numpy_dataset_from_3D_volume(base_folder = "G:/VS_CODE/CV/Vesuvius Challenge/",scroll_number = 1):
  i = scroll_number
  output_folder = base_folder + "Fragments_dataset/3d_surface/"
  frag = torch.load(output_folder + f"scan_{i}_0d64.pt").to('cpu')
  frag = frag.detach().numpy()
  shp = np.shape(frag)
  np.save(base_folder + f'numpy/F{i}_{shp[0]}-{shp[1]}-{shp[2]}', frag)


def split_numpy_layers(numpy_layers_folder = "G:/VS_CODE/CV/Vesuvius Challenge/Fragments_dataset/numpy_layers/", output_folder = "G:/VS_CODE/CV/Vesuvius Challenge/Fragments_dataset/numpy/"):
  files = [os.path.join(numpy_layers_folder, filename) for filename in os.listdir(numpy_layers_folder) if filename.endswith('.npy')]
  images = []
  size = (14830,9506)
  for i,filename in enumerate(tqdm(files)):
    image = np.memmap(os.path.join(numpy_layers_folder, filename), dtype=np.float32, mode='r', shape=size)
    images.append(image[(size[0]//2)-65:,:])
  image_stack = np.stack(images, axis=0)
  shp = np.shape(image_stack)
  np.save(output_folder + f'F2part1_{shp[0]}-{shp[1]}-{shp[2]}', image_stack)
  
def split_label_image(filename = "mask.png"):
  size = (14830,9506)
  FRAG2_LABEL_PNG = Image.open("G:/VS_CODE/CV/Vesuvius Challenge/Fragments/Frag2/" +  filename)
  plt.imshow(FRAG2_LABEL_PNG)
  plt.show()
  print(FRAG2_LABEL_PNG)
  FRAG2_LABEL = torch.from_numpy(np.array(FRAG2_LABEL_PNG))
  print(FRAG2_LABEL)
  FRAG2_LABEL1 = FRAG2_LABEL[:(size[0]//2)+65,:]*255
  FRAG2_LABEL2 = FRAG2_LABEL[(size[0]//2)-65:,:]*255
  
  # Save it as PNG image
  FRAG2_LABEL_PNG1 = Image.fromarray(FRAG2_LABEL1.numpy())
  FRAG2_LABEL_PNG2 = Image.fromarray(FRAG2_LABEL2.numpy())
  FRAG2_LABEL_PNG1.save("G:/VS_CODE/CV/Vesuvius Challenge/Fragments/Frag2/" + f"part1_{filename}")
  FRAG2_LABEL_PNG2.save("G:/VS_CODE/CV/Vesuvius Challenge/Fragments/Frag2/" + f"part2_{filename}")

split_numpy_layers()
#create_3D_volume_from_tif(scroll_number = 2)
#create_numpy_dataset_from_3D_volume(scroll_number = 2)