import sys
sys.path.append('C:/Users/debryu/Desktop/VS_CODE/HOME/CV/Vesuvius Challenge/VesuviusScrolls')  # Add the parent directory to the Python path
import vesuvius_dataloader as dataloader
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import torch.utils.data as data
import math

train_ds = dataloader.train_ds
n_samples = len(train_ds)
batch_size = 7
iters = math.floor(n_samples/batch_size)
print(n_samples)
#LOAD DATA
train_dl = data.DataLoader(train_ds, batch_size = batch_size, shuffle=False)


total_pixels = 0
white_pixels = 0    
frequencies = []
mean_pic = torch.zeros((64,64))
for i,batch in enumerate(tqdm(train_dl)):
    _, labels, _, id = batch
    
    #print(id1,id2)
    ttp = 64*64*labels.shape[0]
    total_pixels += ttp
    twp = torch.sum(labels)
    white_pixels += twp.item()
    mean_pic = torch.add(mean_pic,torch.sum(labels,dim=0))
    
    #print(dataloader.dataset['object'][id1])
    #print(dataloader.dataset['object'][id2])
    #print(torch.max(lab1),torch.max(lab2))
    frequencies.append(twp/ttp)
    if(i == len(train_dl)-2):
        break

# Plot an histogram of the frequencies
plt.hist(frequencies, bins = 10)
plt.show()

mean_pic = mean_pic/n_samples
plt.imshow(mean_pic.cpu().detach().numpy())
plt.show()

#WITH 64 we get 46%s
print("Total pixels: ",total_pixels)
print("White pixels: ",white_pixels)
print("Percentage: ",100*white_pixels/total_pixels, "%")


#SHOW THEM IN A GRID

'''
for batch in dataloader.train_loader:
    chunks, labels, tasks = batch
    # Plotting side by side
    plt.figure(figsize=(10, 4))
    lab1,lab2 = labels
    # Plotting Chunks
    plt.subplot(1, 2, 1)
    plt.imshow(np.array(lab1*255))
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Label Image 1')

    # Plotting Labels
    plt.subplot(1, 2, 2)
    plt.imshow(np.array(lab2*255))
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Label Image 2')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plot
    plt.show()
'''