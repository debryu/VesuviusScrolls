import sys
sys.path.append('C:/Users/debryu/Desktop/VS_CODE/HOME/CV/Vesuvius Challenge/VesuviusScrolls')  # Add the parent directory to the Python path
import vesuvius_dataloader as dataloader
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import torch.utils.data as data

train_ds = dataloader.train_ds
n_samples = len(train_ds)
print(n_samples)
#LOAD DATA
train_dl = data.DataLoader(train_ds, batch_size = 2, shuffle=True)


total_pixels = 0
white_pixels = 0    
frequencies = []
mean_pic = torch.zeros((64,64))
for i,batch in enumerate(tqdm(train_dl)):
    chunks, labels, tasks, id = batch
    lab1,lab2 = labels
    id1,id2 = id
    id1 = int(id1)
    id2 = int(id2)
    #print(id1,id2)
    total_pixels += 64*64*2
    white_pixels += torch.sum(lab1) + torch.sum(lab2)
    mean_pic += lab1 + lab2
    
    #print(dataloader.dataset['object'][id1])
    #print(dataloader.dataset['object'][id2])
    #print(torch.max(lab1),torch.max(lab2))
    frequencies.append(((torch.sum(lab1) + torch.sum(lab2))/(64*64*2)).item())
    if(i > 5000):
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