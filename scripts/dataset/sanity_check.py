import sys
sys.path.append('C:/Users/debryu/Desktop/VS_CODE/HOME/CV/Vesuvius Challenge/VesuviusScrolls')  # Add the parent directory to the Python path
import vesuvius_dataloader as dataloader
import matplotlib.pyplot as plt
import numpy as np

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
