import matplotlib.pyplot as plt
import numpy as np
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.animation import FuncAnimation
from matplotlib import cm


def plot3D(chunk,name,path="G:/VS_CODE/CV/Vesuvius Challenge/Fragments/Frag1/plots/"):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    chunk = chunk.reshape(64,64,64)
    if chunk.device.type == "cuda":
        volume_data = chunk.cpu().detach().numpy()
    else:
        volume_data = chunk.detach().numpy()
        
    size=chunk.shape[0]
    X, Y= np.meshgrid(np.arange(size), np.arange(size))
    
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_zlim(0, size)
    
    def update(frame):
        #ax.cla()  # Clear the previous plot
        ax.set_xlim(0, size)
        ax.set_ylim(0, size)
        ax.set_zlim(0, size)
        Z = np.ones((size,size))*frame
        # Draw a circle on the x=0 'wall'
        #p = Circle((32, 32), 3)
        #ax.add_patch(p)
        #art3d.pathpatch_2d_to_3d(p, z=frame, zdir="z")
        #ax.plot(volume_data[frame,:,0], volume_data[frame,0,:],frame, zdir="z" )
        ax.plot_surface(X, Y, Z, facecolors=cm.viridis(volume_data[frame,:,:]), rstride=5, cstride=5, antialiased=True)
    
    
    # Number of frames (Z-axis slices)
    num_frames = size
    ani = FuncAnimation(fig, update, frames=num_frames, interval=100, repeat=False)
    ani.save(path+name+'.gif', writer='imagemagick')
    plt.show()

def plot2D(chunk,name,path="G:/VS_CODE/CV/Vesuvius Challenge/Fragments/Frag1/plots/"):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    chunk = chunk.reshape(64,64)
    if chunk.device.type == "cuda":
        volume_data = chunk.cpu().detach().numpy()#*255
    else:
        volume_data = chunk.detach().numpy()#*255
        
    size=chunk.shape[0]
    X, Y= np.meshgrid(np.arange(size), np.arange(size))
    
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_zlim(0, size)
    
    def update(frame):
        #ax.cla()  # Clear the previous plot
        ax.set_xlim(0, size)
        ax.set_ylim(0, size)
        ax.set_zlim(0, size)
        Z = np.ones((size,size))*1
        # Draw a circle on the x=0 'wall'
        #p = Circle((32, 32), 3)
        #ax.add_patch(p)
        #art3d.pathpatch_2d_to_3d(p, z=frame, zdir="z")
        #ax.plot(volume_data[frame,:,0], volume_data[frame,0,:],frame, zdir="z" )
        ax.plot_surface(X, Y, Z, facecolors=cm.viridis(volume_data[:,:]), rstride=5, cstride=5, antialiased=True)
    
    
    # Number of frames (Z-axis slices)
    num_frames = 1
    ani = FuncAnimation(fig, update, frames=num_frames, interval=1, repeat=False)
    ani.save(path+name+'.gif', writer='imagemagick')
    plt.show()