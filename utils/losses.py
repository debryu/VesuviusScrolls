import torch
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp


dice_loss = smp.losses.DiceLoss(mode='binary')
softBCE = smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.25)
loss_func = lambda x,y:0.5 * dice_loss(x,y)+0.5*softBCE(x,y)


'''

WINDOW_SIZE = 64
SIGMA = 64

def gaussian_kernel(size = 64, center = 31.5, sigma = 1):
        x, y = np.meshgrid(np.arange(size), np.arange(size))
        d = (x - center[0])**2 + (y - center[1])**2
        kernel = np.exp(-0.5 * (d / sigma**2))
        return kernel / np.sum(kernel)  # Normalize the kernel


# Apply a gaussian filter to the output
center = ((WINDOW_SIZE-1)/2,(WINDOW_SIZE-1)/2)
weight = gaussian_kernel(WINDOW_SIZE,center, sigma = SIGMA)
weight = torch.tensor(weight, dtype=torch.float64)
# Just to visualize that is working

wei = self.gaussian_kernel(4,(1.5,1.5),1.5)
wei = torch.tensor(wei).to(self.device)
tensor = torch.tensor(np.ones((4, 4))).to(self.device)
print(tensor*wei)


def dice_loss_weight(score,target,mask = weight):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target * mask)
    y_sum = torch.sum(target * target * mask)
    z_sum = torch.sum(score * score * mask)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

def dice_loss_weight_noMask(score,target):
      target = target.float()
      smooth = 1e-5
      intersect = torch.sum(score * target)
      y_sum = torch.sum(target * target)
      z_sum = torch.sum(score * score)
      loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
      loss = 1 - loss
      return loss


class dice_loss:
  def __init__(self, **kwargs):
      self.__dict__.update(kwargs)
      self.gaussian_kernel = weight
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  

  def dice_loss_weight(self, score,target, mask = None):
      if mask is None:
        mask = self.gaussian_kernel

      # Print the mask for sanity check
      #plt.imshow(mask.numpy())
      #plt.show()

      target = target.float()
      smooth = 1e-5
      mask = mask.to(self.device)
      intersect = torch.sum(score * target * mask)
      y_sum = torch.sum(target * target * mask)
      z_sum = torch.sum(score * score * mask)
      loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
      loss = 1 - loss
      return loss
  
  def dice_loss_weight_noMask(self, score,target):
      if mask is None:
        mask = self.gaussian_kernel

      # Print the mask for sanity check
      #plt.imshow(mask.numpy())
      #plt.show()

      target = target.float()
      smooth = 1e-5
      mask = mask.to(self.device)
      intersect = torch.sum(score * target)
      y_sum = torch.sum(target * target)
      z_sum = torch.sum(score * score)
      loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
      loss = 1 - loss
      return loss
'''