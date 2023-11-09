import torch
from torch.nn import functional as F
from torchvision.transforms.functional import gaussian_blur


gaussian_kernel = torch.ones((2,1,6,6,6))
gaussian_kernel = gaussian_blur(gaussian_kernel, (3,3,3))
print(gaussian_kernel)

def dice_loss_weight(score,target,mask = gaussian_kernel):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target * mask)
    y_sum = torch.sum(target * target * mask)
    z_sum = torch.sum(score * score * mask)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def wce(logits,target,weights,batch_size=2,H=64,W=64,D=64):
  # Calculate log probabilities
  logp = F.log_softmax(logits,dim=1)
  # Gather log probabilities with respect to target
  logp = logp.gather(1, target.view(batch_size, 1, H, W,D))
  # Multiply with weights
  weighted_logp = (logp * weights).view(batch_size, -1)
  # Rescale so that loss is in approx. same interval
  #weighted_loss = weighted_logp.sum(1) / weights.view(batch_size, -1).sum(1)
  weighted_loss = (weighted_logp.sum(1) - 0.00001) / (weights.view(batch_size, -1).sum(1) + 0.00001)
  # Average over mini-batch
  weighted_loss = -1.0*weighted_loss.mean()
  return weighted_loss

