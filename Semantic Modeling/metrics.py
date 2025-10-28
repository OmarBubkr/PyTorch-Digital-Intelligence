import torch.nn as nn
import torch.nn.functional as F
import torch

class BiDiceLoss(nn.Module):
  def __init__(self, smooth=1e-6):
      super().__init__()
      self.smooth = smooth

  def forward(self, pred, target):
      pred = torch.sigmoid(pred)
      intersection = (pred * target).flatten(1).sum(1)
      union = (pred + target).flatten(1).sum(1)
      coeff = (2 * intersection + self.smooth) / (union + self.smooth)

      return 1 - coeff.mean()


class MultiDiceLoss(nn.Module):
  def __init__(self, smooth=1e-6):
    super().__init__()
    self.smooth = smooth

  def forward(self, pred, target):
    pred = torch.nn.functional.softmax(pred, dim=1)
    num_classes = pred.size(1)
    dice_loss = 0

    for c in range(num_classes):
      pred_c = pred[:, c]
      target_c = target[:, c]

      intersection = (pred * target).sum(dim=(2, 3))
      union = (pred + target).sum(dim=(2, 3))
      coeff = (2 * intersection + self.smooth) / (union + self.smooth)

      dice_loss += (1 - coeff)

    return dice_loss.mean() / num_classes



class MultiDiceLoss3D(nn.Module):
    def __init__(self, smooth=1e-6):
      super().__init__()
      self.smooth = smooth

    def forward(self, pred, target):
      pred = torch.nn.functional.softmax(pred, dim=1)
      num_classes = pred.size(1)
      dice_loss = 0

      for c in range(num_classes):
        pred_c = pred[:, c]
        target_c = target[:, c]
        intersection = (pred_c * target_c).sum(dim=(2, 3, 4))
        union = pred_c.sum(dim=(2, 3, 4)) + target_c.sum(dim=(2, 3, 4))
        dice_loss += (2. * intersection + self.smooth) / (union + self.smooth)

      return 1 - dice_loss.mean() / num_classes


class PixelAcc(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, outputs, targets, batch_size):
    for idx in range(batch_size):
        output = outputs[idx]
        target = targets[idx]
        correct = torch.sum(torch.eq(output, target).long())
        self.acc += correct / np.prod(np.array(output.shape)) / batch_size

    return self.acc.item()



class IOU(nn.Module):
  def __init__(self):
    super().__init__()
    self.eps = 1e-6

  def forward(self, outputs, targets, batch_size, n_classes):
    for idx in range(batch_size):
        outputs_cpu = outputs[idx].cpu()
        targets_cpu = targets[idx].cpu()

        for c in range(n_classes):
            i_outputs = np.where(outputs_cpu == c)  # indices of 'c' in output
            i_targets = np.where(targets_cpu == c)  # indices of 'c' in target
            intersection = np.intersect1d(i_outputs, i_targets).size
            union = np.union1d(i_outputs, i_targets).size
            class_iou[c] += (intersection + self.eps) / (union + self.eps)

    class_iou /= batch_size

    return class_iou


"""def forward(self, outputs, targets, n_classes):
  iou = []
  for c in range(n_classes):
      pred_c = (outputs == c)
      target_c = (targets == c)
      intersection = (pred_c & target_c).sum().float()
      union = (pred_c | target_c).sum().float()
      iou.append((intersection + self.eps) / (union + self.eps))
  return torch.tensor(iou).mean()"""
