#  metrics and losses
import torch


def precision(pred, true, label=1):
    tp = ((pred == label) & (true == label)).sum().item()
    fp = ((pred == label) & (true != label)).sum().item()

    if tp == fp == 0:
        return 0

    return tp / (tp + fp)


def tp_fp_fn(pred, true, label=1):
    tp = ((pred == label) & (true == label)).sum().item()
    fp = ((pred == label) & (true != label)).sum().item()
    fn = ((pred != label) & (true == label)).sum().item()

    return tp, fp, fn


def recall(pred, true, label=1):
    tp = ((pred == label) & (true == label)).sum().item()
    fn = ((pred != label) & (true == label)).sum().item()

    try:
        return tp / (tp + fn)
    except:
        return 0


def pixel_acc(pred, true):
    return (pred == true).sum().item() / true.numel()


def dice(pred, true, label=1):
    tp = ((pred == label) & (true == label)).sum().item()
    fp = ((pred == label) & (true != label)).sum().item()
    fn = ((pred != label) & (true == label)).sum().item()

    return (2 * tp) / (2 * tp + fp + fn)


def IoU(pred, true, label=1):
    tp = ((pred == label) & (true == label)).sum().item()
    fp = ((pred == label) & (true != label)).sum().item()
    fn = ((pred != label) & (true == label)).sum().item()

    return tp / (tp + fp + fn)


class diceloss(torch.nn.Module):
    def __init__(self, act=torch.nn.Sigmoid(), smooth=0):
        super().__init__()
        self.act = act
        self.smooth = smooth

    def forward(self, pred, target):
        pred = self.act(pred)

        iflat = pred.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        A_sum = torch.sum(iflat * iflat)
        B_sum = torch.sum(tflat * tflat)
        union = A_sum + B_sum

        return 1 - ((2.0 * intersection + self.smooth) / (union + self.smooth))
