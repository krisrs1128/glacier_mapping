#  metrics


def precision(pred, true):
    tp = ((pred == 1) & (pred == true)).sum().item()
    fp = ((pred == 1) & (pred != true)).sum().item()

    return tp / (tp + fp)


def recall(pred, true):
    tp = ((pred == 1) & (pred == true)).sum().item()
    fn = ((pred == 0) & (pred != true)).sum().item()

    return tp / (tp + fn)


def pixel_acc(pred, true):
    return (pred == true).sum().item() / true.numel()


def dice(pred, true):
    tp = ((pred == 1) & (pred == true)).sum().item()
    fp = ((pred == 1) & (pred != true)).sum().item()
    fn = ((pred == 0) & (pred != true)).sum().item()

    return (2 * tp) / (2 * tp + fp + fn)


def IoU(pred, true):
    tp = ((pred == 1) & (pred == true)).sum().item()
    fp = ((pred == 1) & (pred != true)).sum().item()
    fn = ((pred == 0) & (pred != true)).sum().item()

    return tp / (tp + fp + fn)
