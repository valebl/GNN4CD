import torch


#-----------------------------------------------------
#---------------------- METRICS ----------------------
#-----------------------------------------------------


class AverageMeter(object):
    '''
    A generic class to keep track of performance metrics during training or testing of models
    (adapted from the Deep Learning tutorials of DSSC)
    '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy_binary_one(prediction, target, reduction="mean"):
    prediction_class = torch.where(prediction > 0.0, 1.0, 0.0)
    correct_items = (prediction_class == target).float()
    if reduction is None:
        return correct_items
    elif reduction == "mean":
        return torch.mean(correct_items)


def accuracy_binary_one_classes(prediction, target, reduction="mean"):
    prediction_class = torch.where(prediction > 0.0, 1.0, 0.0)
    correct_items = prediction_class == target
    correct_items_class0 = correct_items[target==0.0]
    correct_items_class1 = correct_items[target==1.0]
    if reduction is None:
        return correct_items_class0, correct_items_class1
    elif reduction == "mean":
        if correct_items_class0.shape[0] > 0:
            acc_class0 = correct_items_class0.sum() / correct_items_class0.shape[0]
        else:
            acc_class0 = torch.tensor(torch.nan)
        if correct_items_class1.shape[0] > 0:
            acc_class1 = correct_items_class1.sum() / correct_items_class1.shape[0]
        else:
            acc_class1 = torch.tensor(torch.nan)
        return acc_class0, acc_class1
