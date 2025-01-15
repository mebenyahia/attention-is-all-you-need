import torch
import os
from torch.utils.tensorboard import SummaryWriter

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        print("=> Saving a new best")
        torch.save(state, filename)  # save checkpoint
    else:
        print("=> Validation Accuracy did not improve")

def load_checkpoint(model, optimizer, filename='checkpoint.pth.tar'):
    """Load checkpoint if no exceptions"""
    if os.path.isfile(filename):
        print("=> Loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> Loaded checkpoint '{}' (epoch {})"
              .format(filename, checkpoint['epoch']))
        return model, optimizer, checkpoint['epoch']
    else:
        print("=> No checkpoint found at '{}'".format(filename))
        return model, optimizer, 0

def create_optimizer(model, lr=0.01, weight_decay=0.0001):
    """Setup the optimizer"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optimizer

def setup_tensorboard(log_dir='runs/'):
    """Initialize TensorBoard writer"""
    writer = SummaryWriter(log_dir=log_dir)
    return writer
