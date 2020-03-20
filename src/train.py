"""
Training/Eval Pipeline:
    1- Initialize loaders (train & validation)
        1.1-Pass all params onto both loaders
    2- Initialize the framework
    3- Train Loop 10 epochs
        3.1 Pass entire data loader through epoch
        3.2 Iterate over dataloader with specific batch
    4- Log Epoch level train acc, test acc, train loss, test loss.
    5- Save checkpoints after 5 epochs

"""
import torch
from torch.utils.data import DataLoader
import data
from frame import Framework
from pathlib import Path

path = "/scratch/akera/processed_glacier_dataset/"
train_dataset = data.GlacierDataset(Path(path, "train"))
val_dataset = data.GlacierDataset(Path(path, "val"))

train_loader = DataLoader(train_dataset,batch_size=8, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset,batch_size=8, shuffle=True, num_workers=4)

frame = Framework()

## Train Loop
epochs=10
for epoch in range(1, epochs):
    loss = 0
    for x,y in train_loader:
        frame.set_input(x,y)
        loss+=frame.optimize()
    print(train_epoch_loss/len(train_dataset))
    if epoch%5==0:
        frame.save(out_dir, epoch)

    ## validation loop
    for x,y in val_loader:
        y_hat = frame.infer(x)
        loss+=frame.loss(y, y_hat)

