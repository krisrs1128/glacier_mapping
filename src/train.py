#!/usr/bin/env python
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
from pathlib import Path
from src.data import GlacierDataset
from src.frame import Framework
from torch.utils.data import DataLoader
import addict
from torch.utils.tensorboard import SummaryWriter
import torch

path = "/scratch/sankarak/data/glaciers/processed/"
train_dataset = GlacierDataset(Path(path, "train"))
val_dataset = GlacierDataset(Path(path, "test"))

train_loader = DataLoader(train_dataset,batch_size=5, shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=15, shuffle=True, num_workers=3)

model_opts = addict.Dict({"name" : "Unet", "args" : {"inchannels": 3, "outchannels": 1, "net_depth": 2}})
optim_opts = addict.Dict({"name": "Adam", "args": {"lr": 1e-4}})
frame = Framework(model_opts=model_opts, optimizer_opts=optim_opts)

writer = SummaryWriter()

## Train Loop
epochs=10
for epoch in range(1, epochs):
    loss = 0
    for i, (x,y) in enumerate(train_loader):
        frame.set_input(x,y)
        loss += frame.optimize()
    print("epoch Loss:", loss / len(train_dataset))
    writer.add_scalar('Epoch Loss', loss/len(train_dataset), epoch)

    if epoch%5==0:
        frame.save(out_dir, epoch)

    ## validation loop
    loss = 0
    for x,y in val_loader:
        y_hat = frame.infer(x)
        loss += frame.loss(y, y_hat).item()
        writer.add_scalar('Batch Val Loss', loss)
    print("val Loss: ", loss / len(val_loader))