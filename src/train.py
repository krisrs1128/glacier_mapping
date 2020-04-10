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
from torch.utils.data import DataLoader, Subset
import addict
from torch.utils.tensorboard import SummaryWriter
import torch

path = "/scratch/sankarak/data/glaciers/processed/"

train_dataset = GlacierDataset(Path(path, "train"))
train_dataset = Subset(train_dataset, range(10))

val_dataset = GlacierDataset(Path(path, "test"))
val_dataset = Subset(val_dataset, range(10))


train_loader = DataLoader(train_dataset,batch_size=5, shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=15, shuffle=True, num_workers=3)

model_opts = addict.Dict({"name" : "Unet", "args" : {"inchannels": 3, "outchannels": 1, "net_depth": 2}})
optim_opts = addict.Dict({"name": "Adam", "args": {"lr": 1e-4}})
metrics_opts = addict.Dict({"precision": {"threshold": 0.2}, "IoU": {"threshold": 0.4}})
frame = Framework(model_opts=model_opts, optimizer_opts=optim_opts, metrics_opts=metrics_opts)



writer = SummaryWriter()

## Train Loop
epochs=10
for epoch in range(1, epochs):
    loss = 0
    metrics = []
    for i, (x,y) in enumerate(train_loader):
        frame.set_input(x,y)
        loss += frame.optimize()
        if i == 0:
            metrics=frame.calculate_metrics()
        else:
            metrics+=frame.calculate_metrics()

    print("Epoch metrics:", metrics/len(train_dataset))
    print("epoch Loss:", loss / len(train_dataset))
    writer.add_scalar('Epoch Loss', loss/len(train_dataset), epoch)
    for k, item in enumerate(metrics):
        writer.add_scalar('Epoch Metrics '+str(k), item/len(train_dataset), epoch)



    if epoch%5==0:
        frame.save(frame.out_dir, epoch)

    ## validation loop
    loss = 0
    for x,y in val_loader:
        y_hat = frame.infer(x.to(frame.device))
        loss += frame.loss(y.to(frame.device), y_hat).item()
        writer.add_scalar('Batch Val Loss', loss)
    print("val Loss: ", loss / len(val_loader))