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
from pathlib import Path
import math

path = "/scratch/sankarak/data/glaciers/"

train_dataset = GlacierDataset(Path(path, "processed/train"))
# train_dataset = Subset(train_dataset, range(5))

val_dataset = GlacierDataset(Path(path, "processed/test"))
# val_dataset = Subset(val_dataset, range(10))

train_batch_size = 10
val_batch_size = 15

train_loader = DataLoader(train_dataset,batch_size=train_batch_size, shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True, num_workers=3)

model_opts = addict.Dict({"name" : "Unet", "args" : {"inchannels": 12, "outchannels": 1, "net_depth": 4}})
optim_opts = addict.Dict({"name": "Adam", "args": {"lr": 1e-5}})
metrics_opts = addict.Dict({"precision": {"threshold": 0.3}, "IoU": {"threshold": 0.3}})

frame = Framework(model_opts=model_opts, optimizer_opts=optim_opts, metrics_opts=metrics_opts, out_dir=Path(path, "models"))

writer = SummaryWriter()

## Train Loop
epochs=1000

for epoch in range(1, epochs):
    loss = 0
    for i, (x,y) in enumerate(train_loader):
        frame.set_input(x,y)
        _loss = frame.optimize()
        print(f"Epoch {epoch}, Training batch {i+1} of {int(math.ceil(len(train_dataset)/train_batch_size))}, Loss= {_loss/train_batch_size}", end="\r", flush=True)
        loss += _loss
        if i == 0:
            metrics=frame.calculate_metrics()
        else:
            metrics+=frame.calculate_metrics()

    print("\t")
    if epoch % 50 == 0:
        print(f"\nLoss: {loss/len(train_dataset)}")
        print(f"Metrics: {metrics/len(train_dataset)}")

    writer.add_scalar('Epoch Loss', loss/len(train_dataset), epoch)

    for k, item in enumerate(metrics):
        writer.add_scalar('Epoch Metrics '+str(k), item/len(train_dataset), epoch)

    ## validation loop
    loss = 0
    for i, (x,y) in enumerate(val_loader):
        y_hat = frame.infer(x.to(frame.device))
        _loss = frame.calc_loss(y_hat,y.to(frame.device)).item()
        print(f"Validating batch {i+1} of {int(math.ceil(len(val_dataset)/val_batch_size))}, Loss= {_loss/val_batch_size}", end="\r", flush=True)
        loss += _loss
        if i == 0:
            metrics=frame.calculate_metrics()
        else:
            metrics+=frame.calculate_metrics()

    print("\n")
    if epoch % 50==0:
        frame.save(frame.out_dir, epoch)
        writer.add_scalar('Batch Val Loss', loss/len(val_dataset), epoch)
        print("\nVal Loss: ", loss / len(val_dataset))
        print("Val Metrics: ", metrics/len(val_dataset),"\n\n")

    for k, item in enumerate(metrics):
        writer.add_scalar('Val Epoch Metrics '+str(k), item/len(val_dataset), epoch)