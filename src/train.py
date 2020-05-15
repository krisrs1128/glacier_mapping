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
<<<<<<< Updated upstream

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
    for i, (x,y) in enumerate(val_loader):
        y_hat = frame.infer(x.to(frame.device))
        loss += frame.loss(y_hat,y.to(frame.device)).item()
=======
import math 
import json
import yaml
import numpy as np

np.random.seed(7) 

def unnormalize(x, conf, channels=(2,1,0)):
    '''
    Given normalized input, gives RGB image to show in tensorboard
    Input:
        x is tensor of shape B * H * W * C
        conf is path to stats.json
    Output:
        returns unnormalized tensor B * H * W * C 
    '''
    j = json.load(open(conf))
    mean = j['means']
    std = j['stds']
    for i,channel in enumerate(channels):
        x[:,:,:,i] = x[:,:,:,i]*std[channel]
        x[:,:,:,i] += mean[channel]
    return x

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--name', type=str, help='Name of run', dest='run_name', required=True)
    parser.add_argument('-e', '--epochs', type=int, default=500, help='Number of epochs (Default 500)', dest='epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=16, help='Batch size (Default 16)', dest='batch_size')
    parser.add_argument('-s', '--save_every', type=int, default=1, help='Save every n epoch (Default 5)', dest='save_every')
    parser.add_argument('-p', '--path', type=str, default='./data/glaciers_hkh/', help='Root path', dest='path')
    parser.add_argument('-c', '--conf', type=str, default='./conf/train_conf.yaml', help='Configuration File for training', dest='conf')

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()   
    conf = Dict(yaml.safe_load(open(args.conf, "r")))

    filter_channels = (0, 1, 2)
    # filter_channels = np.array(range(12))

    train_dataset = GlacierDataset(Path(args.path, "processed/train"))
    val_dataset = GlacierDataset(Path(args.path, "processed/test"))

    train_loader = DataLoader(train_dataset,batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=3)

    frame = Framework(model_opts=conf.model_opts, optimizer_opts=conf.optim_opts, metrics_opts=conf.metrics_opts, reg_opts=conf.reg_opts, out_dir=f"{args.path}models/{args.run_name}")

    # Tensorboard path
    writer = SummaryWriter(f"{args.path}/runs/{args.run_name}")
    # Write input parameters
    writer.add_text("Arguments", json.dumps(vars(args)))
    writer.add_text("Configuration Parameters", json.dumps(conf))

    # Prepare image grid train/val, x,y to write in tensorboard
    _sample_train_images, _sample_train_labels = iter(train_loader).next()
    _sample_train_images = _sample_train_images[:,:,:,filter_channels]
    _sample_val_images, _sample_val_labels = iter(val_loader).next()
    _sample_val_images = _sample_val_images[:,:,:,filter_channels]

    # Write image to tensorboard
    _view_x_train = _sample_train_images[:,:,:,[2,1,0]]
    _view_x_train = unnormalize(_view_x_train, f"{args.path}/processed/stats.json",channels=(2,1,0)) #(2,1,0)
    _view_x_train = _view_x_train.permute(0,3,1,2)
    train_img_grid = torchvision.utils.make_grid(_view_x_train, nrow=4)
    _labels = _sample_train_labels.permute(0,3,1,2)
    train_label_grid = torchvision.utils.make_grid(_labels, nrow=4)
    writer.add_image("Train/image", train_img_grid)
    writer.add_image("Train/labels", train_label_grid)

    _view_x_val = _sample_val_images[:,:,:,[2,1,0]]
    _view_x_val = unnormalize(_view_x_val, f"{args.path}/processed/stats.json",channels=(2,1,0))
    _view_x_val = _view_x_val.permute(0,3,1,2)
    val_img_grid = torchvision.utils.make_grid(_view_x_val, nrow=4)
    _labels = _sample_val_labels.permute(0,3,1,2)
    val_label_grid = torchvision.utils.make_grid(_labels, nrow=4)
    writer.add_image("Validation/image", val_img_grid)
    writer.add_image("Validation/labels", val_label_grid)

    # Calculate number of batches once
    n_batches = int(math.ceil(len(train_dataset)/args.batch_size))

    for epoch in range(1, args.epochs+1):
        ## Training loop
        loss = 0
        for i, (x,y) in enumerate(train_loader):
            x = x[:,:,:,filter_channels]
            frame.set_input(x,y)
            _loss = frame.optimize()
            print(f"Epoch {epoch}/{args.epochs}, Training batch {i+1} of {n_batches}, Loss= {_loss/args.batch_size:.5f}", end="\r", flush=True)
            loss += _loss
            if i == 0:
                metrics=frame.calculate_metrics()
            else:
                metrics+=frame.calculate_metrics()
        # Print and write scalars to tensorboard
        epoch_train_loss = loss/len(train_dataset)
        print(f"\nT_Loss: {epoch_train_loss:.5f}", end = " ")
        for i, k in enumerate(conf.metrics_opts):
            print(f", {k}: {metrics[i]/len(train_dataset):.3f}", end = " ")
        writer.add_scalar('Loss/train', epoch_train_loss, epoch)
        for i, k in enumerate(conf.metrics_opts):
            writer.add_scalar("Train/"+str(k), metrics[i]/len(train_dataset), epoch)
        # Write Images to tensorboard
        if epoch % args.save_every == 0:
            y_hat = frame.infer(_sample_train_images.to(frame.device))
            y_hat = torch.sigmoid(y_hat)
            _preds = y_hat.permute(0,3,2,1)
            pred_grid = torchvision.utils.make_grid(_preds, nrow=4)
            writer.add_image("Train/predictions", pred_grid, epoch)

        ## Validation loop
        loss = 0
        for i, (x,y) in enumerate(val_loader):
            x = x[:,:,:,filter_channels]
            y_hat = frame.infer(x.to(frame.device))
            _loss = frame.calc_loss(y_hat.to(frame.device), y.to(frame.device)).item()
            loss += _loss
            if i == 0:
                metrics=frame.calculate_metrics()
            else:
                metrics+=frame.calculate_metrics()
        epoch_val_loss = loss / len(val_dataset)
        frame.val_operations(epoch_val_loss)         
        # Print and write scalars to tensorboard
        print(f"\nV_Loss: {epoch_val_loss:.5f}", end = " ")
        for i, k in enumerate(conf.metrics_opts):
            print(f", {k}: {metrics[i]/len(val_dataset):.3f}", end = " ")
        writer.add_scalar('Loss/val', epoch_val_loss, epoch)
        for i, k in enumerate(conf.metrics_opts):
            writer.add_scalar("Validation/"+str(k), metrics[i]/len(val_dataset), epoch)
        # Write images to tensorboard
        if epoch % args.save_every == 0:
            y_hat = frame.infer(_sample_val_images.to(frame.device))
            y_hat = torch.sigmoid(y_hat)
            _preds = y_hat.permute(0,3,2,1)
            val_pred_grid = torchvision.utils.make_grid(_preds, nrow=4)
            writer.add_image("Validation/predictions", val_pred_grid, epoch)
        # Write combined loss graph to tensorboard
        writer.add_scalars('Loss', {'train':epoch_train_loss,
                                    'val':epoch_val_loss}, epoch)
        print("\n")
>>>>>>> Stashed changes
        
        if i == 0:
            metrics=frame.calculate_metrics()
        else:
            metrics+=frame.calculate_metrics()


    writer.add_scalar('Batch Val Loss', loss/len(val_dataset), epoch)
    print("val Loss: ", loss / len(val_loader))

<<<<<<< Updated upstream
    for k, item in enumerate(metrics):
        writer.add_scalar('Val Epoch Metrics '+str(k), item/len(val_dataset), epoch)
=======
    writer.close()
>>>>>>> Stashed changes
