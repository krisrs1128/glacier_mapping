#!/usr/bin/env python
import argparse
import pathlib
import yaml
import json
from addict import Dict
import torch

from glacier_mapping.data.data import fetch_loaders
from glacier_mapping.models.frame import Framework
from glacier_mapping.models.metrics import diceloss
from torch.utils.tensorboard import SummaryWriter
import glacier_mapping.train as tr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess raw tiffs into slices")
    parser.add_argument("-d", "--data_dir", type=str)
    parser.add_argument("-c", "--train_yaml", type=str)
    parser.add_argument("-p", "--postprocess_conf", type=str, default = "conf/process_geo.conf")
    parser.add_argument("-b", "--batch_size", type=int, default = 16)
    parser.add_argument("-r", "--run_name", type=str, default="demo")
    parser.add_argument("-e", "--epochs", type=int, default=200)
    parser.add_argument("-s", "--save_every", type=int, default=50)
    parser.add_argument("-l", "--loss_type", type=str, default="dice")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    conf = Dict(yaml.safe_load(open(args.train_yaml, "r")))
    loss_type = args.loss_type
    device = args.device
    if device is not None:
        device = torch.device(device)


    # Train a model
    args = Dict({
        "batch_size": args.batch_size,
        "run_name": args.run_name,
        "epochs": args.epochs,
        "save_every": args.save_every
    })

    loaders = fetch_loaders(data_dir, args.batch_size, shuffle=True)

    # TODO:handle this error better
    # if input mask dimension different than outchannels
    outchannels = conf.model_opts.args.outchannels
    y_channels = [y.shape[-1] for _, y in loaders["val"]][0]
    if y_channels != outchannels:
        raise ValueError("Output dimension is different from model outchannels.")

    # TODO: try to have less nested if/else
    # get dice loss
    if loss_type == "dice":
        if outchannels > 1:
            loss_weight = [0.6, 0.9, 0.2] # clean ice, debris, background
            label_smoothing = 0.2
            loss_fn = diceloss(act=torch.nn.Softmax(dim=1), w=loss_weight,
                               outchannels=outchannels, label_smoothing=label_smoothing)
        else:
            loss_fn = diceloss()
    else: loss_fn = None

    frame = Framework(
        model_opts=conf.model_opts,
        optimizer_opts=conf.optim_opts,
        reg_opts=conf.reg_opts,
        loss_fn=loss_fn,
        device=device
    )

    # Setup logging
    writer = SummaryWriter(f"{data_dir}/runs/{args.run_name}/logs/")
    writer.add_text("Arguments", json.dumps(vars(args)))
    writer.add_text("Configuration Parameters", json.dumps(conf))
    out_dir = f"{data_dir}/runs/{args.run_name}/models/"
    mask_names = conf.log_opts.mask_names

    for epoch in range(args.epochs):

        # train loop
        loss_d = {}
        loss_d["train"], metrics = tr.train_epoch(loaders["train"], frame, conf.metrics_opts)
        tr.log_metrics(writer, metrics, loss_d["train"], epoch, mask_names=mask_names)
        if (epoch+1) % args.save_every == 0:
            tr.log_images(writer, frame, next(iter(loaders["train"])), epoch)

        # validation loop
        loss_d["val"], metrics = tr.validate(loaders["val"], frame, conf.metrics_opts)
        
        tr.log_metrics(writer, metrics, loss_d["val"], epoch, "val", mask_names=mask_names)
        if (epoch+1) % args.save_every == 0:
            tr.log_images(writer, frame, next(iter(loaders["val"])), epoch, "val")

        # Save model
        writer.add_scalars("Loss", loss_d, epoch)
        if (epoch+1) % args.save_every == 0:
            frame.save(out_dir, epoch)

        print(f"{epoch}/{args.epochs} | train: {loss_d['train']} | val: {loss_d['val']}")

    frame.save(out_dir, "final")
    writer.close()
