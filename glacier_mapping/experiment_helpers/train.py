#!/usr/bin/env python
import pathlib
import yaml
from addict import Dict
from glacier_mapping.data.data import fetch_loaders
from glacier_mapping.models.frame import Framework
from torch.utils.tensorboard import SummaryWriter
import glacier_mapping.train as tr
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess raw tiffs into slices")
    parser.add_argument("-d", "--data_dir", type=str, default = "conf/masks_geo_exper.conf")
    parser.add_argument("-c", "--train_yaml", type=str)
    parser.add_argument("-p", "--postprocess_conf", type=str, default = "conf/process_geo.conf")
    parser.add_argument("-b", "--batch_size", type=int, default = 16)
    parser.add_argument("-r", "--run_name", type=str, default="demo")
    parser.add_argument("-e", "--epochs", type=int, default=200)
    parser.add_argument("-s", "--save_every", type=int, default=50)
    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    conf = Dict(yaml.safe_load(open(args.train_yaml, "r")))
    processed_dir = data_dir / "processed"

    # Train a model
    args = Dict({
        "batch_size": args.batch_size,
        "run_name": args.run_name,
        "epochs": args.epochs,
        "save_every": args.save_every
    })

    loaders = fetch_loaders(processed_dir, args.batch_size)
    frame = Framework(
        model_opts=conf.model_opts,
        optimizer_opts=conf.optim_opts,
        reg_opts=conf.reg_opts
    )

    # Setup logging
    writer = SummaryWriter(f"{data_dir}/runs/{args.run_name}/logs/")
    writer.add_text("Arguments", json.dumps(vars(args)))
    writer.add_text("Configuration Parameters", json.dumps(conf))
    out_dir = f"{data_dir}/runs/{args.run_name}/models/"

    for epoch in range(args.epochs):

        # train loop
        loss_d = {}
        loss_d["train"], metrics = tr.train_epoch(loaders["train"], frame, conf.metrics_opts)
        tr.log_metrics(writer, metrics, loss_d["train"], epoch)
        tr.log_images(writer, frame, next(iter(loaders["train"])), epoch)

        # validation loop
        loss_d["val"], metrics = tr.validate(loaders["val"], frame, conf.metrics_opts)
        tr.log_metrics(writer, metrics, loss_d["val"], epoch, "val")
        tr.log_images(writer, frame, next(iter(loaders["val"])), epoch, "val")

        # Save model
        writer.add_scalars("Loss", loss_d, epoch)
        if epoch % args.save_every == 0:
            frame.save(out_dir, epoch)

        print(f"{epoch}/{args.epochs} | train: {loss_d['train']} | val: {loss_d['val']}")

    frame.save(out_dir, "final")
    writer.close()
