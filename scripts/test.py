# python3 -m scripts.test -d ./processed/ -c ./conf/train.yaml -n 500 -r demo --device cuda
import argparse
import pathlib
import yaml
import json
from addict import Dict
import torch
import glob
from glacier_mapping.data.data import fetch_loaders
from glacier_mapping.models.frame import Framework
from glacier_mapping.models.metrics import diceloss
import numpy as np
import os
import matplotlib.pyplot as plt
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test slices to prediction images")
    parser.add_argument("-d", "--data_dir", type=str)
    parser.add_argument("-c", "--train_yaml", type=str)
    parser.add_argument("-r", "--run_name", type=str, default="demo")
    parser.add_argument("-n", "--epoch_num", type=str, default="100")
    parser.add_argument("-b", "--batch_size", type=int, default = 16)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    conf = Dict(yaml.safe_load(open(args.train_yaml, "r")))
    device = args.device
    if device is not None:
        device = torch.device(device)

    model_dir = f"{data_dir}/runs/{args.run_name}/models/model_{args.epoch_num}.pt"

    outchannels = conf.model_opts.args.outchannels
    if outchannels > 1:
        loss_weight = [1 for _ in range(outchannels)]
        loss_weight[-1] = 0 # background
        loss_fn = diceloss(act=torch.nn.Softmax(dim=1), w=loss_weight,
                               outchannels=outchannels)
    else:
        loss_fn = diceloss()

    frame = Framework(
        model_opts=conf.model_opts,
        optimizer_opts=conf.optim_opts,
        reg_opts=conf.reg_opts,
        loss_fn=loss_fn,
        device=device
    )
    
    unet = frame.model
    unet.load_state_dict(torch.load(model_dir))
    unet = unet.to(device)

    slices_dir = f"{data_dir}/test/*img*"
    pred_dir = f"{data_dir}/preds/"
    
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)

    slices = glob.glob(slices_dir)

    total_inference_time = 0
    for s in slices:
        filename = s.split("/")[-1].replace("npy","png")
        inp_np = np.load(s)
        start = time.time()
        nan_mask = np.isnan(inp_np[:,:,:9]).any(axis=2)
        inp_tensor = torch.from_numpy(np.expand_dims(np.transpose(inp_np, (2,0,1)), axis=0))
        inp_tensor = inp_tensor.to(device)
        output = unet(inp_tensor)
        output_np = output.detach().cpu().numpy()
        output_np = np.transpose(output_np[0], (1,2,0))
        output_np = np.argmax(output_np, axis=2)
        output_np[nan_mask] = 3
        total_inference_time += (time.time() - start)
        plt.imsave(f"{pred_dir}{filename}", output_np, vmin=0, vmax=3)
    
    print(f"Total inference time: {total_inference_time}")
