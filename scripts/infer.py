#!/usr/bin/env python
"""
Example inference

From the colab notebook.
"""
import pathlib
from addict import Dict
from glacier_mapping.models.frame import Framework
import glacier_mapping.infer as gmi
import torch
import yaml

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inference helpers")
    parser.add_argument("-m", "--model_path", type=str, default = "runs/demo/models/model_final.pt")
    parser.add_argument("-o", "--output_dir", type=str, default = "predictions")
    parser.add_argument("-i", "--input_tiff", type=str, default = "demo_ICIMOD_2005_train_2-0000000000-0000008960.tif")
    parser.add_argument("-c", "--train_yaml", type=str, default = "conf/train.yaml")

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    train_conf = Dict(yaml.safe_load(open(args.train_yaml, "r")))
    model = Framework(torch.nn.BCEWithLogitsLoss(), train_conf.model_opts, train_conf.optim_opts).model
    if torch.cuda.is_available():
        state_dict = torch.load(args.model_path)
    else:
        state_dict = torch.load(model_path, map_location="cpu")

    model.load_state_dict(state_dict)
    img, _, y_hat = gmi.predict_tiff(args.input_tiff)
