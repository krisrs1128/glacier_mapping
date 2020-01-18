import math
import yaml
from addict import Dict

import numpy as np
import torch
import torchvision.transforms as T

from src.exper import induce_config
from src.unet import Unet
import src.utils as utils
from src.preprocess import get_normalization


def load_model(conf, model_path):
    """Take a conf dict and a model path and return model"""
    
    inchannels, outchannels, _ = induce_config(conf["data"])
    
    conf["model"]["inchannels"] = inchannels
    conf["model"]["outchannels"] = outchannels

    model = Unet(**conf["model"])
    model.load_state_dict(torch.load(model_path))
    
    return model

def strip_conf(conf_path):
    """Transform a dictionary where the value of interest is hold in value
    key into direct refrence"""
    
    with open(conf_path, "r") as stream: 
        conf = Dict(yaml.safe_load(stream))
    for key, data in conf.items():
        if (isinstance(conf[key], Dict)) and ('value' in conf[key]):
            conf[key] = conf[key]['value']
    return conf


def glacier_data(config, img, img_transform=None):
    """From an numpy array to glacier data
    according to config"""

    img = T.ToTensor()(img)

    if config["data"]["use_snow_i"]:
        snow_index = utils.get_snow_index(img)
        snow_index = np.expand_dims(snow_index, axis=0)
        img = np.concatenate((img, snow_index), axis=0)
        img = torch.from_numpy(img)

    img = img[config["data"]["channels_to_inc"]]

    if img_transform is not None:
        img = img_transform(img)
    return img.unsqueeze(0)

def stitch_slices(slices, h, w, overlap=3):
    """Stitch slices into an wxh size, with overlap"""
    
    stitched = np.random.rand(h, w)
    size = slices[0].shape
    h_slices = math.ceil((h - 2 * overlap) / (size[0] - 2 * overlap))
    w_slices = math.ceil((w - 2 * overlap) / (size[1] - 2 * overlap))

    for i in range(len(slices)):
        left_border = (i % (w_slices)) == 0
        right_border = (i % (w_slices)) == (w_slices - 1)
        top_border = (i / (w_slices)) <= 1
        bottom_border = (i / w_slices) >= (h_slices - 1)
        
        top_crop = 0 if top_border else int(overlap / 2) 
        bottom_crop = 0 if bottom_border else int(overlap / 2) 
        left_crop = 0 if left_border else int(overlap / 2)
        right_crop = 0 if right_border else int(overlap / 2)
        
        bottom_index = -bottom_crop if (bottom_crop > 0) else None
        right_index = -right_crop if (right_crop > 0) else None
        cropped_slice = slices[i][top_crop: bottom_index, left_crop: right_index]
            
        extra_i = int(overlap / 2) if (i // w_slices) > 0 else 0
        extra_j = int(overlap / 2) if (i % w_slices) > 0 else 0
        
        start_i = (i // w_slices) * (size[0] - overlap) + extra_i
        start_j = (i % w_slices) * (size[1] - overlap) + extra_j

        
        if right_border:
            start_j = w - cropped_slice.shape[1]
        if bottom_border:
            start_i = h - cropped_slice.shape[0]
            
        end_i = start_i +  cropped_slice.shape[0]
        end_j = start_j +  cropped_slice.shape[1]
        stitched[start_i: end_i, start_j: end_j] = cropped_slice
        
    return stitched

def tiff_to_seg(img, model_path, conf_path, normalization_folder, slice_size=(512, 512), overlap=6):
    """From a tiff to seg"""

    conf = strip_conf(conf_path)
    conf["data"]["path"] = normalization_folder
    img_transform, _ = get_normalization(conf["data"])
    model = load_model(conf, model_path)
    model.eval()
    img = np.nan_to_num(img)
    img_slices = utils.slice_image(img, size=slice_size, overlap=overlap*2)
    h, w = img.shape[:2]
    pred_slices = []
    for img_slice in img_slices:
        with torch.no_grad():
            img_slice = glacier_data(conf, img_slice, img_transform=img_transform)
            pred = model(img_slice)
            pred_p, _ = utils.get_pred_mask(pred)
        pred_slices.append(pred_p.squeeze())
    stitched = stitch_slices(pred_slices, h, w, overlap=overlap*2)

    return stitched 
