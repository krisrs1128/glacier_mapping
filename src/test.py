#!/usr/bin/env python
"""
Test Pipeline:
    1- Initialize test data
    2- Initialize the framework
    4- Saves an output csv file in the specified directory
    5- Save images in output directory
"""
from src.models.unet import Unet
import glob
import geopandas as gpd
import rasterio
import numpy as np
from matplotlib import pyplot as plt
import torch
import src.metrics as m
import pandas as pd
from statistics import mean
import os

def get_x_y(sat_imgs):
    list_x = []
    list_y = []
    list_imgpath = []
    path = "/".join(sat_imgs[0].split("/")[:-1])
    for fname in sat_imgs:
        n_slice = fname.split("/")[-1].split("_")[1]
        n_img = fname.split("/")[-1].split("_")[3].split(".")[0]
        imagepath = f"{path}/slice_{n_slice}_img_{n_img}.npy"
        maskpath = f"{path}/slice_{n_slice}_mask_{n_img}.npy"
        x = np.load(imagepath)
        y = np.load(maskpath)
        y = np.expand_dims(y, axis=2)
        list_x.append(x)
        list_y.append(y)
        list_imgpath.append(imagepath)
    return list_imgpath, np.asarray(list_x), np.asarray(list_y)

def save_image(savepath, y):
    savepath = "."+savepath.split(".")[1]+".png"
    y = np.squeeze(y.numpy()).astype(int)
    plt.imsave(savepath,y)

if __name__ == '__main__':
    # Define paths and threshold
    basepath = "./data/new/glaciers"
    
    testpath = basepath+"/img/train/*"
    savepath = basepath+"/img/preds/"
    threshold = 0.3

    models = ["model_188.pt"]
    
    # Get X and y
    sat_imgs = sorted(glob.glob(testpath))
    fname, X, y = get_x_y(sat_imgs)

    metric_results = {
            "model_path" : [],
            "fname" : [],
            "pred_fname" : [],
            "mean_precision" : [],
            "mean_tp" : [],
            "mean_fp" : [],
            "mean_fn" : [],
            "mean_pixel_acc" : [],
            "mean_dice" : [],
            "mean_recall" : [],
            "mean_IoU" : []
        }

    mean_metric_results = dict(metric_results)

    for model in models:
        savepath = basepath+"/img/preds/"+model.split(".")[0]+"/"
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        gen = (x for x in metric_results if x not in ["model_path","fname","pred_fname"])
        for key in gen:
            vars()["model_"+key] = []

        model_path = basepath+"/models/"+model
        # Load the U-Net model
        state_dict = torch.load(model_path, map_location="cpu")
        model = Unet(3, 1, 4)
        # model.load_state_dict(state_dict)

        for _name, _x, _y in zip(fname, X, y):
            # get unet prediction
            with torch.no_grad():
                _x = _x.transpose((2,0,1))
                _x = np.expand_dims(_x, axis=0)
                y_hat = model(torch.from_numpy(_x)).numpy()
                y_hat = np.transpose(y_hat[0], (1,2,0))
                y_hat = torch.from_numpy(y_hat > threshold) 
                save_image(savepath+_name.split("/")[-1],y_hat)
            _y = torch.from_numpy(np.array(_y, dtype=bool))

            metric_results["fname"].append(_name)
            metric_results["model_path"].append(model_path)
            metric_results["pred_fname"].append(savepath+_name.split("/")[-1])
            # get channel wise metrices and compute mean

            gen = (x for x in metric_results if x not in ["model_path","fname","pred_fname"])
            for key in gen:
                vars()[key] = []

            for k in range(_y.shape[2]):
                mean_precision.append(m.precision(_y[:, :, k], y_hat[:, :, k]))
                tp,fp,fn = m.tp_fp_fn(_y[:, :, k], y_hat[:, :, k])
                mean_tp.append(tp)
                mean_fp.append(fp)
                mean_fn.append(fn)
                mean_pixel_acc.append(m.pixel_acc(_y[:, :, k], y_hat[:, :, k]))
                mean_dice.append(m.dice(_y[:, :, k], y_hat[:, :, k]))
                mean_recall.append(m.recall(_y[:, :, k], y_hat[:, :, k]))
                mean_IoU.append(m.IoU(_y[:, :, k], y_hat[:, :, k]))

            gen = (x for x in metric_results if x not in ["model_path","fname","pred_fname"])
            for key in gen:
                _ = vars()[key]
                metric_results[key].append(mean(_))

            gen = (x for x in metric_results if x not in ["model_path","fname","pred_fname"])
            for key in gen:
                model_list = vars()["model_"+key]
                k = vars()[key]
                model_list.append(mean(k))

        # Append to mean_metric_results
        mean_metric_results["model_path"] = model_path
        mean_metric_results["fname"] = "Test set"
        mean_metric_results["pred_fname"] = savepath
        gen = (x for x in metric_results if x not in ["model_path","fname","pred_fname"])
        for key in gen:
            model_list = vars()["model_"+key]
            mean_metric_results[key].append(mean(model_list))
    
    # print(metric_results)
    pd.DataFrame(mean_metric_results).to_csv(savepath+"/mean_results.csv", index=False)
    pd.DataFrame(metric_results).to_csv(savepath+"/results.csv", index=False)
