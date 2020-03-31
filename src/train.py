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
import torch
from torch.utils.data import DataLoader
import data
from frame import Framework
from pathlib import Path
import addict

path = "/scratch/sankarak/data/glaciers/processed/"
train_dataset = data.GlacierDataset(Path(path, "train"))
val_dataset = data.GlacierDataset(Path(path, "test"))

train_loader = DataLoader(train_dataset,batch_size=2, shuffle=True, num_workers=1)
val_loader = DataLoader(val_dataset,batch_size=2, shuffle=True, num_workers=1)

model_opts = addict.Dict({'name' : 'Unet', 'args' : {'inchannels': 3, 'outchannels' : 1, 'net_depth':4}})
optim_opts = addict.Dict({'name': 'Adam'})
frame = Framework(model_opts=model_opts, optimizer_opts=optim_opts)


## Train Loop
epochs=10
for epoch in range(1, epochs):
    loss = 0
    for x,y in train_loader:
        frame.set_input(x,y)
        loss+=frame.optimize()
        print("Train Loss:", loss)
    print(loss/len(train_dataset))
    if epoch%5==0:
        frame.save(out_dir, epoch)

    ## validation loop
    loss = 0
    for x,y in val_loader:
        y_hat = frame.infer(x)
        loss+=frame.loss(y, y_hat)
        print("val Loss: ", loss)
    print(loss/len(val_dataset))

