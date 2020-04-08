#!/usr/bin/env python
import torch
import src.unet
import src.metrics

class Framework():

    def __init__(self,loss_fn=None, model_opts=None, optimizer_opts=None, metrics_opts=None):
        if loss_fn is None:
            loss_fn = torch.nn.BCEWithLogitsLoss()
        self.loss_fn = loss_fn
        model_def = getattr(src.unet, model_opts.name)
        self.model = model_def(**model_opts.args)
        

        optimizer_def = getattr(torch.optim, optimizer_opts.name)
        self.optimizer = optimizer_def(self.model.parameters(), **optimizer_opts.args)
        self.metrics_opts = metrics_opts

    def set_input(self, x, y):
        self.x = x.permute(0, 3, 1, 2)
        self.y = y.permute(0, 1, 2)

    def optimize(self):
        self.optimizer.zero_grad()
        self.y_hat = self.model(self.x)
        loss = self.loss(self.y_hat, self.y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save(self, out_dir, epoch):
        model_path = Path(out_dir, f"model_{epoch}.pt")
        optim_path = Path(out_dir, f"optim_{epoch}.pt")
        torch.save(self.model.state_dict, model_path)
        torch.save(self.optimizer.state_dict, optim_path)

    def infer(self, x):
        x = x.permute(0, 3, 1, 2)
        with torch.no_grad():
            return self.model(x)

    def loss(self, y, y_hat):
        return self.loss_fn(y, y_hat)


    def calculate_metrics(self):

        """
        y = np.random.uniform(0,1,(512,512,3))>0.5
        yhat = np.random.unform(0,1,(512,512,3))
        model_opts = addict.Dict({"name" : "Unet", "args" : {"inchannels": 3, "outchannels": 1, "net_depth": 2}})
        optim_opts = addict.Dict({"name": "Adam", "args": {"lr": 1e-4}})

        metrics_opts = addict.Dict({"precision": {"threshold": 0.2}, "IoU": {"threshold": 0.4}})
        frame = Framework(model_opts=model_opts, optimizer_opts=optim_opts, metrics_opts=metrics_opts)
        
        """
        results = []
        for metrics in self.metrics_opts:
            for k,v in self.metrics_opts.items():
                yhat_temp = self.y_hat
                if "threshold" in v.keys():
                   yhat_temp = self.y_hat > v["threshold"]
                metric_fun = getattr(src.metrics,k)
                metric_value = metric_fun(self.y, yhat_temp)
                results.append(metric_value)

        return np.array(results)


    

