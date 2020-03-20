#!/usr/bin/env python



class Framework():

    def __init__(self, loss=None, model_opts=None, optimizer_opts=None):
        if loss is None:
            loss = torch.nn.BCELoss

        model_def = getattr(models, model_opts.name)
        self.model = model_def(**model_opts.args)

        optimizer_def = getattr(torch.optim, optimizer_opts.name)
        self.optimizer = optimizer_def(**optimizer_opts.args)

        def set_input(self, x, y):
            self.x = x
            self.y = y

        def optimize(self):
            y_hat = self.model(self.x)
            loss = self.loss(self.y, y_hat)
            loss.backward()
            self.optimizer.step()
            return loss

        def save(self, out_dir, epoch):
            model_path = Path(out_dir, f"model_{epoch}.pt")
            optim_path = Path(out_dir, f"optim_{epoch}.pt")
            torch.save(self.model.state_dict, model_path)
            torch.save(self.optimizer.state_dict, optim_path)

        def infer(x):
            with torch.no_grad():
                return self.model(x)

        def loss(y, y_hat):
            return self.loss(y, y_hat)
