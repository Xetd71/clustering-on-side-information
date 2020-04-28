import torch
import torch.nn as nn
from torch.nn import functional as F
import logging


from nn_clustering import PairEnum


# This file provides the template Learner. The Learner is used in training/evaluation loop
# The Learner implements the training procedure for specific task.
# The default Learner is from classification task.

class ClusteringTemplate(nn.Module):
    def __init__(self, model, criterion, optimizer, scheduler):
        super(ClusteringTemplate, self).__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epoch = 0
        self.KPI = -1  # An non-negative index, larger is better.

    def forward(self, x):
        logits = self.model.forward(x)
        prob = F.softmax(logits, dim=1)
        return prob

    def forward_with_criterion(self, inputs, simi, mask=None, **kwargs):
        prob = self.forward(inputs)
        prob1, prob2 = PairEnum(prob, mask)
        return self.criterion(prob1, prob2, simi), prob

    def learn(self, inputs, targets, **kwargs):
        loss, out = self.forward_with_criterion(inputs, targets, **kwargs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        loss = loss.detach()
        out = out.detach()
        return loss, out

    def step_schedule(self, epoch):
        self.epoch = epoch
        self.scheduler.step(self.epoch)
        for param_group in self.optimizer.param_groups:
            logging.info("LR: %s", param_group["lr"])

    def save_model(self, savename):
        model_state = self.model.state_dict()
        if isinstance(self.model, nn.DataParallel):
            # Get rid of "module" before the name of states
            model_state = self.model.module.state_dict()
        for key in model_state.keys():  # Always save it to cpu
            model_state[key] = model_state[key].cpu()
        logging.info("=> Saving model to: %s", savename)
        torch.save(model_state, savename + ".pth")
        logging.info("=> Done")

    def snapshot(self, savename, KPI=-1):
        model_state = self.model.state_dict()
        optim_state = self.optimizer.state_dict()
        checkpoint = {
            "epoch": self.epoch,
            "model": model_state,
            "optimizer": optim_state
        }
        logging.info("=> Saving checkpoint to: %s.checkpoint.pth", savename)
        torch.save(checkpoint, savename + ".checkpoint.pth")
        logging.info("=> Done")
        if KPI >= self.KPI:
            logging.info("=> New KPI: %s; previous KPI: %s", KPI, self.KPI)
            self.KPI = KPI
            self.save_model(savename + ".model")

    def resume(self, resumefile):
        logging.info("=> Loading checkpoint: %s", resumefile)
        checkpoint = torch.load(resumefile, map_location=lambda storage, loc: storage)  # Load to CPU as the default!
        self.epoch = checkpoint["epoch"]
        logging.info("=> resume epoch: %s", self.epoch)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        logging.info("=> Done")
        return self.epoch
