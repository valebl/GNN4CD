import torch
import torch.nn as nn


#-----------------------------------------------------
#--------------- CUSTOM LOSS FUNCTIONS ---------------
#-----------------------------------------------------


class weighted_mse_loss():
    def __call__(input_batch, target_batch, weights):
        e = (input_batch - target_batch) ** 2
        return torch.sum(weights * e) / torch.sum(weights)


class weighted_mae_loss():
    def __call__(input_batch, target_batch, weights):
        e = torch.abs(input_batch - target_batch)
        return torch.sum(weights * e) / torch.sum(weights)


class quantized_loss():
    def __init__(self, alpha=0.025):
        self.mse_loss = nn.MSELoss()
        self.alpha = alpha
        print(f"alpha: {self.alpha}")

    def __call__(self, prediction_batch, target_batch, bins):
        loss_mse = self.mse_loss(prediction_batch, target_batch)
        loss_quantized = 0
        bins = bins.int()
        for b in torch.unique(bins):
            mask_b = (bins == b)
            loss_b = self.mse_loss(prediction_batch[mask_b], target_batch[mask_b])
            loss_quantized += loss_b
        return loss_mse + self.alpha * loss_quantized, loss_mse, loss_quantized
