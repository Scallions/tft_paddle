import numpy as np
import paddle
import paddle.nn as nn

class Normalise_QuantileLoss(nn.Layer):
  def __init__(self, quantiles):
    super(Normalise_QuantileLoss, self).__init__()
    self.quantiles = quantiles

  def forward(self, preds, target):
    prediction_underflow = target - preds
    constant = paddle.to_tensor(0.,dtype=paddle.float32)
    weighted_errors = self.quantiles * paddle.maximum(prediction_underflow, constant) \
                      + (1. - self.quantiles) * paddle.maximum(-prediction_underflow, constant)
    quantile_loss = weighted_errors.mean()
    normaliser = target.abs().mean()

    return 2 * quantile_loss / normaliser

class QuantileLoss(nn.Layer):
    def __init__(self, quantiles):
      super(QuantileLoss, self).__init__()
      self.quantiles = quantiles

    def forward(self, preds, target):
      assert target.stop_gradient
      assert preds.shape[0] == target.shape[0]
      losses = []
      for i, q in enumerate(self.quantiles):
        errors = target - preds[:, i]
        left = (q - 1) * errors
        right = q * errors
        losses.append(paddle.maximum(left, right).unsqueeze(1))
      loss = paddle.mean(paddle.sum(paddle.concat(losses, axis=1), axis=1))
      return loss