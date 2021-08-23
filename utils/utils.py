# -*- coding: utf-8 -*-
# ---------------------

import json
import os
from datetime import datetime
from enum import Enum

import PIL
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import paddle
from PIL.Image import Image
from matplotlib import cm
from matplotlib import figure
from pathlib import Path
from paddle import nn



class QuantileLoss(nn.Layer):
    ## From: https://medium.com/the-artificial-impostor/quantile-regression-part-2-6fdbc26b2629

    def __init__(self, quantiles):
        ##takes a list of quantiles
        super().__init__()
        self.quantiles = quantiles

    def numpy_normalised_quantile_loss(self, y_pred, y, quantile):
        """Computes normalised quantile loss for numpy arrays.
        Uses the q-Risk metric as defined in the "Training Procedure" section of the
        main TFT paper.
        Args:
          y: Targets
          y_pred: Predictions
          quantile: Quantile to use for loss calculations (between 0 & 1)
        Returns:
          Float for normalised quantile loss.
        """
        if not isinstance(y_pred, paddle.Tensor):
            y_pred = paddle.to_tensor(y_pred,paddle.float32)

        if len(y_pred.shape) == 3:
            ix = self.quantiles.index(quantile)
            y_pred = y_pred[..., ix]

        if not isinstance(y, paddle.Tensor):
            y = paddle.to_tensor(y,paddle.float32)

        prediction_underflow = y - y_pred
        weighted_errors = quantile * paddle.maximum(prediction_underflow, paddle.to_tensor(0.,paddle.float32)) \
                          + (1. - quantile) * paddle.maximum(-prediction_underflow, paddle.to_tensor(0.))

        quantile_loss = paddle.mean(weighted_errors)
        normaliser = paddle.abs(y).mean()

        return 2 * quantile_loss / normaliser

    def forward(self, preds, target, ret_losses=True):
        assert target.stop_gradient
        assert preds.shape[0] == target.shape[0]
        losses = []

        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, :, i]
            losses.append(
                paddle.maximum(
                    (q - 1) * errors,
                    q * errors
                ).unsqueeze(1))
        loss = paddle.mean(
            paddle.sum(paddle.concat(losses, axis=1), axis=1))
        if ret_losses:
            return loss, losses
        return loss


def unnormalize_tensor(data_formatter, data, identifier):
    data = pd.DataFrame(
        list(data.numpy()),
        columns=[
            't+{}'.format(i)
            for i in range(data.shape[1])
        ])

    data['identifier'] = np.array(identifier)
    data = data_formatter.format_predictions(data)

    return data.drop(columns=['identifier']).values


def symmetric_mean_absolute_percentage_error(forecast, actual):
    # Symmetric Mean Absolute Percentage Error (SMAPE)
    sequence_length = forecast.shape[1]
    sumf = paddle.sum(paddle.abs(forecast - actual) / (paddle.abs(actual) + paddle.abs(forecast)), axis=1)
    return paddle.mean((2 * sumf) / sequence_length)


def plot_temporal_serie(y_pred, y_true):
    if isinstance(y_pred, paddle.Tensor):
        y_pred = y_pred.numpy()

    if isinstance(y_true, paddle.Tensor):
        y_true = y_true.numpy()

    ind = np.random.choice(y_pred.shape[0])
    plt.plot(y_pred[ind, :, 0], label='pred_1')
    plt.plot(y_pred[ind, :, 1], label='pred_5')
    plt.plot(y_pred[ind, :, 2], label='pred_9')
    plt.plot(y_true[ind, :, 0], label='true')
    plt.legend()
    plt.show()


def imread(path):
    """
    Reads the image located in `path`
    :param path:
    :return:
    """
    with open(path, 'rb') as f:
        with PIL.Image.open(f) as img:
            return img.convert('RGB')


def pyplot_to_numpy(pyplot_figure):
    """
    Converts a PyPlot figure into a NumPy array
    :param pyplot_figure: figure you want to convert
    :return: converted NumPy array
    """
    pyplot_figure.canvas.draw()
    x = np.fromstring(pyplot_figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    x = x.reshape(pyplot_figure.canvas.get_width_height()[::-1] + (3,))
    return x


def pyplot_to_tensor(pyplot_figure):
    """
    Converts a PyPlot figure into a PyTorch tensor
    :param pyplot_figure: figure you want to convert
    :return: converted PyTorch tensor
    """
    x = pyplot_to_numpy(pyplot_figure=pyplot_figure)
    return x


def apply_colormap_to_tensor(x, cmap='jet', range=(None, None)):
    """
    :param x: Tensor with shape (1, H, W)
    :param cmap: name of the color map you want to apply
    :param range: tuple of (minimum possible value in x, maximum possible value in x)
    :return: Tensor with shape (3, H, W)
    """
    cmap = cm.ScalarMappable(cmap=cmap)
    cmap.set_clim(vmin=range[0], vmax=range[1])
    x = x.detatch().cpu().numpy()
    x = x.squeeze()
    x = cmap.to_rgba(x)[:, :, :-1]
    return x

