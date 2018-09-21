# @File  : mxnet_metric.py
# @Author: Piston.Y
# @Contact : pistonyang@gmail.com
# @Date  : 18-9-21

from mxnet.metric import EvalMetric, check_label_shapes
import numpy as np


class CS(EvalMetric):
    def __init__(self, n, name='CS', **kwargs):
        super(CS, self).__init__(name, **kwargs)
        self.n = n

    def update(self, labels, preds):
        labels, preds = check_label_shapes(labels, preds, True)

        for label, pred_label in zip(labels, preds):
            label = label.asnumpy().astype('int32').ravel()
            pred_label = pred_label.asnumpy().astype('int32').ravel()

            self.sum_metric += (np.abs(label - pred_label) < self.n).sum()
            self.num_inst += len(label)


class e_error(EvalMetric):
    def __init__(self, name='e-error', **kwargs):
        super(e_error, self).__init__(name, **kwargs)

    def update(self, labels, preds):
        labels, preds = check_label_shapes(labels, preds, True)

        for label, pred_label in zip(labels, preds):
            label = label.asnumpy().astype('int32').ravel()
            pred_label = pred_label.asnumpy().astype('int32').ravel()

            mu = np.mean(label)
            sigma = np.sqrt(np.mean(np.power(label - mu, 2)))

            self.sum_metric += np.sum(1 - np.exp(-np.power(pred_label - mu, 2) / (2 * np.power(sigma, 2))))
            self.num_inst += len(label)
