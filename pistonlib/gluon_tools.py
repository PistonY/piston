# @File  : gluon_tools.py
# @Author: X.Yang
# @Contact : pistonyang@gmail.com
# @Date  : 18-12-7

import mxnet as mx
from mxnet import nd, gluon, image
from mxnet.gluon.data import Dataset, DataLoader
import os

__all__ = ['get_ensemble', 'predict_dataset',
           'predict']


def get_ensemble(nets, test_loader, ctx=mx.gpu()):
    metric = mx.metric.Accuracy()
    num_nets = len(nets)
    recored = {'result': [],
               'labels': []}

    for batch in test_loader:
        trans = batch[0].as_in_context(ctx)
        labels = batch[1].as_in_context(ctx)
        recored['labels'].append(labels)
        results = []
        for i, net in enumerate(nets):
            result = net(trans)
            result = nd.softmax(result, axis=1)
            results.append(result)
        results = sum(results) / num_nets
        recored['result'].append(nd.argmax(results, axis=1))
        metric.update(labels, results)
    recored['result'] = nd.concatenate(recored['result']).asnumpy()
    recored['labels'] = nd.concatenate(recored['labels']).asnumpy().ravel()
    _, acc = metric.get()
    return recored, acc


class predict_dataset(Dataset):
    def __init__(self, root_path, transform=None):
        self.transform = transform
        self.root_path = root_path
        self.list_file = self._get_file_list()

    def _get_file_list(self):
        return os.listdir(self.root_path)

    def __len__(self):
        return len(self.list_file)

    def __getitem__(self, item):
        file_path = os.path.join(self.root_path, self.list_file[item])
        img = image.imread(file_path)
        if self.transform:
            return self.transform(img, item)
        return img, item


def predict(nets: list, root_path, out_path,
            trans=None, batch_size=64):
    dt = predict_dataset(root_path, trans)
    val_data = gluon.data.DataLoader(
        dt, batch_size=batch_size, num_workers=2, shuffle=True, last_batch='keep')
    ctx = mx.gpu()
    num_nets = len(nets)
    output = ""

    for batch in val_data:
        trans = batch[0].as_in_context(ctx)
        file_idx = batch[1].asnumpy().ravel().tolist()
        results = []
        for i, net in enumerate(nets):
            result = net(trans)
            result = nd.softmax(result, axis=1)
            results.append(result)
        results = nd.argmax(sum(results) / num_nets, axis=1).asnumpy().tolist()
        for idx, rt in zip(file_idx, results):
            output += str(int(rt)) + '#' + str(dt.list_file[int(idx)]) + '\n'
    with open(out_path, 'w') as o:
        o.write(output)
    print('Done.')
