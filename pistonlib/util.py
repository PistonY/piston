# @File  : util.py
# @Author: Piston.Y
# @Contact : pistonyang@gmail.com
# @Date  : 18-9-18
import os


def format_time(prev_time, cur_time) -> str:
    h, remainder = divmod((cur_time - prev_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    time_str = "Time %02d:%02d:%02d." % (h, m, s)
    return time_str


def inf_train_gen(loader):
    while True:
        for batch in loader:
            yield batch


def rescale(x, x_min=None, x_max=None):
    if x_min is None:
        x_min = x.min().asscalar()
    if x_max is None:
        x_max = x.max().asscalar()
    return (x - x_min) / (x_max - x_min)


def checkout_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
