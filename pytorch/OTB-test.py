from test import test
from utils import get_transform
from train import Reinforce
from environment import Env
import os
import numpy as np
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def main():
    transform = get_transform()
    R = Reinforce(None, transforms=transform)
    env = Env()
    data_root = '../data/OTB100'
    model_root = '../cv/run2/latest.pth'

    # load weights
    if os.path.isfile(model_root):
        print("=> loading checkpoint '{}'".format(model_root))
        checkpoint = torch.load(model_root)
        R.agent.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(model_root, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(model_root))

    test_list = os.listdir(data_root)
    for i, filename in enumerate(test_list):
        print('[%d/%d] testing %s' % (i, len(test_list), filename))
        predicted_bbox = test(env, R, os.path.join(data_root, filename), data_name='otb')
        predicted_bbox = np.vstack(predicted_bbox)
        np.savetxt(os.path.join(data_root, filename, 'pred_rect_sl.txt'), predicted_bbox, fmt='%.3f')


if __name__ == '__main__':
    main()