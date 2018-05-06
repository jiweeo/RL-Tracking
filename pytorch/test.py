from train import *
import os
import numpy as np
import torch
from torch.nn import functional as F
from utils import cropping, load_data, get_transform, load_otb_data
import os

os.environ['CUDA_VISIBLE_DEVICES']='0'

def test(env, R, root_dir, data_name='vot'):
    R.agent.eval()

    if data_name=='vot':
        imglist, gt_bboxes = load_data(root_dir)
    elif data_name == 'otb':
        imglist, gt_bboxes = load_otb_data(root_dir)
    else:
        return []

    init_gt = gt_bboxes[0]
    end = len(gt_bboxes)
    predicted_list = [init_gt]

    bbox = init_gt
    e = 0
    while e < end-1:
        # initialize the env. Current frame and previous bbox
        env.reset(imglist[e+1], gt_bboxes[e], bbox)

        # predict full episode
        print('tracking %d frame...' % (e+1))
        while True:
            bbox_img = cropping(env.img, env.state)
            bbox_img = R.transforms(bbox_img).cuda()
            bbox_img = bbox_img.unsqueeze(0)
            q_value, stop = R.agent(bbox_img)
            stop = F.softmax(stop.cpu(), dim=1)
            if stop[0][1] > 0.5:
                action = 10     # stop
            else:
                action = torch.argmax(q_value)
            _, is_t, r = env.step(action)
            if is_t:
                break

        e = e + 1
        predicted_list.append(env.state)
        bbox = env.state

    return predicted_list


def main():
    # create model
    args = parse_arguments()
    env = Env()
    transforms = get_transform()
    R = Reinforce(None, transforms=transforms)

    # load weights
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        R.agent.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

    predicted_list = test(env, R, args.data)

    # save result
    predicted_list = np.vstack(predicted_list)
    np.savetxt('predicted_bbox', predicted_list, fmt='%.3f', delimiter=',')


if __name__ == '__main__':
    main()
