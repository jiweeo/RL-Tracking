from train import *
import os
from torch.autograd import Variable
import numpy as np
import torch

def test(env, R, start_epoch):

    e = start_epoch
    init_gt = env.gt_bboxes[e]
    end = len(env.gt_bboxes)
    predicted_list = [init_gt]

    env.state = init_gt
    while e < end-1:

        # initialize the env
        env.cur_img = env.imglist[e+1]
        env.cur_idx = e+1

        # predict full episode
        print('tracking %d frame...' % e)
        while True:
            bbox_img = env.cur_img.crop(env.state)
            bbox_img = Variable(R.transform(bbox_img)).cuda()
            bbox_img = bbox_img.unsqueeze(0)
            q_value = R.agent(bbox_img).view(-1)
            action = torch.argmax(q_value)
            next_bbox_img, is_t, r = env.step(action)
            if is_t:
                break
        e = e + 1
        predicted_list.append(env.state)

    return predicted_list


def main():
    # create model
    args = parse_arguments()
    env = Env(args)
    R = Reinforce()

    # load weights
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        R.agent.load_state_dict(checkpoint['state_dict'])
        R.optim.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

    start_epoch = args.num_train + 1
    predicted_list = test(env, R, start_epoch)

    # save result
    predicted_list = np.vstack(predicted_list)
    np.savetxt('predicted_bbox', predicted_list, delimiter=',')


if __name__ == '__main__':
    main()
