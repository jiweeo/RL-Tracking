import torch
import numpy as np
import argparse
from environment import Env
from Network import Tracknet
from torch.nn import functional as F
import os
import shutil
import Dataset
from utils import get_transform
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

os.environ['CUDA_VISIBLE_DEVICES']='3'


def save_checkpoint(state, dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    torch.save(state, os.path.join(dir, 'latest.pth'))


class Reinforce(object):
    def __init__(self, dataloader, transforms):
        super().__init__()
        # net
        self.agent = Tracknet().cuda()
        self.optim = torch.optim.Adam([{'params': self.agent.fc_a.parameters()},
                                       {'params': self.agent.fc_o.parameters()}], lr=1e-4)
        self.agent.train()
        self.dataloader = dataloader
        self.transforms = transforms

    def train(self, env, epoch, train_loader, gamma=0.99, logging=False):

        for i, (img, gt_bbox) in enumerate(self.dataloader):
            # pre-process
            to_pil = transforms.ToPILImage()
            img = img.squeeze(0).numpy()
            img = to_pil(img)
            gt_bbox = gt_bbox.squeeze(0).numpy()

            states, actions, rewards, q_values = self.generate_episode(env, img, gt_bbox)

            horizon = len(states)

            # compute return G_t and loss
            g = np.zeros(horizon)
            loss = 0
            for t in range(horizon-1, -1, -1):
                if t == horizon-1:
                    g[t] = rewards[t]
                else:
                    g[t] = gamma * g[t+1] + rewards[t]

                loss += torch.log(q_values[t]) * g[t]

            loss = - loss/horizon
            self.optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.agent.parameters(), max_norm=1)
            self.optim.step()
            if logging:
                print('E: %d [%d/%d]\t loss: %.3f\t reward:%.3f \thorizon: %d' %(epoch, i, len(self.dataloader), loss, sum(rewards), horizon))
            if i>500:
                break

    def generate_episode(self, env, img, gt_bbox):
        states = []
        actions = []
        rewards = []
        q_values = []
        patch = env.reset(img, gt_bbox)
        while True:
            states.append(patch)        # state is a patch

            # feed to net
            patch_var = (self.transforms(patch)).cuda()
            patch_var = patch_var.unsqueeze(0)
            q_value, stop = self.agent(patch_var)
            q_value = q_value.view(-1)
            stop = stop.view(-1)
            q_value = F.softmax(q_value, dim=0)
            stop = F.softmax(stop, dim=0)
            stop_numpy = stop.data.cpu().numpy()
            q_value_numpy = q_value.data.cpu().numpy()

            # fusion of exploit and explore
            alpha = 0
            exploration = np.ones(11) / 11
            prob = q_value_numpy*(1-alpha) + alpha * exploration

            action_o = np.random.choice(2, 1, p=stop_numpy)
            if action_o == 1:
                # stop
                action = 10
                q_values.append(stop[1])
            else:
                # not stopping, sample an action
                action = np.random.choice(11, 1, p=prob / prob.sum())[0]
                q_values.append(q_value[action])

            ns, is_t, reward = env.step(action)
            patch = ns

            actions.append(action)
            rewards.append(reward)


            if is_t:
                break

        return states, actions, rewards, q_values

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--max_epochs', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--data', type=str, default='../data/test/vot2017/ball1')
    parser.add_argument('--train_data', type=str, default='../data/train/')
    parser.add_argument('--test_data', type=str, default='../data/test/')
    parser.add_argument('--num_train', type=int, default=50, help='number of frames used for training in one video')
    parser.add_argument('--resume', type=str, default='',  help='path to checkpoint (default: none)')
    parser.add_argument('--load', type=str, default='', help='path to checkpoint(default: none)')
    parser.add_argument('--init_sl', type=str, default='cv/run2/latest.pth', help='path to checkpoint(default: none)')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--name', type=str, default='rl')
    parser.add_argument('--num_workers', type=int, default=8)

    return parser.parse_args()


def main():
    args = parse_arguments()

    # dataset, dataloader
    transforms = get_transform()
    train_dataset = Dataset.TrackData_RL(args.train_data, transform=transforms)
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, shuffle=True, batch_size=1)

    # model, environment
    R = Reinforce(train_loader, transforms)
    env = Env(args)

    start_epoch = 1

    if args.init_sl:
        if os.path.isfile(args.init_sl):
            print("=> loading checkpoint '{}'".format(args.init_sl))
            checkpoint = torch.load(args.init_sl)
            R.agent.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.init_sl))
    elif args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch= checkpoint['epoch']
            R.agent.load_state_dict(checkpoint['state_dict'])
            R.optim.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    for epoch in range(start_epoch, args.max_epochs+1):
        R.train(env, epoch, args.gamma, logging=True)
        if epoch % args.save_freq == 0:
            # save model
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': R.agent.state_dict(),
                'optimizer': R.optim.state_dict(),
            }, dir='cv/%s/' % args.name)


if __name__ == '__main__':
    main()
