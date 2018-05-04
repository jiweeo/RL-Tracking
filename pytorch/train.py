import torch
import numpy as np
import argparse
from environment import Env
from Network import Tracknet
from torchvision import transforms
from torch.autograd import Variable
import torchvision
import os
import shutil

os.environ['CUDA_VISIBLE_DEVICES']='1'


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    shutil.copyfile(filename, 'latest.pth.tar')


class Reinforce(object):
    def __init__(self):
        super().__init__()
        # net
        self.agent = Tracknet().cuda()

        # load imagenet pre-trained weights
        vgg11 = torchvision.models.vgg11(pretrained=True)
        pretrained_dict = vgg11.state_dict()
        model_dict = self.agent.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.agent.load_state_dict(model_dict)

        # trainsform
        self.transform = transforms.Compose([
            transforms.Resize((114, 114)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

        ])

        self.optim = torch.optim.Adam(self.agent.fc.parameters(), lr=1e-4)
        self.agent.train()

    def train(self, env, epoch, gamma=1.00, logging=False):
        states, actions, rewards, q_values = self.generate_episode(env)

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
        self.optim.step()

        if logging:
            print('E: %d \t loss: %.3f\t reward:%.3f \thorizon: %d' %(epoch, loss, sum(rewards), horizon))

    def generate_episode(self, env, is_training=True):
        states = []
        actions = []
        rewards = []
        q_values = []
        state = env.reset()
        while True:
            states.append(state)
            state_var = Variable(self.transform(state)).cuda()
            state_var = state_var.unsqueeze(0)
            q_value = self.agent(state_var)
            q_value = q_value.view(-1)
            q_value_numpy = q_value.data.cpu().numpy()

            exploration = np.ones(11) / 11
            alpha = 0.1

            prob = q_value_numpy*(1-alpha) + alpha * exploration
            if is_training:
                action = np.random.choice(11, 1, p=prob / prob.sum())[0]
            else:
                action = torch.argmax(q_value)

            ns, is_t, reward = env.step(action)
            state = ns
            actions.append(action)
            rewards.append(reward)
            q_values.append(q_value[action])
            if is_t:
                break
        return states, actions, rewards, q_values


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epochs', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--save_freq', type=int, default=5000)
    parser.add_argument('--data', type=str, default='../vot2016/ball1/')
    parser.add_argument('--num_train', type=int, default=50, help='number of frames used for training in one video')
    parser.add_argument('--resume', type=str, default='',  help='path to checkpoint (default: none)')
    parser.add_argument('--load', type=str, default='', help='path to checkpoint(default: none)')
    return parser.parse_args()


def main():
    args = parse_arguments()
    R = Reinforce()
    env = Env(args)

    start_epoch = 1
    if args.resume:
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
            }, filename='ckpt_%d.pth.tar' % epoch)


if __name__ == '__main__':
    main()
