import torch
import numpy as np
import sys
import argparse
from environment import Env
from Network import Tracknet
from torchvision import transforms
from torch.autograd import Variable
import torchvision
import os

os.environ['CUDA_VISIBLE_DEVICES']='3'

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

    def train(self, env, epoch, gamma=0.99, logging=False):
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

        loss = -loss/horizon
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
            if is_training:
                action = np.random.choice(11, 1, p=q_value_numpy)[0]
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
    parser.add_argument('--gamma', type=float, default=0.99)
    return parser.parse_args()


def main(args):
    args = parse_arguments()
    R = Reinforce()
    env = Env()
    for epoch in range(1, args.max_epochs+1):
        R.train(env, epoch, args.gamma, logging=True)


if __name__ == '__main__':
    main(sys.argv)