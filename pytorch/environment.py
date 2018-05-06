import PIL.Image as Image
import numpy as np
from utils import calculate_iou, gen_gaussian_noise, cropping
import os

warp = np.array(
    [
        [-1, 0, -1, 0],  # left
        [1, 0, 1, 0],  # right
        [0, -1, 0, -1],  # up
        [0, 1, 0, 1],  # down
        [-2, 0, -2, 0],  # 2 left
        [2, 0, 2, 0],  # 2 right
        [0, -2, 0, -2],  # 2 up
        [0, 2, 0, 2],  # 2 down
        [-1, -1, 1, 1],  # enlarge
        [1, 1, -1, -1],  # shrink
        [0, 0, 0, 0]  # terminate
    ]
)


class Env(object):

    def __init__(self, args):
        self.img = None
        self.state = np.zeros(4)
        self.step_count = 0


    def reset(self, img, gt_bbox, state=None):
        self.img = img
        self.gt_bbox = gt_bbox
        self.step_count = 0

        if state is None:
            # then sample a state
            while True:
                # keep sample until a valid starting bbox
                self.state = gen_gaussian_noise(gt_bbox)
                if self.is_valid(self.state):
                    break
        else:
            self.state = state

        return cropping(img, self.state)

    def is_valid(self, bbox):
        # check out of range error
        if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
            return False
        if min(bbox) < 0:
            return False
        if max([bbox[0], bbox[2]]) >= self.img.size[0]:
            return False
        if max([bbox[1], bbox[3]]) >= self.img.size[1]:
            return False
        return True

    def step(self, action):

        '''
        :param action: int, range[0, 10]
        :return:
            new_state (img)
            is_terminate,
            reward
        '''

        # calculate step size
        w = self.state[2]-self.state[0]
        h = self.state[3]-self.state[1]
        step_size = 1
        # compute new bbox
        new_bbox = self.state + warp[action] * step_size

        # check if the new bbox is valid
        if not self.is_valid(new_bbox):
            # return current bbox and Termination
            return cropping(self.img, self.state), True, -1

        # if valid
        self.state = new_bbox
        self.step_count += 1
        ns = cropping(self.img, self.state)
        if action == 10 or self.step_count == 100:
            # if curruent action is termination or the episode is long enough
            is_t = True
        else:
            is_t = False

        # computing reward
        reward = 0
        if is_t:
            iou = calculate_iou(self.state, self.gt_bbox)
            if iou > 0.7:
                reward = 100
            else:
                reward = -1

        return ns, is_t, reward



