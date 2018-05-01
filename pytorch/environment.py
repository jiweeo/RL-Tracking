import PIL.Image as Image
import numpy as np

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


def calculate_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou



class Env(object):

    def __init__(self):
        self.root_dir = '../vot2016/ball1/'
        self.num_train = 50                     # number of frames used for training
        self.imglist, self.gt_bboxes= self.load_data()
        self.cur_idx = 0
        self.cur_img = None
        self.state = np.zeros(4)
        self.step_count = 0


    def load_data(self):
        '''
        :return:
            imglist list([h*w*3])
            gt_bbox nparray [n*4] (x,y,delta_x, delta_y)
        '''

        imglsit = []
        for index in range(1, self.num_train+1):
            imgpath = self.root_dir+str(index).zfill(8)+'.jpg'
            img = Image.open(imgpath)
            imglsit.append(img)

        gt_bbox = np.genfromtxt(self.root_dir+'groundtruth.txt', delimiter=',')
        x = gt_bbox[:, [0, 2, 4, 6]]
        y = gt_bbox[:, [1, 3, 5, 7]]
        x_min = np.min(x, 1)
        x_max = np.max(x, 1)
        y_min = np.min(y, 1)
        y_max = np.max(y, 1)
        gt_bbox = np.column_stack((x_min, y_min, x_max, y_max))
        gt_bbox = gt_bbox[:self.num_train, :]
        return imglsit, gt_bbox


    def reset(self):
        self.cur_idx = np.random.randint(self.num_train)    #frame idx
        self.cur_img = self.imglist[self.cur_idx]           #
        gt_bbox = self.gt_bboxes[self.cur_idx]
        self.step_count = 0
        # gaussian randomly initialize the starting location.
        h, w = self.cur_img.size
        std = 0.01 * min(h, w)

        while True:
            # keep sample until a valid starting bbox
            dx, dy, dxx, dyy = np.random.normal(0, std, size=4)
            self.state = gt_bbox + np.array([dx, dy, dxx, dyy])
            if self.is_valid(self.state):
                break

        return self.cur_img.crop(self.state)


    def is_valid(self, bbox):
        # check out of range error
        if bbox[0]>=bbox[2] or bbox[1]>=bbox[3]:
            return False
        if min(bbox) < 0:
            return False
        if max([bbox[0], bbox[2]]) >= self.cur_img.size[0]:
            return False
        if max([bbox[1], bbox[3]]) >= self.cur_img.size[1]:
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
        h, w = self.cur_img.size
        # step_size = min(h, w) * 0.03
        step_size = 0.03

        # compute new bbox
        new_bbox = self.state + warp[action] * step_size * self.state

        # check if the new bbox is valid
        if not self.is_valid(new_bbox):
            # return current bbox and Termination
            return self.cur_img.crop(self.state), True, -1

        # if valid
        self.state = new_bbox
        self.step_count += 1
        ns = self.cur_img.crop(self.state)
        if action == 10 or self.step_count == 300:
            # if curruent action is termination or the episode is long enough
            is_t = True
        else:
            is_t = False

        # computing reward
        reward = 0
        if is_t:
            iou = calculate_iou(self.state, self.gt_bboxes[self.cur_idx])
            if iou > 0.7:
                reward = 1
            else:
                reward = -1

        return ns, is_t, reward



