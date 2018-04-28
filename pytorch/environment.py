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
        self.num_train = 50 # number of frames used for training
        self.imglist, self.gt_bboxes= self.load_data()
        self.cur_idx = 0
        self.cur_img = None
        self.state = np.zeros(4)
        self.gt_bbox = np.zeros(4)
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
        gt_bbox = [x_min, y_min, x_max, y_max]
        gt_bbox = gt_bbox[:self.num_train, :]
        return imglsit, gt_bbox

    def cropping(self, state):
        h, w, _ = self.cur_img.shape
        x = np.max(0, state[0])
        y = np.max(0, state[1])
        xx = np.min(state[2], h-1)
        yy = np.min(state[3], w-1)
        return self.cur_img.crop([x, y, xx, yy])


    def reset(self):
        self.cur_flame = np.random.randint(self.num_train)
        self.cur_img = self.imglist[self.cur_flame]
        self.gt_bbox = self.gt_bboxes
        self.step_count = 0
        # gaussian randomly initialize the starting location.
        std_x = 0.1 * self.gt_bbox[2]
        std_y = 0.1 * self.gt_bbox[3]
        dx = np.random.normal(0, std_x)
        dy = np.random.normal(0, std_y)
        self.state = self.gt_bbox + [dx, dy, 0, 0]
        return self.cropping(self.state)

    def step(self, action):

        '''
        :param action: int, range[0, 10]
        :return:
            new_state (img)
            is_terminate,
            reward
        '''
        self.state += warp[action]
        self.step_count += 1

        ns = self.cropping()
        if action == 10 or self.step_count == 100:
            is_t = True
        else:
            is_t = False

        reward = 0
        if is_t:
            iou = calculate_iou(self.state, self.gt_bbox)
            if iou > 0.7:
                reward = 1
            else:
                reward = -1

        return ns, is_t, reward



