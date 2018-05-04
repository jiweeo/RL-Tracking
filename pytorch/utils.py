import os
import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 1
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


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

def gen_gaussian_noise(bbox, alpha=0.18):
    '''
    :param bbox: (4,) nparry
    :return: bbox with gausian noise
    '''

    length = min([bbox[3]-bbox[1], bbox[2]-bbox[0]])
    std = length * alpha
    dx, dy, dxx, dyy = np.random.normal(0, std, size=4)
    noisy_bbox = bbox + np.array([dx, dy, dxx, dyy])

    return noisy_bbox

def gen_label(root_dir):
    '''
    generate gt_bbox and cls for supervised learning
    :return:

    '''


    imglist_path = os.path.join(root_dir, 'imglist')

    f = open(imglist_path, 'r')
    gt_for_write = []
    lines = f.readlines()
    for line in lines:
        print(line)
        line = line[:-1]
        parent_path = os.path.dirname(os.path.join(root_dir, line))
        gt_file_path = os.path.join(parent_path, 'groundtruth.txt')
        gt_polyes = np.loadtxt(gt_file_path, delimiter=',', dtype=float)
        str_frame_id = line.split('/')[-1]
        frame_id = int(str_frame_id[:-4])
        gt_poly= gt_polyes[frame_id-1]

        # polygons --> box
        x_value = [gt_poly[0], gt_poly[2], gt_poly[4], gt_poly[6]]
        y_value = [gt_poly[1], gt_poly[3], gt_poly[5], gt_poly[7]]
        x = min(x_value)
        xx = max(x_value)
        y = min(y_value)
        yy = max(y_value)

        gt_for_write.append([x, y, xx, yy])

    gt_for_write = np.vstack(gt_for_write)
    np.savetxt(os.path.join(root_dir, 'target'), gt_for_write, fmt='%.3f')


if __name__ == '__main__':
    gen_label('../data/train/')