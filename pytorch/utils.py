import os
import numpy as np
from torchvision.transforms import transforms
from PIL import Image


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


def cropping(img ,bbox):
    '''
        A costumized cropping function
        return a slightly larger bounding box
    :return:
    '''
    beta = 0.15
    dx = (bbox[2] - bbox[0]) * beta
    dy = (bbox[3] - bbox[1]) * beta

    large_bbox = bbox + np.array([-dx, -dy, dx, dy])

    return img.crop(large_bbox)


def get_transform():
    transform = transforms.Compose([
        transforms.Resize((114, 114)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform


def load_data(root_dir):
    '''
    :return:
        imglist list([h*w*3])
        gt_bbox nparray [n*4] (x,y,delta_x, delta_y)
    '''

    gt_bbox = np.genfromtxt(os.path.join(root_dir, 'groundtruth.txt'), delimiter=',')
    x = gt_bbox[:, [0, 2, 4, 6]]
    y = gt_bbox[:, [1, 3, 5, 7]]
    x_min = np.min(x, 1)
    x_max = np.max(x, 1)
    y_min = np.min(y, 1)
    y_max = np.max(y, 1)
    gt_bbox = np.column_stack((x_min, y_min, x_max, y_max))

    num_frame = len(gt_bbox)

    imglist = []
    for index in range(1, num_frame + 1):
        imgpath = os.path.join(root_dir, str(index).zfill(8) + '.jpg')
        img = Image.open(imgpath)
        imglist.append(img)

    return imglist, gt_bbox


def gen_gaussian_noise(bbox, alpha=0.5):

    '''
    :param bbox: (4,) nparry
    :return: bbox with gausian noise
    '''

    length = min([bbox[3]-bbox[1], bbox[2]-bbox[0]])
    std = length * alpha
    dx, dy = np.random.normal(0, std, size=2)
    scale = np.random.normal(0, std*0.3)
    noisy_bbox = bbox + np.array([dx, dy, dx, dy])
    noisy_bbox += np.array([scale, scale, -scale, -scale])

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