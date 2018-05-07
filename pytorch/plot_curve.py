import os
import numpy as np
from utils import calculate_iou
import matplotlib.pyplot as plt

def main():
    root_dir = '../data/OTB100'
    list = os.listdir(root_dir)
    iou_list = []

    for name in list:
        pred_path = os.path.join(root_dir, name, 'pred_rect_sl2.txt')
        gt_path = os.path.join(root_dir, name, 'groundtruth_rect.txt')
        if os.path.isfile(pred_path):
            print (name)
            pred_bbox = np.loadtxt(pred_path)
            gt_bbox = np.genfromtxt(gt_path, delimiter=',')
            if len(gt_bbox.shape) == 1:
                gt_bbox = np.genfromtxt(gt_path)

            for i in range(len(pred_bbox)):
                pb = pred_bbox[i, :]
                gb = gt_bbox[i, :]
                gb[2] = gb[0] + gb[2]
                gb[3] = gb[1] + gb[3]
                iou = calculate_iou(pb, gb)
                iou_list.append(iou)
        else:
            continue
    iou_list.sort()
    iou_list = np.array(iou_list)
    total = len(iou_list)
    precicion = np.zeros(6)
    for i in range(6):
        thresh = i * 0.2
        precicion[i] = (iou_list >= thresh).sum()/total

    print(precicion)
    x = np.arange(0, 1.2, 0.2)
    plt.plot(x, precicion, 'r--')
    t = plt.xlabel('IoU [AUC: %.3f]' % precicion.mean(), fontsize=14, color='black')
    t = plt.ylabel('success_rate', fontsize=14, color='black')
    plt.show()

    
if __name__ == '__main__':
    main()