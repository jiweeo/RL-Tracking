import os
import cv2
import re
import argparse
import numpy as np

def display(img_path, bd_path, video_name):
    if os.path.exists(os.path.join(os.getcwd(), video_name)):
        os.remove(os.path.join(os.getcwd(), video_name))
    frame = cv2.imread(os.path.join(img_path, os.listdir(img_path)[0]))
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    video = cv2.VideoWriter(video_name, fourcc, 30, (width, height), True)
    print(video.isOpened())
    # with open(bd_path, "r") as f:
    #     dbs = f.readlines()

    dbs = np.loadtxt(args.rect, delimiter=',')
    imglist = np.loadtxt(os.path.join(img_path, 'images.txt'), dtype=np.str)

    for i in range(len(imglist)):
        file = imglist[i]
        img = cv2.imread(os.path.join(img_path, file))
        db = dbs[i].astype(np.int)
        pt1 = (db[0], db[1])
        pt2 = (db[2], db[3])
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        video.write(img)
    video.release()
    cv2.destroyAllWindows()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='../data/test/vot2017/car1', help='path to test frames')
    parser.add_argument('--video_name', type=str, default='result.mp4', help='name of output video file')
    parser.add_argument('--rect', type=str, default='predicted_bbox', help='path to rects file')
    return parser.parse_args()


if __name__=="__main__":

    # path = "data/Biker/img"
    # video_name = "test_rect_bike.mp4"
    # bd_path = "data/Biker/groundtruth_rect.txt"
    args = parse_arguments()
    display(args.path, args.rect, args.video_name)