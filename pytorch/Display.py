import os
import cv2
import re

def display(img_path, bd_path, video_name):
    if os.path.exists(os.path.join(os.getcwd(),video_name)):
        os.remove(os.path.join(os.getcwd(),video_name))
    frame = cv2.imread(os.path.join(img_path, os.listdir(img_path)[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, -1, 15, (width, height))
    with open(bd_path,"r") as f:
        dbs = f.readlines()
    for i,file in enumerate(os.listdir(img_path)):
        img = cv2.imread(os.path.join(img_path,file))
        db = re.split(r'[ ,|;\t"]+', dbs[i][:-1])
        db = list(map(int, db))
        pt1 = (db[0],db[1])
        pt2 = (db[0]+db[2],db[1]+db[3])
        cv2.rectangle(img, pt1, pt2, (0,255,0),1)
        video.write(img)
    cv2.destroyAllWindows()
    video.release()



if __name__=="__main__":
    path = "data/Crossing/img"
    video_name = "test_rect_cross.mp4"
    bd_path = "data/Crossing/groundtruth_rect.txt"

    # path = "data/Biker/img"
    # video_name = "test_rect_bike.mp4"
    # bd_path = "data/Biker/groundtruth_rect.txt"

    display(path,bd_path,video_name)