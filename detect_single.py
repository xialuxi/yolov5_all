# from utils.datasets import *
# from utils.utils import *
import torch
import cv2
import numpy as np
import time
import random
import glob
import os

from utils.augmentations import letterbox
from utils.general import non_max_suppression,xyxy2xywh,scale_coords

cuda = True
device = torch.device('cuda:0' if cuda else 'cpu')

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()
    
    
def get_model(weights):
    #fuse conv_bn and repvgg
    model = torch.load(weights, map_location=device)['model'].float().fuse_model().eval()
    #only fuse conv_bn
    #model = torch.load(weights, map_location=device)['model'].float().fuse().eval()
    return model

def process_img(orgimg):
    import copy
    image = copy.deepcopy(orgimg)
    img = letterbox(image, new_shape=(416,416), auto=False)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img

def show_results(img, xywh, class_num, conf=0.4):
    h,w,c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
    y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
    x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
    y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)
    color = (0, 0, 255)
    cv2.rectangle(img, (x1,y1), (x2, y2), color, thickness=tl+2, lineType=cv2.LINE_AA)
    label = str(int(class_num)) + ' : ' + str(round(float(conf), 4))
    cv2.putText(img, label, (x1, y1 - 2), 0, tl , [225, 255, 255], thickness=tl+2, lineType=cv2.LINE_AA)
    return img


def detect(model, image, conf_thres, iou_thres):

    #img
    #h, w, c = image.shape
    #h_4, w_4 = h //4, w // 4
    #image = image[:, 240:1680, :]
    #t_img = np.ones((1440, 1920, 3), dtype=np.uint8)
    #t_img[:, :, :] = 114
    #t_img[180:1260, :, :] = image
    #image = t_img

    img = process_img(image)
    #print(img.shape)
    pred = model(img)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres)
    for i, det in enumerate(pred):  # detections per image
        gn = torch.tensor(image.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()
            # Write results
            for *xyxy, conf, cls in det:
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                image = show_results(image, xywh, cls, conf)
    return image




def detect_video(model, path, save_path = None):
    cv2.namedWindow("video",cv2.WINDOW_NORMAL)
    conf_thres, iou_thres = 0.4, 0.4
    capture = cv2.VideoCapture(path)
    fps = capture.get(cv2.CAP_PROP_FPS)
    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    #size = (1920, 1440)
    print('video fram size: ', size)
    save = False
    if save_path is not None:
        save = True
        print('save video to ', save_path)
        videoWriter = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, size)


    if capture.isOpened():
        while True:
            ret, frame = capture.read()
            if ret == True:
                frame = detect(model, frame, conf_thres, iou_thres)
                cv2.imshow('video', frame)
                if save:
                    videoWriter.write(frame)  # 写视频帧
            else:
                break
            if cv2.waitKey(1) == ord('q'):
                break
    cv2.destroyAllWindows()
    
    
def detect_image(model, path, save_path = None):
    conf_thres, iou_thres = 0.4, 0.4
    cv2.namedWindow("image",cv2.WINDOW_NORMAL)
    frame = cv2.imread(path)
    frame = detect(model, frame, conf_thres, iou_thres)
    if save_path is not None:
        cv2.imwrite(save_path, frame)
    cv2.imshow('image', frame)
    if cv2.waitKey(0) == ord('q'):
        cv2.destroyAllWindows()


if __name__ == '__main__':
    #detect_test()
    weights = './runs/train/exp48/weights/last.pt'
    model = get_model(weights)

    video_path = '../sample/20211028_20211028180130_20211028181132_180131.mp4'
    save_path = '../sample/save.mp4'
    #detect_video(model, video_path, save_path)

    image_path = './43eb0e68965711513412c4b051051770.JPG'
    detect_image(model, image_path)


