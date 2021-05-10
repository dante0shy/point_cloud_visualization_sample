import json,os,glob
import numpy as np
import cv2

images = glob.glob(r'D:\data_collection_1\dataset_agv3\realsense\color\*')
images = list(sorted(images, key=lambda x: int(x.split('.')[0].split('\\')[-1])))#[5:]
images = {x.split('.')[0].split('\\')[-1]: x for x in images}

masks = glob.glob(r'D:\data_collection_1\demo_result\dataset_agv3_x\*')
# masks = list(sorted(masks, key=lambda x: int(x.split('.')[0].split('\\')[-1])))
masks_a = [x for x in masks if '-a.' in x]
masks_a = list(sorted(masks_a, key=lambda x: int(x.split('.')[0].split('\\')[-1].split('-')[0])))
masks_b = [x for x in masks if '-b.' in x]
masks_b = list(sorted(masks_b, key=lambda x: int(x.split('.')[0].split('\\')[-1].split('-')[0])))
masks = zip(masks_a,masks_b)

for i,imgs in enumerate(masks):
    img_a = cv2.imread(imgs[0], cv2.IMREAD_UNCHANGED)
    img_b = cv2.imread(imgs[1], cv2.IMREAD_UNCHANGED)
    ori_a = cv2.imread(images[imgs[0].split('.')[0].split('\\')[-1].split('-')[0]], cv2.IMREAD_UNCHANGED)
    ori_b = cv2.imread(images[imgs[1].split('.')[0].split('\\')[-1].split('-')[1]], cv2.IMREAD_UNCHANGED)
    img_a = cv2.resize(img_a, (1280, 720)) #*255
    img_b = cv2.resize(img_b, (1280, 720)) #*255
    size_a = ori_a.shape[:2]
    size_b = ori_b.shape[:2]

    img = np.vstack((img_a, img_b))
    ori = np.vstack((ori_a, ori_b))
    of = cv2.bitwise_and(ori, ori, mask=img)
    # o_f = cv2.resize(np.hstack((frame,of)), (frame.shape[1],frame.shape[0]//2))

    # cv2.imshow('out',  cv2.resize(of, (of.shape[1]//2,of.shape[0]//2)))
    # cv2.waitKey()
    # cv2.waitKey()
    cv2.imwrite(r'D:\PyProjects\vis_v1\extras\demo_delta_agv_x\{:06d}.jpg'.format(i),of)
    pass


pass