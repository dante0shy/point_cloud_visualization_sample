import json,os,glob
import numpy as np
import cv2

images = glob.glob(r'D:\data_collection_1\dataset_person\realsense\color\*')
# images = glob.glob(r'D:\data_collection_1\dataset_agv3\realsense\color\*')
images = list(sorted(images, key=lambda x: int(x.split('.')[0].split('\\')[-1])))[5:]

masks = glob.glob(r'D:\data_collection_1\demo_result\dataset_person\*')
# masks = glob.glob(r'D:\data_collection_1\demo_result\dataset_agv3\*')
masks = list(sorted(masks, key=lambda x: int(x.split('.')[0].split('\\')[-1])))
for i,img in enumerate(images):
    frame = cv2.imread(img)
    m = cv2.imread(masks[i])[:,:,2:]#.astype(np.int)
    # of = frame * m
    of = cv2.bitwise_and(frame, frame, mask=m)
    o_f = cv2.resize(np.hstack((frame,of)), (frame.shape[1],frame.shape[0]//2))

    cv2.imshow('out', o_f)
    # cv2.waitKey()
    cv2.waitKey(30)
    cv2.imwrite(r'D:\PyProjects\vis_v1\extras\demo_delta_person\{}.jpg'.format(int(img.split('.')[0].split('\\')[-1])),o_f)
    pass


pass