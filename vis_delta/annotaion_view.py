import json
import numpy as np
import cv2

image = cv2.imread(r'D:\data_collection_1\dataset_person\realsense\color\000004.jpg')
# annotation = json.load(open(r'D:\data_collection_1\dataset_agv3\000007.json'))
annotation_2 = json.load(open(r'D:\data_collection_1\dataset_person\annotation\000004.json'))
r'D:\data_collection_1\dataset_agv3\realsense\color'
# cv2.fillPoly(mask, [arr], color=(255))
mask = np.zeros([image.shape[0],image.shape[1]],dtype= np.int)
cv2.fillPoly(mask,[np.array(annotation_2['shapes'][1]['points'],dtype= np.int)],(1))

cv2.imwrite(r'D:\data_collection_1\dataset_person\annotation\000004.png',mask)
pass