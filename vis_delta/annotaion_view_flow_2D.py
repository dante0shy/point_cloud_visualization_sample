import json,os,glob,cv2
import numpy as np
import open3d as o3d
import uv2color

images = glob.glob(r'D:\data_collection_3\dataset_chair_fixed\realsense\color\*')
images = list(sorted(images, key=lambda x: int(x.split('.')[0].split('\\')[-1])))#[5:]
images = {x.split('.')[0].split('\\')[-1]: x for x in images}

masks = glob.glob(r'D:\rescource\2D_flow\chair\inference\run.epoch-0-flow-field\*')
masks = list(sorted(masks, key=lambda x: int(x.split('\\')[-1].split('.')[0])))#[5:]

# masks = list(sorted(pts, key=lambda x: int(x.split('.')[0].split('\\')[-1])))
# masks_a = [x for x in masks if '-pc1.' in x]
# masks_a = list(sorted(masks_a, key=lambda x: int(x.split('.')[0].split('\\')[-1].split('-')[0])))
# masks_b = [x for x in masks if '-pc2.' in x]
# masks_b = list(sorted(masks_b, key=lambda x: int(x.split('.')[0].split('\\')[-1].split('-')[0])))
# masks_flow = [x for x in masks if '-depth.' in x]
# masks_flow  = list(sorted(masks_flow, key=lambda x: int(x.split('.')[0].split('\\')[-1].split('-')[0])))
# masks = zip(masks_a,masks_b,masks_flow)



def read_flo_file(filename, memcached=False):
    """
    Read from Middlebury .flo file
    :param flow_file: name of the flow file
    :return: optical flow data in matrix
    """
    if memcached:
        filename = io.BytesIO(filename)
    f = open(filename, 'rb')
    magic = np.fromfile(f, np.float32, count=1)[0]
    data2d = None

    if 202021.25 != magic:
        print('Magic number incorrect. Invalid .flo file')
    else:
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        data2d = np.fromfile(f, np.float32, count=2 * w * h)
        # reshape data into 3D array (columns, rows, channels)
        data2d = np.resize(data2d, (h, w, 2))
    f.close()
    return data2d

def trans(pt):
    pt[:, 1] *= -1
    pt[:, 2] *= -1
    # return
for i,imgs in enumerate(masks):
    # if i <65:
    #     continue
    pt_1 = read_flo_file(imgs)
    # tmp = np.zeros([pt_1.shape[0],pt_1.shape[1],3])
    # tmp[:,:,:2] = pt_1
    img2d = cv2.imread(list(images.values())[i])
    tmp = cv2.resize(uv2color.flow_to_color(pt_1, convert_to_bgr=True),(img2d.shape[1],img2d.shape[0],))
    tmp_flow = np.vstack((img2d,tmp))
    cv2.imshow('a',cv2.resize(tmp_flow,(img2d.shape[1]//2,img2d.shape[0],)))
    print('{}: {}-{}'.format(i,tmp.shape,img2d.shape))
    cv2.waitKey()
    pass


pass