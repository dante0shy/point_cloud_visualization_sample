import json,os,glob

import cv2
import numpy as np
import open3d as o3d
import uv2color
import sklearn.neighbors as skn
import sklearn.cluster as skc
FOCALLENGTH = 640
CENTERX = 640
CENTERY = 360
SCALINGFACTOR = 2.0

COLOR_DICT =np.array([
    [255,0,0],
    [0,255,0],
    [0,0,255],
    [128,128,0],
    [120,180,120],
    [200,180,0],
    [255,0,128],
    [0,255,128],
    [128,0,255],
    [128,128,255],
    [255,255,120],
    [120,120,0],
    [0,128,200],
])


def uv2xyz(rgb,depth):
    depth = (depth - 0) * 1.0 / (np.max(depth.flatten()) - 0)

    points = []
    for v in range(rgb.shape[0]):
        for u in range(rgb.shape[1]):
            color = rgb[v, u]
            # Z = depth.getpixel((u,v)) / scalingFactor
            Z = depth[v, u] * 1.0 / SCALINGFACTOR
            if Z == 0:
                continue
            X = (u - CENTERX) * Z / FOCALLENGTH
            Y = (v - CENTERY) * Z / FOCALLENGTH
            points.append([X, Y, Z] + list(color))
    return np.array(points)*10

images = glob.glob(r'D:\data_collection_3\dataset_agv_fixed2\realsense\color\*')
images = list(sorted(images, key=lambda x: int(x.split('.')[0].split('\\')[-1])))#[5:]
images = {x.split('.')[0].split('\\')[-1]: x for x in images}

depths = glob.glob(r'D:\data_collection_3\dataset_agv_fixed\realsense\depth\*')
depths = list(sorted(depths, key=lambda x: int(x.split('.')[0].split('\\')[-1])))#[5:]
depths = {x.split('.')[0].split('\\')[-1]: x for x in depths}

masks_2d = glob.glob(r'D:\rescource\2D_flow\agv\inference\run.epoch-0-flow-field\*')
masks_2d = list(sorted(masks_2d, key=lambda x: int(x.split('\\')[-1].split('.')[0])))#[5:]
masks_2d = {x.split('\\')[-1].split('.')[0]: x for x in masks_2d}

masks  = glob.glob(r'D:\data_collection_1\demo_result\flow_res_agv_fixed\*')
# masks = list(sorted(pts, key=lambda x: int(x.split('.')[0].split('\\')[-1])))
masks_a = [x for x in masks if '-pc1.' in x]
masks_a = list(sorted(masks_a, key=lambda x: int(x.split('.')[0].split('\\')[-1].split('-')[0])))
masks_b = [x for x in masks if '-pc2.' in x]
masks_b = list(sorted(masks_b, key=lambda x: int(x.split('.')[0].split('\\')[-1].split('-')[0])))
masks_flow = [x for x in masks if '-depth.' in x]
masks_flow  = list(sorted(masks_flow, key=lambda x: int(x.split('.')[0].split('\\')[-1].split('-')[0])))
masks = zip(masks_a,masks_b,masks_flow)


def trans(pt):
    pt[:, 1] *= -1
    pt[:, 2] *= -1
    # return
for i,imgs in enumerate(masks):
    if i<20:
        continue
    print(i)

    idx =(list(images.values()))[0].split('.')[0].split('\\')[-1]
    pt_1 = np.load(open(imgs[0],'rb'))
    pt_2 = np.load(open(imgs[1],'rb'))
    flow = np.load(open(imgs[2],'rb'))

    img2d = images[idx]
    img2d = cv2.imread(img2d)
    # cv2.imshow('a',img2d)
    # cv2.waitKey()
    depth = depths[idx]
    depth = cv2.imread(depth,-1)

    mask_2d = masks_2d[idx]
    mask_2d =cv2.resize(uv2color.read_flo_file(mask_2d), (img2d.shape[1],img2d.shape[0],))

    # pc_t = uv2xyz(img2d,depth)
    # pc_t_f = uv2xyz(mask_2d,depth)
    # tree_pc = skn.KDTree(pc_t[:,:3])
    # tree_pc_f = skn.KDTree(pc_t_f[:,:3])

    # s_pc = pc_t[tree_pc.query(pt_1,k =3,return_distance= False)]
    # s_pc_f = pc_t_f[tree_pc_f.query(pt_1,k =3,return_distance= False)]
    # s_f = s_pc_f[:,:,3:].mean(-2)
    from sklearn.metrics.pairwise import euclidean_distances
    cluster = skc.DBSCAN(eps=0.3, min_samples=50).fit(np.hstack((pt_1/np.abs(pt_1).max(0),flow/np.abs(flow).max(0))))#s_f,np.hstack((pt_1/np.abs(pt_1).max(0),flow/np.abs(flow).max(0)))
    # cluster = skc.MeanShift(bandwidth=5).fit(np.hstack((pt_1,flow)))#s_f,
    label = cluster.labels_
    color_pt = COLOR_DICT[label]
    # pt_2_f = pt_1
    # pt_2_f = pt_1+flow
    # pt_2_f[:,1] += 4
    # pt_2[:,1] += 8

    trans(pt_1)
    # trans(pt_2_f)
    # trans(pt_2)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pt_1)#np.vstack((pt_1,pt_2_f,pt_2)）
    pcd.colors = o3d.utility.Vector3dVector(color_pt)#np.vstack((pt_1,pt_2_f,pt_2)）
#np.vstack((pt_1,pt_2_f,pt_2))np.vstack((pt_1,pt_2_f,pt_2))

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
    del vis
    del pcd

    pass


pass