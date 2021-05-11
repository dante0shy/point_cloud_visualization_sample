import json,os,glob
import numpy as np
import open3d as o3d

images = glob.glob(r'D:\data_collection_1\flow_res_w\realsense\color\*')
images = list(sorted(images, key=lambda x: int(x.split('.')[0].split('\\')[-1])))#[5:]
images = {x.split('.')[0].split('\\')[-1]: x for x in images}

masks = glob.glob(r'D:\data_collection_1\demo_result\flow_res_agv_fixed_1\*')
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
    # if i <65:
    #     continue
    pt_1 = np.load(open(imgs[0],'rb'))
    pt_2 = np.load(open(imgs[1],'rb'))
    flow = np.load(open(imgs[2],'rb'))
    # pt_1[:,1] *=-1
    # pt_1[:,2] *=-1
    # pt_1[:,0] += pt_1[:,1]
    # pt_1[:,1] = pt_1[:,0] -  pt_1[:,1]
    # pt_1[:,0] = pt_1[:,0] -  pt_1[:,1]
    pt_2_f = pt_1+flow
    pt_2_f[:,1] += 4
    pt_2[:,1] += 8

    trans(pt_1)
    trans(pt_2_f)
    trans(pt_2)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack((pt_1,pt_2_f,pt_2)))#np.vstack((pt_1,pt_2_f,pt_2)）
#np.vstack((pt_1,pt_2_f,pt_2))

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
    print(i)
    del vis
    del pcd

    pass


pass