import open3d as o3d
import os,glob
import numpy as np
from numpy.linalg import inv

def transform(scan, poss_b):
    tmp = np.zeros([scan.shape[0],4])
    tmp[:,:3] = scan
    tmp[:,3] = 1
    diff = np.matmul(inv(poss_b[1]), poss_b[0])
    tmp = np.matmul(diff, tmp.T).T
    return tmp[:,:3]

if __name__ == '__main__':
    base_dir = r"D:\lidar_nuscene"
    datas = glob.glob(os.path.join(base_dir,'*'))
    lidar = [x for x in datas if 'pose' not in x]
    pose = [x for x in datas if 'pose' in x]

    date = zip(sorted(lidar),sorted(pose))
    merged = None
    pose_p = None

    for i,d in enumerate(date):

        xyz_r = np.load(open(d[0],'rb'))
        pose = np.load(open(d[1],'rb'))
        pcd = o3d.geometry.PointCloud()
        if i:
            merged = transform(merged,[pose_p,pose])
            # tmp = np.copy(merged)
            # tmp[:,0] += 500
            # tmp = np.vstack((tmp, xyz_r[:,:3]))
            merged = np.vstack((merged, xyz_r[:,:3]))

            pcd.points = o3d.utility.Vector3dVector(merged)
            # merged = np.vstack((merged, xyz_r[:,:3]))
            pose_p = pose

        else:

            pcd.points = o3d.utility.Vector3dVector(xyz_r[:,:3])
            merged = xyz_r[:, :3]
            pose_p = pose


        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.run()
        vis.destroy_window()

        del vis
        del pcd