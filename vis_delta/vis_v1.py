import open3d as o3d
import os

print("Read Redwood dataset")
color_raw = o3d.io.read_image(
    r"D:\data_collection_1\dataset_agv1_moving\realsense\color\000226.jpg"
)
depth_raw = o3d.io.read_image(
    r"D:\data_collection_1\dataset_agv1_moving\realsense\depth\000226.png"
)
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)
print(rgbd_image)

pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image,
    o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
    ),
)
# Flip it, otherwise the pointcloud will be upside down
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
o3d.visualization.draw_geometries([pcd])

# o3d.camera.PinholeCameraIntrinsic, zoom=0.5
