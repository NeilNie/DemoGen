import numpy as np
from scipy.spatial.transform import Rotation as R


robot_2_cam = np.array([
    [0.4919347656212202, -0.49046766542986825, 0.7193475757325765, 0.028804696241265915],
    [-0.8691879149480207, -0.22891099007933832, 0.43832811546573464, -0.3695261485093276],
    [-0.05031824266161347, -0.8408607667659251, -0.5389074117801784, 0.6030444321235613],
    [0.0, 0.0, 0.0, 1.0]
])
ROBOT2CAM_POS = robot_2_cam[:3, 3]
ROBOT2CAM_QUAT_INITIAL = R.from_matrix(robot_2_cam[:3, :3]).as_quat()

OFFSET_POS=np.array([0.0, 0.0, 0.0])
OFFSET_ORI_X=R.from_euler('x', 0, degrees=True)
OFFSET_ORI_Y=R.from_euler('y', 0, degrees=True)
OFFSET_ORI_Z=R.from_euler('z', 0, degrees=True)

ROBOT2CAM_POS = ROBOT2CAM_POS + OFFSET_POS
ori = R.from_quat(ROBOT2CAM_QUAT_INITIAL) * OFFSET_ORI_X * OFFSET_ORI_Y * OFFSET_ORI_Z
ROBOT2CAM_QUAT = ori.as_quat()

robot2cam_mat = np.eye(4)
robot2cam_mat[:3, :3] = R.from_quat(ROBOT2CAM_QUAT).as_matrix()
robot2cam_mat[:3, 3] = ROBOT2CAM_POS

REALSENSE_SCALE = 1

intrinsic_matrix = np.array([
    [610.0545654296875, 0.0, 316.09674072265625],
    [0.0, 609.664306640625, 240.20550537109375],
    [0.0, 0.0, 1.0]]
)

T_link2viz = np.eye(4)
transform_realsense_util = np.eye(4)
image_size = (640, 480)