import numpy as np
from scipy.spatial.transform import Rotation as R

# arm to world
arm_base_T = np.array([
    [-0.9998666111795577, -0.005214134787925708, -0.016084222515222367, 0.4961327418151297],
    [0.005179609437566609, -0.999993835407142, 0.0021846504383549866, -0.3148368999205948],
    [-0.016096495623723783, 0.0021016687657644775, 0.9998682786596049, -0.048373076122982195],
    [0.0, 0.0, 0.0, 1.0]
])

# TODO
# cam to world
robot_2_cam = np.array([
    [0.023677974939346435, 0.41928422451019287, -0.9075461030006409, 0.747499482412909],
    [0.999719500541687, -0.009647518396377504, 0.021625697612762614, 0.008440321225613855],
    [0.00031167268753064, -0.9078036546707153, -0.4193951487541199, 0.587605608926162],
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