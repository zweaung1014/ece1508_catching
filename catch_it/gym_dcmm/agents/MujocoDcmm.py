"""
Author: Zwe Min Htet Aung
Version@2026-10-30
ABOUT: this file contains the basic class of the fixed-base xArm6 with a hoop end-effector in the MuJoCo simulation environment.
Adapted from the Catch It codebase (MIT License, see LICENSE).
"""
import os, sys
sys.path.append(os.path.abspath('../'))
import copy
import configs.env.DcmmCfg as DcmmCfg
import mujoco
from utils.pid import PID
import numpy as np
from utils.ik_pkg.ik_arm import IKArm
import xml.etree.ElementTree as ET

# Function to convert XML file to string
def xml_to_string(file_path):
    try:
        # Parse the XML file
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Convert the XML element tree to a string
        xml_str = ET.tostring(root, encoding='unicode')

        return xml_str
    except Exception as e:
        print(f"Error: {e}")
        return None

DEBUG_ARM = False


class MJ_DCMM(object):
    """
    Class of the fixed-base arm with a hoop
    in the MuJoCo simulation environment.

    Args:
    - model: the MuJoCo model of the Dcmm
    - model_arm: the MuJoCo model of the arm (used by the IK solver)
    - viewer: whether to show the viewer of the simulation
    - object_name: the name of the object in the MuJoCo model
    - timestep: the simulation timestep
    - open_viewer: whether to open the viewer initially

    """
    def __init__(self,
                 model=None,
                 model_arm=None,
                 viewer=True,
                 object_name='object',
                 timestep=0.002):
        self.viewer = None
        self.open_viewer = viewer
        # Load the MuJoCo model
        if model is None:
            model_path = os.path.join(DcmmCfg.ASSET_PATH, DcmmCfg.XML_DCMM_HOOP_OBJECT_PATH)
            self.model_xml_string = xml_to_string(model_path)
            self.model = mujoco.MjModel.from_xml_string(self.model_xml_string)
        else:
            self.model = model
        if model_arm is None:
            model_arm_path = os.path.join(DcmmCfg.ASSET_PATH, DcmmCfg.XML_ARM_PATH)
            self.model_arm = mujoco.MjModel.from_xml_path(model_arm_path)
        else:
            self.model_arm = model_arm
        self.model.opt.timestep = timestep
        self.model_arm.opt.timestep = timestep
        self.data = mujoco.MjData(self.model)
        self.data_arm = mujoco.MjData(self.model_arm)

        ## Get the qpos/qvel addresses of the arm and the object by name
        # (no hard-coded indices, so editing the MJCF does not silently break them)
        try:
            _ = self.data.body(object_name)
        except:
            print("The object name is not found in the model!\
                  \nPlease check the object name in the .xml file.")
            raise ValueError
        self.object_name = object_name
        arm_joint_names = ["joint{}".format(i) for i in range(1, 7)]
        self.arm_qpos_adr = np.array([self.model.joint(name).qposadr[0] for name in arm_joint_names])
        self.arm_qvel_adr = np.array([self.model.joint(name).dofadr[0] for name in arm_joint_names])
        self.arm_ctrl_ids = np.array([self.model.actuator("arm_actuator_{}".format(i)).id for i in range(1, 7)])
        self.object_qpos_adr = self.model.joint(object_name).qposadr[0]
        self.object_qvel_adr = self.model.joint(object_name).dofadr[0]
        self.object_ctrl_id = self.model.actuator("object_actuator").id

        self.data.qpos[self.arm_qpos_adr] = DcmmCfg.arm_joints[:]
        self.data_arm.qpos[0:6] = DcmmCfg.arm_joints[:]
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_forward(self.model_arm, self.data_arm)

        ## Offset of the hoop center in the link6 frame (the IK solves for link6)
        R_link6 = self.data.body("link6").xmat.reshape(3, 3)
        self.hoop_offset = R_link6.T @ (self.data.site("hoop_center").xpos - self.data.body("link6").xpos)

        # Arm Control
        self.arm_pid = PID("arm", DcmmCfg.Kp_arm, DcmmCfg.Ki_arm, DcmmCfg.Kd_arm, dim=6, llim=DcmmCfg.llim_arm, ulim=DcmmCfg.ulim_arm, debug=False)

        ## Define Inverse Kinematics Solver for the Arm
        self.ik_arm = IKArm(solver_type=DcmmCfg.ik_config["solver_type"], ilimit=DcmmCfg.ik_config["ilimit"],
                            ps=DcmmCfg.ik_config["ps"], λΣ=DcmmCfg.ik_config["λΣ"], tol=DcmmCfg.ik_config["ee_tol"],
                            free_ee_z_rot=DcmmCfg.ik_config["free_ee_z_rot"], gain=DcmmCfg.ik_config["gain"])

        ## Initialize the camera parameters
        self.model.vis.global_.offwidth = DcmmCfg.cam_config["width"]
        self.model.vis.global_.offheight = DcmmCfg.cam_config["height"]
        self.create_camera_data(DcmmCfg.cam_config["width"], DcmmCfg.cam_config["height"], DcmmCfg.cam_config["name"])

        ## Initialize the target joint positions of the arm
        self.target_arm_qpos = np.zeros(6)
        self.target_arm_qpos[:] = DcmmCfg.arm_joints[:]

    def show_model_info(self):
        """
        Displays relevant model info for the user, namely bodies, joints, actuators, as well as their IDs and ranges.
        """

        print("\nNumber of bodies: {}".format(self.model.nbody))
        for i in range(self.model.nbody):
            print("Body ID: {}, Body Name: {}".format(i, self.model.body(i).name))

        print("\nNumber of joints: {}".format(self.model.njnt))
        for i in range(self.model.njnt):
            print(
                "Joint ID: {}, Joint Name: {}, Limits: {}, Damping: {}".format(
                    i, self.model.joint(i).name, self.model.jnt_range[i], self.model.dof_damping[i]
                )
            )

        print("\nNumber of Actuators: {}".format(len(self.data.ctrl)))
        for i in range(len(self.data.ctrl)):
            print(
                "Actuator ID: {}, Actuator Name: {}, Controlled Joint: {}, Control Range: {}".format(
                    i,
                    self.model.actuator(i).name,
                    self.model.joint(self.model.actuator(i).trnid[0]).name,
                    self.model.actuator(i).ctrlrange,
                )
            )
        print("\nArm PID Info: \n")
        print(
            "P: {}, I: {}, D: {}".format(
                self.arm_pid.Kp,
                self.arm_pid.Ki,
                self.arm_pid.Kd,
            )
        )
        print("\nSimulation Timestep: ", self.model.opt.timestep)

    def get_target_hoop_pose(self):
        """
        The hoop pose of the current IK target (arm-only model, arm-base frame).

        Return:
        - hoop center position (3,)
        - link6 rotation matrix (3, 3); its z column is the hoop axis
        """
        R_link6 = self.data_arm.body("link6").xmat.reshape(3, 3).copy()
        center = self.data_arm.body("link6").xpos + R_link6 @ self.hoop_offset
        return center, R_link6

    def move_hoop_pose(self, target_center, target_rot):
        """
        Solve the IK so the hoop center reaches target_center and the hoop axis
        reaches the z column of target_rot (arm-base frame).

        Return:
        - result_QP: (target joint positions of the arm, success, ...)
        """
        Tep = np.eye(4)
        Tep[:3, :3] = target_rot
        Tep[:3, 3] = target_center - target_rot @ self.hoop_offset
        result_QP = self.ik_arm.solve(self.model_arm, self.data_arm, Tep, self.data_arm.qpos[0:6])
        if DEBUG_ARM: print("result_QP: ", result_QP)
        # Update the qpos of the arm with the IK solution (the previous one if the IK fails)
        self.data_arm.qpos[0:6] = result_QP[0]
        mujoco.mj_fwdPosition(self.model_arm, self.data_arm)
        return result_QP

    def set_throw_pos_vel(self,
                          pose = np.array([0, 0, 0, 1, 0, 0, 0]),
                          velocity = np.array([0, 0, 0, 0, 0, 0])):
        self.data.qpos[self.object_qpos_adr:self.object_qpos_adr+7] = pose
        self.data.qvel[self.object_qvel_adr:self.object_qvel_adr+6] = velocity

    def pixel_2_world(self, pixel_x, pixel_y, depth, camera="top"):
        """
        Converts pixel coordinates into world coordinates.

        Args:
            pixel_x: X-coordinate in pixel space.
            pixel_y: Y-coordinate in pixel space.
            depth: Depth value corresponding to the pixel.
            camera: Name of camera used to obtain the image.
        """

        if not self.cam_init:
            self.create_camera_data(DcmmCfg.cam_config["width"], DcmmCfg.cam_config["height"], camera)

        # Create coordinate vector
        pixel_coord = np.array([pixel_x,
                                pixel_y,
                                1]) * (depth)

        # Get position relative to camera
        pos_c = np.linalg.inv(self.cam_matrix) @ pixel_coord
        # Transform to the global frame axis
        pos_c[1] *= -1
        pos_c[1], pos_c[2] = pos_c[2], pos_c[1]
        # Get world position
        pos_w = self.cam_rot_mat @ (pos_c) + self.cam_pos

        return pos_c, pos_w

    def depth_2_meters(self, depth):
        """
        Converts the depth array delivered by MuJoCo (values between 0 and 1) into actual m values.

        Args:
            depth: The depth array to be converted.
        """

        extend = self.model.stat.extent
        near = self.model.vis.map.znear * extend
        far = self.model.vis.map.zfar * extend

        return near / (1 - depth * (1 - near / far))

    def create_camera_data(self, width, height, camera):
        """
        Initializes all camera parameters that only need to be calculated once.
        """

        cam_id = self.model.camera(camera).id
        # Get field of view
        fovy = self.model.cam_fovy[cam_id]
        # Calculate focal length
        f = 0.5 * height / np.tan(fovy * np.pi / 360)
        # Construct camera matrix
        self.cam_matrix = np.array(((f, 0, width / 2), (0, f, height / 2), (0, 0, 1)))
        # Rotation of camera in world coordinates
        self.cam_rot_mat = self.model.cam_mat0[cam_id]
        self.cam_rot_mat = np.reshape(self.cam_rot_mat, (3, 3)) @ np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        # Position of camera in world coordinates
        self.cam_pos = self.model.cam_pos0[cam_id] + self.data.body("base_link").xpos - self.data.body("arm_base").xpos
        self.cam_init = True
