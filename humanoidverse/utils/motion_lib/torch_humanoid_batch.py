"""
详细解答如下：

1. 什么是MJCF的根节点？
- MJCF（MuJoCo XML）是一种用于描述机器人模型的XML格式。根节点通常是< mujoco >，它包裹了整个机器人模型的所有信息。
- 例如：
    <mujoco>
            ...  <!-- 这里是所有的子节点 -->
    </mujoco>
- 你可以通过tree.getroot()拿到这个根节点。

2. 什么是worldbody？
- < worldbody > 是MJCF文件中的一个重要子节点，直接位于根节点<mujoco>下。
- 它用于描述机器人在世界坐标系下的所有“刚体”（body），即机器人的各个部分（躯干、四肢等）。
- 结构示例：
    <mujoco>
            <worldbody>
                    <body name="torso">
                            ...
                    </body>
                    ...
            </worldbody>
            ...
    </mujoco>
- 你可以通过tree.getroot().find("worldbody")获取到它。

3. './/joint'中的.//是什么意思，为什么能对应<joint>？
- 这是XML的XPath语法，`.//joint`表示“当前节点下所有后代（不管嵌套多少层）中名字为joint的节点”。
- 具体来说：
    - `.` 代表当前节点（比如worldbody）。
    - `//` 代表递归查找所有子孙节点。
    - `joint` 是节点名。
- 所以`.//joint`会找到worldbody下所有的<joint>节点，无论它们嵌套在多少层<body>里面。
- 例如：
    <worldbody>
            <body>
                    <joint name="root"/>
                    <body>
                            <joint name="knee"/>
                    </body>
            </body>
    </worldbody>
    这里`.//joint`会找到“root”和“knee”两个joint。

4. 如何理解MJCF的结构？
- MJCF结构是树状的，主要分为以下部分：
    - <mujoco> 根节点
        - <compiler>、<option>、<asset>、<worldbody>、<actuator>等
        - <worldbody>：描述机器人本体的刚体结构（body），body可以嵌套body，形成树状结构
            - <body>：每个刚体，可以有<joint>（关节）、<geom>（几何体）、<site>（标记点）等
        - <actuator>：描述电机/驱动器，通常与joint一一对应
            - <motor>：每个电机，name属性与joint对应
- 典型结构如下：
    <mujoco>
            <worldbody>
                    <body name="torso">
                            <joint name="root"/>
                            <body name="thigh">
                                    <joint name="hip"/>
                                    <body name="shin">
                                            <joint name="knee"/>
                                    </body>
                            </body>
                    </body>
            </worldbody>
            <actuator>
                    <motor name="hip"/>
                    <motor name="knee"/>
            </actuator>
    </mujoco>
- 这样，代码里通过`.//joint`可以一次性找到所有关节，通过`find("actuator").getchildren()`可以找到所有电机。

总结：
- 根节点是<mujoco>，worldbody是描述机器人结构的主要部分。
- `.//joint`是递归查找所有关节节点的XPath表达式。
- MJCF结构是树状的，body嵌套body，joint描述关节，motor描述驱动器，actuator统一管理所有电机。

如需更详细的MJCF结构图或实例，可继续提问！
"""
import os # 操作系统接口，用于路径处理
import os.path as osp # 路径拼接工具
import torch # 强化学习张量计算和 GPU 加速的核心库
from collections import defaultdict # 提供带有默认值的字典，方便处理一对多的网格映射

import numpy as np # CPU 端数值计算库
# 导入大量的底层旋转矩阵与四元数转换工具，FK (正向运动学) 的核心就是矩阵乘法
from isaac_utils.rotations import (
    quaternion_to_matrix, # 四元数转旋转矩阵
    wxyz_to_xyzw, # 四元数标准转换（不同物理引擎标准不同）
    axis_angle_to_quaternion, # 轴角转四元数 (把动作库的 pose_aa 变回四元数)
    matrix_to_quaternion, # 旋转矩阵转四元数
    quat_mul_norm, # 四元数乘法并归一化
    quat_identity_like, # 生成单位四元数（代表无旋转）
    quat_inverse, # 求四元数的逆（用于求旋转差/角速度）
    quat_angle_axis # 提取四元数的转轴和角度
)
from scipy.spatial.transform import Rotation as sRot # 高精度 3D 旋转转换库
import xml.etree.ElementTree as ETree # 原生 XML 解析器
from easydict import EasyDict # 允许用点操作符访问字典属性的便捷库
import scipy.ndimage.filters as filters # 信号滤波工具，用于平滑通过差分求出来的速度
from lxml.etree import XMLParser, parse, ElementTree, Element, SubElement # 更高级的 XML 解析器 (lxml)，用于读取 MJCF
from lxml import etree
from io import BytesIO # 内存字节流，辅助解析 XML 文件
import copy # 深度拷贝，防止改坏原始网格数据
from collections import OrderedDict # 有序字典，确保骨骼节点的顺序和 XML 定义一致
import hydra # 配置文件管理神器
from omegaconf import DictConfig # Hydra 配套的配置对象
from stl import mesh # 处理 3D 打印标准网格文件 (.stl)
# import logging
import open3d as o3d # 强大的 3D 数据处理和可视化库

from loguru import logger # 终端彩色日志打印
from pathlib import Path # 面向对象的文件路径处理

from rich.progress import track # 酷炫的终端加载进度条

# Configure logging
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

logger.info("Using Humanoid Batch") # 提示正在使用批量并行的人形机器人运动学解算器

# ============================================================================
# 核心类：Humanoid_Batch (批量运动学解析引擎)
# 【作用】：解析 MJCF 机器人模型文件，建立树状骨架结构。提供高并发的 FK（正向运动学）解算能力。
# 【被谁调用】：在 MotionLibBase 的 __init__ 中被实例化为 `self.mesh_parsers`。
# ============================================================================
class Humanoid_Batch:
    def __init__(self, cfg, device = torch.device("cpu")):
        """
        【函数作用】：初始化引擎。读取机器人的描述文件，提取所有连杆(body)、关节(joint)和电机(motor)的信息，建立父子坐标系偏移。
        【被谁调用】：上层系统初始化 MotionLib 时。
        """
        self.cfg = cfg # 保存配置
        self.asset_root = Path(cfg.asset.assetRoot) # 模型文件夹路径
        self.asset_file = cfg.asset.assetFileName # 模型文件名 (如 g1.xml)
        self.mjcf_file = self.asset_root / self.asset_file # 拼接得到完整的 MJCF 文件路径
        
        # 建立 XML 解析器，忽略空白字符
        parser = XMLParser(remove_blank_text=True)
        # 读取 XML 字节流并解析为树结构
        # 读取 MJCF 机器人模型文件（XML），先用二进制方式读入内存，再用 BytesIO 包装成字节流，
        # 交给 lxml 的 parse 方法和自定义 parser 解析成 XML 树结构。
        # 这样可以高效地处理大文件、支持复杂的 XML 格式。
        tree = parse(BytesIO(open(self.mjcf_file, "rb").read()), parser=parser,)
        self.dof_axis = [] # 记录所有驱动关节的局部旋转轴（如绕 x、y 还是 z 转）

        # 遍历 XML 找所有 joint（关节）的名称
        joints = sorted([j.attrib['name'] for j in tree.getroot().find("worldbody").findall('.//joint')])# j 是一个 XML 节点对象（Element），比如通过 for j in ...findall('.//joint') 得到的每个 joint 节点。比如：<joint name="knee" type="hinge" axis="0 1 0"/>attrib 是 Element 对象的一个字典属性，存储了该节点的所有属性（attribute）。
        # 遍历 XML 找所有 actuator/motor（电机）的名称
        motors = sorted([m.attrib['name'] for m in tree.getroot().find("actuator").getchildren()])
        
        assert len(motors) > 0, "No motors found in the mjcf file"
        
        self.num_dof = len(motors) # 驱动自由度数量 (如 G1 的 23 或 29)
        self.num_extend_dof = self.num_dof # 包含虚拟扩展点后的总自由度数量，初始化时等于物理电机数
        
        # ★ 核心步骤：调用内部函数递归解析 XML，构建骨骼拓扑结构图
        self.mjcf_data = mjcf_data = self.from_mjcf(self.mjcf_file)
        self.body_names = copy.deepcopy(mjcf_data['node_names']) # 连杆名称列表
        self._parents = mjcf_data['parent_indices'] # 父节点索引数组 (谁连着谁)
        self.body_names_augment = copy.deepcopy(mjcf_data['node_names']) # 包含虚拟点后的连杆名称
        
        # 将骨骼初始的相对位移和旋转（Rest Pose）放入 GPU 显存备用
        self._offsets = mjcf_data['local_translation'][None, ].to(device)
        self._local_rotation = mjcf_data['local_rotation'][None, ].to(device)
        
        # 找出哪些连杆是挂载了实际驱动电机的（非固定连接的）
        self.actuated_joints_idx = np.array([self.body_names.index(k) for k, v in mjcf_data['body_to_joint'].items()])
        # k 是连杆（body）的名字
        # v 是该连杆对应的关节（joint）的名字
        # 这行代码的作用是：找出所有“挂载了驱动关节的连杆”在 body_names 列表中的索引，存到 self.actuated_joints_idx 数组里。这样后续可以直接用这些索引定位到需要驱动的连杆。
        
        # 检查有没有只定义了电机没定义关节的异常情况
        for m in motors:
            if not m in joints:
                print(m)
        
        # 处理 Root 根节点的自由关节 (freejoint)
        # 人形机器人的躯干是在空间自由漂浮的，前 6 个自由度通常是空间平移和旋转，不属于电机
        if "type" in tree.getroot().find("worldbody").findall('.//joint')[0].attrib and tree.getroot().find("worldbody").findall('.//joint')[0].attrib['type'] == "free":
            for j in tree.getroot().find("worldbody").findall('.//joint')[1:]:
                # 提取驱动电机的旋转轴 (如 "0 1 0")
                self.dof_axis.append([int(i) for i in j.attrib['axis'].split(" ")])
            self.has_freejoint = True
        elif not "type" in tree.getroot().find("worldbody").findall('.//joint')[0].attrib:
            for j in tree.getroot().find("worldbody").findall('.//joint'):
                self.dof_axis.append([int(i) for i in j.attrib['axis'].split(" ")])
            self.has_freejoint = True
        else:
            for j in tree.getroot().find("worldbody").findall('.//joint')[6:]:
                self.dof_axis.append([int(i) for i in j.attrib['axis'].split(" ")])
            self.has_freejoint = False
        
        # 将局部旋转轴变为张量备用 (这正是我们之前用来合成 pose_aa 的那个 dof_axis)
        self.dof_axis = torch.tensor(self.dof_axis)

        # 处理扩展点 (extend_config，如手心、剑尖等虚拟追踪标记点)
        for extend_config in cfg.extend_config:
            self.body_names_augment += [extend_config.joint_name] # 名字加入列表
            # 把它挂载到配置中指定的父连杆上。从配置中读取每个虚拟点的父连杆名称，并将父连杆名称转换为对应的索引，添加到父节点索引数组中。
            self._parents = torch.cat([self._parents, torch.tensor([self.body_names.index(extend_config.parent_name)]).to(device)], dim = 0)
            # 添加这个虚拟点相对于父连杆的固定位移偏置
            self._offsets = torch.cat([self._offsets, torch.tensor([[extend_config.pos]]).to(device)], dim = 1)
            # 添加固定的旋转偏置
            self._local_rotation = torch.cat([self._local_rotation, torch.tensor([[extend_config.rot]]).to(device)], dim = 1)
            self.num_extend_dof += 1
            
        self.num_bodies = len(self.body_names)
        self.num_bodies_augment = len(self.body_names_augment)
        
        # 读取并转储关节限位（用于计算限位惩罚 Reward）
        self.joints_range = mjcf_data['joints_range'].to(device)
        # 预先将局部的固定四元数旋转，转成旋转矩阵（矩阵乘法在 GPU 上最快）
        self._local_rotation_mat = quaternion_to_matrix(self._local_rotation).float() # w, x, y ,z
        self.load_mesh() # 加载 3D 渲染用的网格模型
        
    def from_mjcf(self, path):
        """
        【函数作用】：递归遍历器。从上到下爬取 XML 树，建立机器人各关节的层级关系（拓扑图）。
        【被谁调用】：自己类的 __init__ 过程。
        """
        # function from Poselib: 
        tree = ETree.parse(path)
        xml_doc_root = tree.getroot()
        xml_world_body = xml_doc_root.find("worldbody")
        if xml_world_body is None:
            raise ValueError("MJCF parsed incorrectly please verify it.")
        # assume this is the root
        xml_body_root = xml_world_body.find("body") # 找到根连杆（通常是躯干 torso 或 pelvis）
        if xml_body_root is None:
            raise ValueError("MJCF parsed incorrectly please verify it.")
            
        xml_joint_root = xml_body_root.find("joint")
        
        node_names = []
        parent_indices = []
        local_translation = []
        local_rotation = []
        joints_range = []
        body_to_joint = OrderedDict()

        # recursively adding all nodes into the skel_tree (递归内函数)
        def _add_xml_node(xml_node, parent_index, node_index):
            node_name = xml_node.attrib.get("name")
            # parse the local translation into float list (解析静态零位的相对偏移)
            pos = np.fromstring(xml_node.attrib.get("pos", "0 0 0"), dtype=float, sep=" ")
            quat = np.fromstring(xml_node.attrib.get("quat", "1 0 0 0"), dtype=float, sep=" ")
            
            # 存入列表
            node_names.append(node_name)
            parent_indices.append(parent_index)
            local_translation.append(pos)
            local_rotation.append(quat)
            curr_index = node_index
            node_index += 1
            all_joints = xml_node.findall("joint") # joints need to remove the first 6 joints
            # 跳过浮动基座的前 6 个空间自由度
            if len(all_joints) == 6:
                all_joints = all_joints[6:]
            
            # 记录电机的限位（range）
            for joint in all_joints:
                if not joint.attrib.get("range") is None: 
                    joints_range.append(np.fromstring(joint.attrib.get("range"), dtype=float, sep=" "))
                else:
                    if not joint.attrib.get("type") == "free":
                        joints_range.append([-np.pi, np.pi]) # 没写限位默认可以转一整圈
            for joint_node in xml_node.findall("joint"):
                body_to_joint[node_name] = joint_node.attrib.get("name") # 连杆与关节的映射
                
            # 向下一层层递归找子连杆
            for next_node in xml_node.findall("body"):
                node_index = _add_xml_node(next_node, curr_index, node_index)

            return node_index
        
        # 从根节点开始触发递归，根的父亲是 -1（即没有父亲，代表世界坐标系原点）
        _add_xml_node(xml_body_root, -1, 0)
        assert(len(joints_range) == self.num_dof) 
        
        # 返回收集到的这套完备的机器人拓扑字典
        return {
            "node_names": node_names,
            "parent_indices": torch.from_numpy(np.array(parent_indices, dtype=np.int32)),
            "local_translation": torch.from_numpy(np.array(local_translation, dtype=np.float32)),
            "local_rotation": torch.from_numpy(np.array(local_rotation, dtype=np.float32)),
            "joints_range": torch.from_numpy(np.array(joints_range)),
            "body_to_joint": body_to_joint
        }

        
    def fk_batch(self, pose, trans, convert_to_mat=True, return_full = False, dt=1/30):
        """
        【函数作用】：正向运动学（FK）的对外观测入口。接收全局的根坐标和所有关节轴角，返回每个身体部件的三维坐标！
        【RL 背景】：这是模仿学习算 Reward 的命脉。模型通过 pose 算出标准动作的手脚位置，再和物理引擎里真的手脚位置一减，就是惩罚值。
        【被谁调用】：`MotionLibBase.load_motion_with_skeleton` 和 `mesh_fk`。
        """
        device, dtype = pose.device, pose.dtype
        pose_input = pose.clone()
        B, seq_len = pose.shape[:2]
        # 裁剪掉多余的关节点（比如参考数据是 27 维，这里只留下机器人图纸上有的那些维数）
        pose = pose[..., :len(self._parents), :] # H1 fitted joints might have extra joints
        
        if convert_to_mat:
            # 将 3D 轴角 (Axis-Angle) 先变四元数，再变旋转矩阵 (3x3)，方便做连乘运算
            pose_quat = axis_angle_to_quaternion(pose.clone())
            pose_mat = quaternion_to_matrix(pose_quat)
        else:
            pose_mat = pose
            
        if pose_mat.shape != 5:
            pose_mat = pose_mat.reshape(B, seq_len, -1, 3, 3)
        J = pose_mat.shape[2] - 1  # Exclude root (刨去躯干，剩下的是电机)
        
        # ★ 调用纯数学核，推算 FK
        wbody_pos, wbody_mat = self.forward_kinematics_batch(pose_mat[:, :, 1:], pose_mat[:, :, 0:1], trans)
        
        return_dict = EasyDict()
        
        # 把最终世界坐标系下的绝对旋转矩阵，退回四元数方便物理引擎和神经网络使用
        wbody_rot = wxyz_to_xyzw(matrix_to_quaternion(wbody_mat))
        
        # 如果挂载了虚拟追踪点（extend）
        if len(self.cfg.extend_config) > 0:
            if return_full:
                # 对坐标和旋转进行关于时间的求导，算出虚拟点的线速度和角速度
                return_dict.global_velocity_extend = self._compute_velocity(wbody_pos, dt) 
                return_dict.global_angular_velocity_extend = self._compute_angular_velocity(wbody_rot, dt)
                
            return_dict.global_translation_extend = wbody_pos.clone()
            return_dict.global_rotation_mat_extend = wbody_mat.clone()
            return_dict.global_rotation_extend = wbody_rot
            
            # 把原生刚体的数据截取出来（分离本体和附加的虚拟点）
            wbody_pos = wbody_pos[..., :self.num_bodies, :]
            wbody_mat = wbody_mat[..., :self.num_bodies, :, :]
            wbody_rot = wbody_rot[..., :self.num_bodies, :]

        
        # 存入标准返回字典
        return_dict.global_translation = wbody_pos
        return_dict.global_rotation_mat = wbody_mat
        return_dict.global_rotation = wbody_rot
        
        if return_full:
            # 同样对本体的所有刚体求线速度和角速度
            rigidbody_linear_velocity = self._compute_velocity(wbody_pos, dt)  # Isaac gym is [x, y, z, w]. All the previous functions are [w, x, y, z]
            rigidbody_angular_velocity = self._compute_angular_velocity(wbody_rot, dt)
            return_dict.local_rotation = wxyz_to_xyzw(pose_quat)
            return_dict.global_root_velocity = rigidbody_linear_velocity[..., 0, :]
            return_dict.global_root_angular_velocity = rigidbody_angular_velocity[..., 0, :]
            return_dict.global_angular_velocity = rigidbody_angular_velocity
            return_dict.global_velocity = rigidbody_linear_velocity
            
            # 从 3D 轴角逆向求和得出单轴电机的标量角度 dof_pos
            if len(self.cfg.extend_config) > 0:
                return_dict.dof_pos = pose.sum(dim = -1)[..., 1:self.num_bodies] # you can sum it up since unitree's each joint has 1 dof. Last two are for hands. doesn't really matter. 
            else:
                if not len(self.actuated_joints_idx) == len(self.body_names):
                    return_dict.dof_pos = pose.sum(dim = -1)[..., self.actuated_joints_idx]
                else:
                    return_dict.dof_pos = pose.sum(dim = -1)[..., 1:]
            
            # 对关节角度求一次差分（求导），得出关节电机的运转速度 dof_vel
            dof_vel = ((return_dict.dof_pos[:, 1:] - return_dict.dof_pos[:, :-1] )/dt)
            return_dict.dof_vels = torch.cat([dof_vel, dof_vel[:, -2:-1]], dim = 1)
            return_dict.fps = int(1/dt)
        
        return return_dict
    
    def forward_kinematics_batch(self, rotations, root_rotations, root_positions):
        """
        【函数作用】：全场最底层的数学运算！利用前向动力学（FK），基于各关节相对旋转，计算出所有连杆在世界空间的绝对位置。
        Perform forward kinematics using the given trajectory and local rotations.
        Arguments (where B = batch size, J = number of joints):
         -- rotations: (B, J, 4) tensor of unit quaternions describing the local rotations of each joint. (关节局部旋转)
         -- root_positions: (B, 3) tensor describing the root joint positions. (根节点绝对位置)
        Output: joint positions (B, J, 3)
        【被谁调用】：内部的 fk_batch()。
        """
        
        device, dtype = root_rotations.device, root_rotations.dtype
        B, seq_len = rotations.size()[0:2]
        J = self._offsets.shape[1]
        positions_world = [] # 记录所有节点的世界坐标
        rotations_world = [] # 记录所有节点的世界旋转矩阵

        # 展开局部偏移常数，适应 Batch 运算
        expanded_offsets = (self._offsets[:, None].expand(B, seq_len, J, 3).to(device).type(dtype))
        # print(expanded_offsets.shape, J)

        # import ipdb; ipdb.set_trace()   
        # 按照拓扑图从躯干往手脚方向进行矩阵乘法连乘（链式法则）
        for i in range(J):
            if self._parents[i] == -1:
                # 第一个节点是躯干根节点，直接赋予传入的绝对位置和旋转
                positions_world.append(root_positions)
                rotations_world.append(root_rotations)
            else:
                try:
                    # 子节点绝对坐标 = 父节点绝对坐标 + 父节点绝对旋转阵 × 子节点相对偏置向量
                    jpos = (torch.matmul(rotations_world[self._parents[i]][:, :, 0], expanded_offsets[:, :, i, :, None]).squeeze(-1) + positions_world[self._parents[i]])
                    # 子节点绝对旋转 = 父节点绝对旋转 × 子节点局部初始旋转 × 子节点当前的电机转角旋转
                    rot_mat = torch.matmul(rotations_world[self._parents[i]], torch.matmul(self._local_rotation_mat[:,  (i):(i + 1)], rotations[:, :, (i - 1):i, :]))
                except Exception as e:
                    logger.error(f"Error at joint index {i}")
                    logger.error(f"Parent index: {self._parents[i]}")
                    logger.error(f"Error details: {str(e)}")
                    import ipdb; ipdb.set_trace()
                # rot_mat = torch.matmul(rotations_world[self._parents[i]], rotations[:, :, (i - 1):i, :])
                # print(rotations[:, :, (i - 1):i, :].shape, self._local_rotation_mat.shape)
                
                # 算好一个挂入列表，供下一次它的子节点循环使用
                positions_world.append(jpos)
                rotations_world.append(rot_mat)
        
        # 将列表堆叠为标准的批量张量
        positions_world = torch.stack(positions_world, dim=2)
        rotations_world = torch.cat(rotations_world, dim=2)
        return positions_world, rotations_world
    
    @staticmethod
    def _compute_velocity(p, time_delta, guassian_filter=True):
        """
        【函数作用】：通过时间序列位置的差分（求导），计算出线速度。
        【RL 背景】：模仿学习要求机器人不仅姿势要对，挥拳的速度也得对。
        """
        # 使用 numpy.gradient 进行中心差分
        velocity = np.gradient(p.numpy(), axis=-3) / time_delta
        if guassian_filter:
            # 差分会放大高频噪声，使用高斯滤波让速度曲线变得平滑自然
            velocity = torch.from_numpy(filters.gaussian_filter1d(velocity, 2, axis=-3, mode="nearest")).to(p)
        else:
            velocity = torch.from_numpy(velocity).to(p)
        
        return velocity
    
    @staticmethod
    def _compute_angular_velocity(r, time_delta: float, guassian_filter=True):
        """
        【函数作用】：通过姿态的差分计算角速度。四元数不能直接做差，必须使用逆与乘法。
        """
        # assume the second last dimension is the time axis
        diff_quat_data = quat_identity_like(r).to(r)
        # 当前帧四元数 乘以 上一帧四元数的逆 = 这一帧发生的那一点点微小旋转差
        diff_quat_data[..., :-1, :, :] = quat_mul_norm(r[..., 1:, :, :], quat_inverse(r[..., :-1, :, :], w_last=True), w_last=True)
        # 将这个微小四元数转化为转轴和转角
        diff_angle, diff_axis = quat_angle_axis(diff_quat_data, w_last=True)
        # 转角 / 时间 dt = 角速度
        angular_velocity = diff_axis * diff_angle.unsqueeze(-1) / time_delta
        if guassian_filter:
            angular_velocity = torch.from_numpy(filters.gaussian_filter1d(angular_velocity.numpy(), 2, axis=-3, mode="nearest"),)
        return angular_velocity  
    
    def load_mesh(self):
        """
        【函数作用】：从文件系统读取机器人的 3D 网格文件（STL/OBJ）。
        【被谁调用】：自己类的 __init__。主要用于生成动作库动图或离线渲染检查，不参与强化学习核心物理逻辑。
        """
        xml_base = os.path.dirname(self.mjcf_file)
        # Read the compiler tag from the g1.xml file to find if there is a meshdir defined
        tree = ETree.parse(self.mjcf_file)
        xml_doc_root = tree.getroot()
        compiler_tag = xml_doc_root.find("compiler")
        
        # 寻找包含几何网格的文件夹
        if compiler_tag is not None and "meshdir" in compiler_tag.attrib:
            mesh_base = os.path.join(xml_base, compiler_tag.attrib["meshdir"])
        else:
            mesh_base = xml_base
            
        self.tree = tree = ETree.parse(self.mjcf_file)
        xml_doc_root = tree.getroot()
        xml_world_body = xml_doc_root.find("worldbody")

        xml_assets = xml_doc_root.find("asset")
        all_mesh = xml_assets.findall(".//mesh")

        geoms = xml_world_body.findall(".//geom")

        all_joints = xml_world_body.findall(".//joint")
        all_motors = tree.findall(".//motor")
        all_bodies = xml_world_body.findall(".//body")

        # 辅助函数：找父节点
        def find_parent(root, child):
            for parent in root.iter():
                for elem in parent:
                    if elem == child:
                        return parent
            return None

        mesh_dict = {}
        mesh_parent_dict = {}

        # 真正开始利用 Open3D 把 3D 打印模型文件读入内存
        for mesh_file_node in track(all_mesh, description="Loading Meshes ..."):
            mesh_name = mesh_file_node.attrib["name"]
            mesh_file = mesh_file_node.attrib["file"]
            mesh_full_file = osp.join(mesh_base, mesh_file)
            mesh_obj = o3d.io.read_triangle_mesh(mesh_full_file)
            mesh_dict[mesh_name] = mesh_obj

        geom_transform = {}
        
        body_to_mesh = defaultdict(set)
        mesh_to_body = {}
        # 把读入的静态 3D 模型，跟我们代码里建好的骨骼拓扑树一一挂钩
        for geom_node in track(geoms, description="Loading Geoms..."):
            if 'mesh' in geom_node.attrib: 
                parent = find_parent(xml_doc_root, geom_node)
                body_to_mesh[parent.attrib['name']].add(geom_node.attrib['mesh'])
                mesh_to_body[geom_node] = parent
                # 记录渲染外观（geom）相对于物理碰撞骨架的位置偏移
                if "pos" in geom_node.attrib or "quat" in geom_node.attrib:
                    geom_transform[parent.attrib['name']] = {}
                    geom_transform[parent.attrib['name']]["pos"] = np.array([0.0, 0.0, 0.0])
                    geom_transform[parent.attrib['name']]["quat"] = np.array([1.0, 0.0, 0.0, 0.0])
                    if "pos" in geom_node.attrib:
                        geom_transform[parent.attrib['name']]["pos"] = np.array([float(f) for f in geom_node.attrib['pos'].split(" ")])
                    if "quat" in geom_node.attrib:
                        geom_transform[parent.attrib['name']]["quat"] = np.array([float(f) for f in geom_node.attrib['quat'].split(" ")])
                    
            else:
                pass
            
        self.geom_transform = geom_transform
        self.mesh_dict = mesh_dict
        self.body_to_mesh = body_to_mesh
        self.mesh_to_body = mesh_to_body

    def mesh_fk(self, pose = None, trans = None):
        """
        【函数作用】：可视化魔法！把你计算好的骨骼动作（pose 和 trans），赋予到那些冷冰冰的 3D 模型块上，并拼接成一个完整的变形网格。
        Load the mesh from the XML file and merge them into the humanoid based on the current pose.
        【被谁调用】：常用于导出供 Blender/Maya 等软件渲染的静态模型，或调试对地面的穿模修复。
        """
        # 如果不给动作，就渲染一个全 0 角度的站立形态（T-pose 或 A-pose）
        if pose is None:
            fk_res = self.fk_batch(torch.zeros(1, 1, len(self.body_names_augment), 3), torch.zeros(1, 1, 3))
        else:
            fk_res = self.fk_batch(pose, trans) # 拿到算好的世界坐标
        
        g_trans = fk_res.global_translation.squeeze()
        g_rot = fk_res.global_rotation_mat.squeeze()
        geoms = self.tree.find("worldbody").findall(".//geom")
        joined_mesh_obj = []
        
        # 遍历每一块模型（如左大腿外壳），强行把它平移和旋转到 FK 算出的绝对位置上去
        for geom in geoms:
            if 'mesh' not in geom.attrib:
                continue
            parent_name = geom.attrib['mesh']

            k = self.mesh_to_body[geom].attrib['name']
            mesh_names = self.body_to_mesh[k]
            body_idx = self.body_names.index(k)
            
            body_trans = g_trans[body_idx].numpy().copy()
            body_rot = g_rot[body_idx].numpy().copy()
            for mesh_name in mesh_names:
                mesh_obj = copy.deepcopy(self.mesh_dict[mesh_name])
                if k in self.geom_transform:
                    pos = self.geom_transform[k]['pos']
                    quat = self.geom_transform[k]['quat']
                    body_trans = body_trans + body_rot @ pos
                    global_rot =  (body_rot   @ sRot.from_quat(quat[[1, 2, 3, 0]]).as_matrix()).T
                else:
                    global_rot = body_rot.T
                mesh_obj.rotate(global_rot.T, center=(0, 0, 0)) # 执行刚体旋转
                mesh_obj.translate(body_trans) # 执行刚体平移
                joined_mesh_obj.append(mesh_obj) # 存入待合并队列
                
        # Merge all meshes into a single mesh
        # 用 open3d 把成百上千的小外壳零件，黏合成一个单一的整体 3D 网格模型
        merged_mesh = joined_mesh_obj[0]
        for mesh in joined_mesh_obj[1:]:
            merged_mesh += mesh
        
        # Save the merged mesh to a file (注释掉的存盘代码)
        # merged_mesh.compute_vertex_normals()
        # o3d.io.write_triangle_mesh(f"data/{self.cfg.humanoid_type}/combined_{self.cfg.humanoid_type}.stl", merged_mesh)
        return merged_mesh

    
    
@hydra.main(version_base=None, config_path="../../config", config_name="base")
def main(config: DictConfig):
    """
    【函数作用】：单文件调试入口。当你直接 python 运行这个脚本时触发。通常用来校验 MJCF 图纸读取或 3D 渲染功能有没有写错。
    """
    device = torch.device("cpu")
    humanoid_fk = Humanoid_Batch(config.robot.motion, device)
    humanoid_fk.mesh_fk() # 试着拼装一次模型看会不会崩

if __name__ == "__main__":
    main()