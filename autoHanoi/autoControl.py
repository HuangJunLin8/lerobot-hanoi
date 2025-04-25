import torch
import logging
from get_state import get_state
from pathlib import Path
from typing import List
from hanoi_algo import *
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.robot_devices.control_configs import ControlConfig, RecordControlConfig
from lerobot.common.robot_devices.robots.configs import So100RobotConfig
from lerobot.common.robot_devices.robots.utils import make_robot_from_config
from lerobot.common.utils.utils import get_safe_torch_device, init_logging, log_say
from lerobot.common.robot_devices.utils import busy_wait
from lerobot.common.policies.factory import make_policy
from lerobot.common.robot_devices.control_utils import (
    control_loop,
    has_method,
    init_keyboard_listener,
    log_control_info,
    record_episode,
    reset_environment,
    stop_recording,
    warmup_record,
)
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.scripts.control_robot import _init_rerun


def generate_task_path(initial_state, move):
    """
    根据 initial_state 和 move 生成路径
    """
    
    # 将状态格式化为 A[123]-B[4]-C[]
    state_repr = "-".join([f"{rod}{''.join(map(str, disks))}" for rod, disks in initial_state.items()])
    
    # 动作描述
    move_desc = f"mv{move['from']}2{move['to']}"
    move_detil = f"moving disk from {move['from']} to {move['to']}"
    
    # 生成路径
    path = Path(f"outputs/train/{state_repr}_{move_desc}/checkpoints/last/pretrained_model")

    # 生成合并动作后的路径
    # path = Path(f"outputs/train/{move_desc}/checkpoints/last/pretrained_model")

    return path, move_detil


def execute_actions(
        repo_id: str,
        fps:int,
        root: Path = None,
        warmup_time_s: int | float = 3, 
        num_episodes: int = 1,
        episode_time_s: int = 30,
        display_data: bool = True,
        task_name: str = None,
        model_paths: list=None,
        move_steps: list=None 
        ):
    """
    执行多个预训练模型的动作，每个模型顺序执行一次动作。
    """

    init_logging()

    # 注册控制参数
    control_cfg = RecordControlConfig(
        repo_id=repo_id,
        fps=fps,
        warmup_time_s=warmup_time_s,
        episode_time_s=episode_time_s,
        num_episodes=num_episodes,
        display_data=display_data,
        single_task=task_name
    )
    cli_overrides = parser.get_cli_overrides("control.policy")

    # # 初始化 rerun 可视化软件
    _init_rerun(control_config=control_cfg, session_name="lerobot_control_loop_record")

    # 加载机械臂参数（端口号等）
    robot = make_robot_from_config(So100RobotConfig())

    # 建立空的数据集（每个episode存所有7步动作）
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=root,
        robot=robot,
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=4 * len(robot.cameras),
    )    

    # 连接机械臂
    if not robot.is_connected:
        robot.connect()


    # 初始化键盘监听
    listener, events = init_keyboard_listener()


    log_say("Warmup record", play_sounds=True, blocking=False)

    # 热身
    warmup_record(robot, events, 
                  enable_teleoperation=True, 
                  warmup_time_s=warmup_time_s, 
                  display_data=display_data, 
                  fps=fps)
    
    if has_method(robot, "teleop_safety_stop"):
        robot.teleop_safety_stop()
    
    steps = len(model_paths)
    recorded_episodes = 0

    policies = []
    for i in range(steps):
        # 加载模型
        policy_cfg = PreTrainedConfig.from_pretrained(model_paths[i], cli_overrides=cli_overrides)
        policy_cfg.pretrained_path = model_paths[i]

        policy = make_policy(policy_cfg, ds_meta=dataset.meta)

        policies.append(policy)
    
    # 循环录制
    while True:
        if recorded_episodes >= num_episodes:
            break

        for i in range(steps):
            log_say(f"{move_steps[i]}", play_sounds=True, blocking=True)

            # 执行单个动作
            record_episode(
                robot=robot,
                dataset=dataset,
                events=events,
                episode_time_s=episode_time_s,
                display_data=display_data,
                policy=policies[i],
                fps=fps,
                single_task=task_name,
            )

            if events["rerecord_episode"]:
                log_say("stopping", play_sounds=True)
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()

                warmup_record(robot, events, 
                    enable_teleoperation=False, 
                    warmup_time_s=100, 
                    display_data=display_data, 
                    fps=fps)

                continue

            policies[i] = None

        dataset.save_episode()
        recorded_episodes += 1

    log_say("Mission Accomplished, exiting", play_sounds=True, blocking=True)
    stop_recording(robot, listener, display_data)
    
    if robot.is_connected:
        robot.disconnect()


def main():
    # 初始状态
    initial_state = get_state()

    # 测试
    initial_state= {'A': [1, 2, 3, 4], 'B': [], 'C': []}
    
    # 汉诺塔算法求解
    moves, states = solve_hanoi(initial_state)
    
    # 解析模型路径
    model_paths = []
    move_steps = []
    for i in range(len(moves)):
        path, step = generate_task_path(states[i], moves[i])
        model_paths.append(path) 
        move_steps.append(step)

    # 执行每个模型的动作
    execute_actions( 
        repo_id="ricaal/autoHanoi",
        fps=30,  
        warmup_time_s=10,
        episode_time_s=50,
        num_episodes=10,
        display_data=True,
        model_paths=model_paths,
        move_steps=move_steps,
        task_name="four disks hanio solution"
        )       



if __name__ == "__main__":
    main()


"""
开始测试一次性走完所有动作
python3 autoHanoi/autoControl.py
"""
