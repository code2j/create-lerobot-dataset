import os
import time
import threading
import numpy as np
import torch
import cv2
from typing import Dict, List, Any

# ROS2 관련 임포트
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

# 사용자 정의 모듈
from lerobot.policies.pretrained import PreTrainedPolicy
from file_utils import read_json_file
from subscriber_hub import SubscriberHub
from data_converter import decode_image



def tensor_array2joint_msgs(action, joint_names):
    """
    Twist 제거 버전: action 텐서를 JointTrajectory 메시지로 변환
    """
    msg = JointTrajectory()
    msg.joint_names = joint_names

    # 만약 action이 (Time_Step, Joint_Dim) 형태라면 첫 번째 액션만 사용
    if action.ndim > 1:
        target_positions = action[0].tolist()
    else:
        target_positions = action.tolist()

    point = JointTrajectoryPoint()
    point.positions = [float(p) for p in target_positions] # 명시적 float 형변환

    # 제어기가 부드럽게 동작하도록 시간 간격 설정 (예: 0.1초 내 도달)
    point.time_from_start.sec = 0
    point.time_from_start.nanosec = 100_000_000 # 100ms

    msg.points = [point]
    return msg


# ==================================================================
# 1. InferenceManager 클래스 (기존 로직 유지)
# ==================================================================
class InferenceManager:
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.policy_type = None
        self.policy_path = None
        self.policy = None

    def validate_policy(self, policy_path: str) -> bool:
        if not os.path.exists(policy_path) or not os.path.isdir(policy_path):
            return False, f'Path {policy_path} not found.'
        config_path = os.path.join(policy_path, 'config.json')
        if not os.path.exists(config_path):
            return False, 'config.json missing.'
        config = read_json_file(config_path)
        available_policies = self.get_available_policies()
        policy_type = config.get('type') or config.get('model_type')
        if policy_type not in available_policies:
            return False, f'Unsupported policy: {policy_type}'
        self.policy_path = policy_path
        self.policy_type = policy_type
        return True, f'Policy {policy_type} is valid.'

    def load_policy(self):
        try:
            policy_cls = self._get_policy_class(self.policy_type)
            self.policy = policy_cls.from_pretrained(self.policy_path)
            self.policy.to(self.device)
            self.policy.eval()
            return True
        except Exception as e:
            print(f'Failed to load policy: {e}')
            return False

    def predict(self, images: dict, state: list, task_instruction: str = None) -> list:
        observation = self._preprocess(images, state, task_instruction)
        with torch.inference_mode():
            action = self.policy.select_action(observation)
            action = action.squeeze(0).to('cpu').numpy()
        return action

    def _preprocess(self, images: dict, state: list, task_instruction: str = None) -> dict:
        observation = self._convert_images2tensors(images)
        observation['observation.state'] = self._convert_np2tensors(state)
        for key in observation.keys():
            observation[key] = observation[key].to(self.device)
        if task_instruction is not None:
            observation['task'] = [task_instruction]
        return observation

    def _convert_images2tensors(self, images: dict) -> dict:
        processed_images = {}
        for key, value in images.items():
            image = torch.from_numpy(value).to(torch.float32) / 255
            image = image.permute(2, 0, 1).to(self.device).unsqueeze(0)
            processed_images['observation.images.' + key] = image
        return processed_images

    def _convert_np2tensors(self, data):
        if isinstance(data, list): data = np.array(data)
        return torch.from_numpy(data).to(torch.float32).to(self.device).unsqueeze(0)

    def _get_policy_class(self, name: str):
        if name == 'act': from lerobot.policies.act.modeling_act import ACTPolicy; return ACTPolicy
        elif name == 'diffusion': from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy; return DiffusionPolicy
        raise NotImplementedError(f'Policy {name} not implemented.')

    @staticmethod
    def get_available_policies():
        return ['tdmpc', 'diffusion', 'act', 'vqbet', 'pi0', 'pi0fast', 'smolvla']

    def clear_policy(self):
        if hasattr(self, 'policy'): del self.policy; self.policy = None



# ==================================================================
# 2. Main 실행 루프 (Topic Publisher 방식)
# ==================================================================
def main():
    if not rclpy.ok():
        rclpy.init()

    # --- ROS2 허브 설정 ---
    hub = SubscriberHub()
    spin_thread = threading.Thread(target=lambda: rclpy.spin(hub), daemon=True)
    spin_thread.start()

    # --- 로봇 제어 노드 설정 ---
    node = rclpy.create_node('inference_publisher_node')
    joint_pub = node.create_publisher(
        JointTrajectory,
        '/right_robot/leader/joint_trajectory',
        10
    )

    # --- 제어 대상 조인트 리스트 ---
    TOTAL_JOINT_NAMES = [
        'right_joint1',
        'right_joint2',
        'right_joint3',
        'right_joint4',
        'right_joint5',
        'right_joint6',
        'right_rh_r1_joint'
    ]

    # --- 모델 로드 ---
    inference_manager = InferenceManager(device='cuda')
    POLICY_PATH = "/home/uon/workspace/create-lerobot-dataset-master/dataset/train/act_uon/checkpoints/last/pretrained_model"

    is_valid, message = inference_manager.validate_policy(POLICY_PATH)
    print(f"모델 검사: {message}")

    if not is_valid or not inference_manager.load_policy():
        print("모델 로드 실패.")
        return

    # --- 제어 루프 설정 (30 FPS) ---
    target_fps = 30
    target_period = 1.0 / target_fps
    print(f"\n[시작] 토픽 발행 방식으로 제어를 시작합니다. (주기: {target_fps}Hz)")

    try:
        while rclpy.ok():
            start_time = time.perf_counter()

            # 1) 최신 데이터 수집
            kinect_msg, wrist_cam_msg, follower_msg, _ = hub.get_latest_msg()

            if kinect_msg is None or follower_msg is None:
                time.sleep(0.01)
                continue

            # 2) 전처리
            kinect_img = decode_image(kinect_msg)
            wrist_img = decode_image(wrist_cam_msg)

            # Follower 상태를 학습 모델 순서에 맞춤
            state_dict = dict(zip(follower_msg.name, follower_msg.position))
            current_states = np.array([state_dict[name] for name in TOTAL_JOINT_NAMES], dtype=np.float32)

            # 3) 모델 추론
            input_images = {
                'cam_top': kinect_img,
                'cam_wrist': wrist_img,
            }

            predicted_action = inference_manager.predict(
                images=input_images,
                state=current_states.tolist(),
                task_instruction="pick up the zipper bag"
            )

            print(f'[Debug] state:\n {current_states}')

            # 4) JointTrajectory 메시지 생성 및 발행
            # if predicted_action is not None:
            #     msg = tensor_array2joint_msgs(predicted_action, TOTAL_JOINT_NAMES)
            #
            #     # 토픽 발행
            #     joint_pub.publish(msg)
            #
            #     # 디버깅 출력: msg에서 직접 값을 가져오도록 변경
            #     print(f"Published Joint States: {np.round(msg.points[0].positions, 4)}")

            # 5) 주기 유지
            elapsed = time.perf_counter() - start_time
            remaining = target_period - elapsed
            if remaining > 0:
                time.sleep(remaining)

    except KeyboardInterrupt:
        print("\n중단됨.")
    finally:
        inference_manager.clear_policy()
        node.destroy_node()
        hub.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()