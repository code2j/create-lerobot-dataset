#!/usr/bin/env python3

import os
import time
import threading
import numpy as np
import torch
import cv2

# ROS2 관련 임포트
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import CompressedImage
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration

# 사용자 정의 모듈 (기존 파일들이 같은 경로에 있다고 가정)
from lerobot.policies.pretrained import PreTrainedPolicy
from file_utils import read_json_file
from subscriber_hub import SubscriberHub
from data_converter import decode_image

# ==================================================================
# 1. InferenceManager 클래스 (제공해주신 코드 유지)
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
        # 추가 정책들은 필요시 여기에 추가
        raise NotImplementedError(f'Policy {name} not implemented in this snippet.')

    @staticmethod
    def get_available_policies():
        return ['tdmpc', 'diffusion', 'act', 'vqbet', 'pi0', 'pi0fast', 'smolvla']

    def clear_policy(self):
        if hasattr(self, 'policy'): del self.policy; self.policy = None

# ==================================================================
# 2. Main 실행 루프
# ==================================================================
def main():
    if not rclpy.ok():
        rclpy.init()

    # --- ROS2 인프라 설정 ---
    hub = SubscriberHub()
    # SubscriberHub 스핀을 위한 별도 쓰레드
    spin_thread = threading.Thread(target=lambda: rclpy.spin(hub), daemon=True)
    spin_thread.start()

    # 액션 클라이언트를 위한 임시 노드 생성
    action_node = rclpy.create_node('inference_action_client')
    action_client = ActionClient(
        action_node,
        FollowJointTrajectory,
        '/arm_controller2/follow_joint_trajectory'
    )

    print("기다리는 중: 액션 서버...")
    if not action_client.wait_for_server(timeout_sec=10.0):
        print("에러: 액션 서버를 찾을 수 없습니다.")
        return

    # [중요] 로봇의 실제 조인트 이름 리스트 (자신의 로봇 설정에 맞게 수정)
    JOINT_NAMES = ['right_joint1', 'right_joint2', 'right_joint3', 'right_joint4', 'right_joint5', 'right_joint6']

    # --- 모델 로드 ---
    inference_manager = InferenceManager(device='cuda')
    POLICY_PATH = "/home/uon/workspace/jusik_dataset/dataset/train/act_uon/checkpoints/last/pretrained_model"

    is_valid, message = inference_manager.validate_policy(POLICY_PATH)
    print(f"모델 검사: {message}")

    if not is_valid or not inference_manager.load_policy():
        print("모델 로드 실패.")
        return

    # --- 제어 루프 설정 (10 FPS) ---
    target_fps = 10
    target_period = 1.0 / target_fps
    print(f"\n[시작] {target_fps} FPS 주기로 제어를 시작합니다.")

    try:
        while rclpy.ok():
            start_time = time.perf_counter()

            # 1) 최신 데이터 수집
            kinect_msg, wrist_cam_msg, follower_msg, _ = hub.get_latest_msg()

            if kinect_msg is None or follower_msg is None:
                # 데이터가 올 때까지 아주 짧게 대기
                time.sleep(0.01)
                continue

            # 2) 데이터 전처리 (디코딩 및 변환)
            kinect_img = decode_image(kinect_msg)
            wrist_img = decode_image(wrist_cam_msg)

            total_joint_order = ['right_joint1', 'right_joint2', 'right_joint3', 'right_joint4', 'right_joint5', 'right_joint6', 'right_rh_r1_joint']
            state_dict = dict(zip(follower_msg.name, follower_msg.position))
            current_states = np.array([state_dict[name] for name in total_joint_order], dtype=np.float32)

            # 3) 모델 추론
            input_images = {
                'cam_top': kinect_img,
                'cam_wrist': wrist_img,
            }

            predicted_action = inference_manager.predict(
                images=input_images,
                state=current_states.tolist(),
                task_instruction="pick the zipper bag"
            )

            # 4) 액션 서버로 명령 전송 (비동기)
            if predicted_action is not None:
                goal_msg = FollowJointTrajectory.Goal()
                goal_msg.trajectory.joint_names = JOINT_NAMES

                point = JointTrajectoryPoint()
                # 추론된 액션 값 적용 (조인트 개수에 맞춰 슬라이싱)
                point.positions = predicted_action.tolist()[:len(JOINT_NAMES)]

                # 10FPS 주기에 맞춰 1초 내에 도달하도록 설정
                point.time_from_start = Duration(sec=0, nanosec=500000000)

                goal_msg.trajectory.points = [point]

                # 결과를 기다리지 않고(Non-blocking) 다음 루프로 바로 넘어감
                action_client.send_goal_async(goal_msg)

                # 디버깅 출력 (주기가 빠르므로 필요시 주석 처리)
                print(f"Sent Action: {point.positions}")

            # 5) 주기 유지를 위한 정밀 대기
            elapsed = time.perf_counter() - start_time
            remaining = target_period - elapsed
            if remaining > 0:
                time.sleep(remaining)
            else:
                print(f"Warning: 루프 지연 발생! 현재 속도: {1.0/elapsed:.2f} FPS")

    except KeyboardInterrupt:
        print("\n사용자에 의해 중단됨.")
    finally:
        # 리소스 정리
        inference_manager.clear_policy()
        action_node.destroy_node()
        hub.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()