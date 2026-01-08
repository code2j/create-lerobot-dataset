#!/usr/bin/env python3
#
# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Dongyun Kim

import os

from lerobot.policies.pretrained import PreTrainedPolicy
import numpy as np
from file_utils import read_json_file
import torch


class InferenceManager:

    def __init__(
            self,
            device: str = 'cuda'):

        self.device = device
        self.policy_type = None
        self.policy_path = None
        self.policy = None

    def validate_policy(self, policy_path: str) -> bool:
        result_message = ''
        if not os.path.exists(policy_path) or not os.path.isdir(policy_path):
            result_message = f'Policy path {policy_path} does not exist or is not a directory.'
            return False, result_message

        config_path = os.path.join(policy_path, 'config.json')
        if not os.path.exists(config_path):
            result_message = f'config.json file does not exist in {policy_path}.'
            return False, result_message

        config = read_json_file(config_path)
        if (config is None or
                ('type' not in config and 'model_type' not in config)):
            result_message = f'config.json malformed or missing fields in {policy_path}.'
            return False, result_message

        available_policies = self.__class__.get_available_policies()
        policy_type = config.get('type') or config.get('model_type')
        if policy_type not in available_policies:
            result_message = f'Policy type {policy_type} is not supported.'
            return False, result_message

        self.policy_path = policy_path
        self.policy_type = policy_type
        return True, f'Policy {policy_type} is valid.'

    def load_policy(self):
        try:
            policy_cls = self._get_policy_class(self.policy_type)
            self.policy = policy_cls.from_pretrained(self.policy_path)
            return True
        except Exception as e:
            print(f'Failed to load policy from {self.policy_path}: {e}')
            return False

    def clear_policy(self):
        if hasattr(self, 'policy'):
            del self.policy
            self.policy = None
        else:
            print('No policy to clear.')

    def get_policy_config(self):
        return self.policy.config

    def predict(
            self,
            images: dict[str, np.ndarray],
            state: list[float],
            task_instruction: str = None) -> list:

        observation = self._preprocess(images, state, task_instruction)
        with torch.inference_mode():
            action = self.policy.select_action(observation)
            action = action.squeeze(0).to('cpu').numpy()

        return action

    def _preprocess(
            self,
            images: dict[str, np.ndarray],
            state: list,
            task_instruction: str = None) -> dict:

        observation = self._convert_images2tensors(images)
        observation['observation.state'] = self._convert_np2tensors(state)
        for key in observation.keys():
            observation[key] = observation[key].to(self.device)

        if task_instruction is not None:
            observation['task'] = [task_instruction]

        return observation

    def _convert_images2tensors(
            self,
            images: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:

        processed_images = {}
        for key, value in images.items():
            image = torch.from_numpy(value)
            image = image.to(torch.float32) / 255
            image = image.permute(2, 0, 1)
            image = image.to(self.device, non_blocking=True)
            image = image.unsqueeze(0)
            processed_images['observation.images.' + key] = image

        return processed_images

    def _convert_np2tensors(
            self,
            data):
        if isinstance(data, list):
            data = np.array(data)
        tensor_data = torch.from_numpy(data)
        tensor_data = tensor_data.to(torch.float32)
        tensor_data = tensor_data.to(self.device, non_blocking=True)
        tensor_data = tensor_data.unsqueeze(0)

        return tensor_data

    def _get_policy_class(self, name: str) -> PreTrainedPolicy:
        if name == 'tdmpc':
            from lerobot.policies.tdmpc.modeling_tdmpc import TDMPCPolicy

            return TDMPCPolicy
        elif name == 'diffusion':
            from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

            return DiffusionPolicy
        elif name == 'act':
            from lerobot.policies.act.modeling_act import ACTPolicy

            return ACTPolicy
        elif name == 'vqbet':
            from lerobot.policies.vqbet.modeling_vqbet import VQBeTPolicy

            return VQBeTPolicy
        elif name == 'pi0':
            from lerobot.policies.pi0.modeling_pi0 import PI0Policy

            return PI0Policy
        elif name == 'pi0fast':
            from lerobot.policies.pi0fast.modeling_pi0fast import PI0FASTPolicy
            return PI0FASTPolicy
        elif name == 'smolvla':
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
            return SmolVLAPolicy
        # TODO: Uncomment when GrootN1Policy is implemented
        # elif name == 'groot-n1':
        #     from Isaac.groot_n1.policies.groot_n1 import GrootN1Policy
        #     return GrootN1Policy
        else:
            raise NotImplementedError(
                f'Policy with name {name} is not implemented.')

    @staticmethod
    def get_available_policies() -> list[str]:
        return [
            'tdmpc',
            'diffusion',
            'act',
            'vqbet',
            'pi0',
            'pi0fast',
            'smolvla',
        ]

    @staticmethod
    def get_saved_policies():
        import os
        import json

        home_dir = os.path.expanduser('~')
        hub_dir = os.path.join(home_dir, '.cache/huggingface/hub')
        models_folder_list = [d for d in os.listdir(hub_dir) if d.startswith('models--')]

        saved_policy_path = []
        saved_policy_type = []

        for model_folder in models_folder_list:
            model_path = os.path.join(hub_dir, model_folder)
            snapshots_path = os.path.join(model_path, 'snapshots')

            # Check if snapshots directory exists
            if os.path.exists(snapshots_path) and os.path.isdir(snapshots_path):
                # Get list of folders inside snapshots directory
                snapshot_folders = [
                    d for d in os.listdir(snapshots_path)
                    if os.path.isdir(os.path.join(snapshots_path, d))
                ]

            # Check if pretrained_model folder exists in each snapshot folder
            for snapshot_folder in snapshot_folders:
                snapshot_path = os.path.join(snapshots_path, snapshot_folder)
                pretrained_model_path = os.path.join(snapshot_path, 'pretrained_model')

                # If pretrained_model folder exists, add to saved_policies
                if os.path.exists(pretrained_model_path) and os.path.isdir(pretrained_model_path):
                    config_path = os.path.join(pretrained_model_path, 'config.json')
                    if os.path.exists(config_path):
                        try:
                            with open(config_path, 'r') as f:
                                config = json.load(f)
                                if 'type' in config:
                                    saved_policy_path.append(pretrained_model_path)
                                    saved_policy_type.append(config['type'])
                                elif 'model_type' in config:
                                    saved_policy_path.append(pretrained_model_path)
                                    saved_policy_type.append(config['model_type'])
                        except (json.JSONDecodeError, IOError):
                            # If config.json cannot be read, store path only
                            print('File IO Errors : ', IOError)

        return saved_policy_path, saved_policy_type


import cv2
import threading
import rclpy
from sensor_msgs.msg import CompressedImage
from rclpy.qos import QoSProfile, ReliabilityPolicy

class KinectSingleton:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(KinectSingleton, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized: return

        if not rclpy.ok(): rclpy.init()
        self.node = rclpy.create_node('kinect_subscriber')
        self.images = {}  # 여러 토픽에 대응할 수 있도록 딕셔너리 사용
        self.data_lock = threading.Lock()

        # 백그라운드 스핀 시작
        self.spin_thread = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
        self.spin_thread.start()
        self._initialized = True

    def subscribe_topic(self, topic_name):
        """특정 토픽을 처음 호출할 때만 구독 생성"""
        with self.data_lock:
            if topic_name not in self.images:
                self.images[topic_name] = None
                self.node.create_subscription(
                    CompressedImage,
                    topic_name,
                    lambda msg: self._callback(msg, topic_name),
                    QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=1)
                )

    def _callback(self, msg, topic_name):
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        with self.data_lock:
            self.images[topic_name] = cv_img

    def get_latest(self, topic_name):
        self.subscribe_topic(topic_name) # 아직 구독 전이면 구독 시작
        with self.data_lock:
            return self.images.get(topic_name)



def get_kinect_data(topic_name: str):
    """키넥트 데이터를 가져옴"""
    return KinectSingleton().get_latest(topic_name)

def get_right_wrist_data(topic_name):
    """오른쪽 손목 카메라 데이터를 가져옴"""
    return KinectSingleton().get_latest(topic_name)



def main():
    # # ==================================================================
    # Infernce 테스트
    # # ==================================================================
    inference_manager = InferenceManager(device='cuda')


    # ==================================================================
    # 2. 모델 파일 설정
    # ==================================================================
    POLICY_PATH = "/home/jusik/TEST/test_download-dataset/model"
    if not os.path.exists(POLICY_PATH):
        print(f"\n오류: 지정된 경로를 찾을 수 없습니다: {POLICY_PATH}")


    # 모델 유효성 검사
    is_valid, message = inference_manager.validate_policy(POLICY_PATH)
    print(f"모델 유효성 검사 결과: {is_valid}, 메시지: {message}")

    if not is_valid:
        print("유효하지 않은 모델이므로 프로그램 종료됨")
        return

    # 설정: 검증된 모델을 메모리에 로드합니다.
    if not inference_manager.load_policy():
        print("모델 로드에 실패하여 테스트를 중단합니다.")
        return

    # ==================================================================
    # 3. 입력 데이터 설정
    # ==================================================================
    while True:
        kinect_image = get_kinect_data('/kinect/color/compressed')
        # right_wrist_image = get_right_wrist_data('right/') # TODO

        # 키넥트 토픽 데이터 디버깅 코드
        # if kinect_image is not None:
        #     print(f"Type: {type(kinect_image)}")      # <class 'numpy.ndarray'>
        #     print(f"Shape: {kinect_image.shape}")     # (720, 1280, 3) 등
        #     print(f"Dtype: {kinect_image.dtype}")     # uint8

        # 영상 데이터
        input_images = {
            'right_cam_wrist': np.random.randint(0, 256, size=(848, 480, 3), dtype=np.uint8),
            'cam_top': np.random.randint(0, 256, size=(1280, 720, 3), dtype=np.uint8)
        }
        # 현제 조인트 데이터
        input_state = np.random.rand(7).astype(np.float32)

        # 명령어
        input_instruction = "블록 밀기"


        # ==================================================================
        # 4. 모델 추론
        # ==================================================================
        predicted_action = inference_manager.predict(
            images=input_images,
            state=input_state.tolist(),
            task_instruction=input_instruction
        )


        # TODO: 결과를 엑션 서버에 보내 줘야함...
        if predicted_action is not None:
            pass
            print("\n--- 추론 결과 ---")
            print(f"예측된 행동: {predicted_action}")

    # Inference manager 리소스 정리
    inference_manager.clear_policy()

    print("\n 끝")



if __name__ == '__main__':
    main()
