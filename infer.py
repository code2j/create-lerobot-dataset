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
        """ Raw 이미지 데이터를 학습용 텐서로 변환 (numpy -> torch) """

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
        """ numpy 데이터(센서, State)를 학습용 텐서로 변환 (numpy -> torch) """
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

from subscriber_hub import SubscriberHub
from data_converter import decode_image

import time # 상단에 import 추가
def main():

    if not rclpy.ok():
        rclpy.init()

    # 데이터 수집 쓰레드 시작
    hub = SubscriberHub()
    threading.Thread(target=lambda: rclpy.spin(hub), daemon=True).start()


    def get_latest_data():
        kinect_msg, right_wrist_cam_msg, right_follower_msg, _ = hub.get_latest_msg()

        # 디코딩
        kinect_img = decode_image(kinect_msg)
        right_wrist_img = decode_image(right_wrist_cam_msg)

        # JointStates -> np.array(7)
        follower_joint_data = np.array(right_follower_msg.position, dtype=np.float32)

        return kinect_img, right_wrist_img, follower_joint_data



    # 1. Inference Manager 초기화
    inference_manager = InferenceManager(device='cuda')

    # 2. 모델 파일 설정 및 로드
    POLICY_PATH = "/home/jusik/TEST/test_download-dataset/model"
    is_valid, message = inference_manager.validate_policy(POLICY_PATH)
    print(f"모델 유효성 검사 결과: {is_valid}, 메시지: {message}")

    if not is_valid or not inference_manager.load_policy():
        print("모델 로드 실패로 종료합니다.")
        return

    # ==================================================================
    # 30 FPS 루프 설정
    # ==================================================================
    target_fps = 30
    target_period = 1.0 / target_fps  # 약 0.0333초

    print(f"\n[시작] {target_fps} FPS 주기로 추론을 시작합니다.")

    try:
        while True:
            start_time = time.perf_counter() # 고정밀 타이머 시작

            # 1) 실시간 데이터 가져오기
            kinect_image, wrist_image, follower_joint_data = get_latest_data()


            # 데이터가 아직 안 들어왔을 경우 처리
            if kinect_image is None:
                # print("데이터 대기 중...")
                time.sleep(0.01)
                continue

            # 2) 입력 데이터 구성 (실제 데이터로 교체)
            input_images = {
                'cam_top': kinect_image, # 예시 해상도
                'right_cam_wrist': wrist_image, # 임시
            }

            # 현재 조인트 데이터 (실제 로봇 상태 데이터로 교체 필요)
            input_state = follower_joint_data
            input_instruction = "pick the zipper bag"

            # 3) 모델 추론
            predicted_action = inference_manager.predict(
                images=input_images,
                state=input_state.tolist(),
                task_instruction=input_instruction
            )

            # 4) 결과 출력 (또는 액션 서버 전송)
            if predicted_action is not None:
                print(f"Action: {predicted_action}")
                pass

            # 5) 30 FPS 유지를 위한 정밀 대기
            elapsed_time = time.perf_counter() - start_time
            sleep_time = target_period - elapsed_time

            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # 추론 시간이 1/30초를 초과한 경우 경고
                fps_actual = 1.0 / (time.perf_counter() - start_time)
                print(f"Warning: Inference slow! Actual FPS: {fps_actual:.2f}")

    except KeyboardInterrupt:
        print("\n사용자에 의해 중단되었습니다.")
    finally:
        inference_manager.clear_policy()
        print("리소스 정리 완료.")

if __name__ == '__main__':
    main()
