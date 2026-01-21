import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np

# 허깅페이스 오프라인 모드 ON
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"

# ROS2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, JointState

# Lerobot
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_FEATURES

# 내 모듈
from subscriber_hub import SubscriberHub


# ------------------------------------------
# 유틸리티
# ------------------------------------------
def decode_image(msg: CompressedImage):
    try:
        if msg is None: return None
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if cv_image is None: return None
        return cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"이미지 디코딩 오류: {e}")
        return None

def jointState_to_nparray(msg: JointState, target_names: list) -> np.ndarray:
    if msg is None: return np.zeros(len(target_names), dtype=np.float32)
    name_to_pos_map = dict(zip(msg.name, msg.position))
    ordered_values = [name_to_pos_map.get(name, 0.0) for name in target_names]
    return np.array(ordered_values, dtype=np.float32)

# ------------------------------------------
# Lerobot 데이터 매니저
# ------------------------------------------
class LerobotDatasetManager:
    def __init__(self, subscriber_hub: SubscriberHub):
        self.subscriber_hub = subscriber_hub
        self.dataset = None
        self.lock = threading.Lock()
        self.recording_start_time = 0
        self.num_frames = 0
        self.fps = 30

        self.status = ""
        self.last_save_result_message = ""

        self.joint_names = [
            'right_joint1',
            'right_joint2',
            'right_joint3',
            'right_joint4',
            'right_joint5',
            'right_joint6',
            'right_rh_r1_joint'
        ]

        self.stop_event = threading.Event()
        self.timer_thread = None

    def init_dataset(self, repo_id="my_dataset", root_dir="data", task_name="teleop", fps=30) -> str:
        self.repo_id = repo_id
        self.root_path = Path(root_dir).absolute() / self.repo_id
        self.task_name = task_name
        self.fps = fps
        meta_info_path = self.root_path  / "meta" / "info.json"

        # 이미 데이터셋이 있다면 기존 데이터셋 불러오기
        if meta_info_path.exists():
            self.dataset = LeRobotDataset(repo_id=self.repo_id, root=self.root_path)
            self.dataset.start_image_writer(num_processes=2, num_threads=4) # 프로세서 늘리기
            return f"✅ 기존 데이터셋 불러오기 성공"

        # 데이터셋 생성 및 초기화
        features = DEFAULT_FEATURES.copy()
        features['observation.images.cam_top'] = {
            'dtype': 'video',
            'names': ['height', 'width', 'channels'],
            'shape': (720, 1280, 3)
        }
        features['observation.images.cam_wrist'] = {
            'dtype': 'video',
            'names': ['height', 'width', 'channels'],
            'shape': (480, 848, 3)
        }
        features['observation.state'] = {
            'dtype': 'float32',
            'names': self.joint_names,
            'shape': (7,)
        }
        features['action'] = {
            'dtype': 'float32',
            'names': self.joint_names,
            'shape': (7,)
        }

        # 생성
        self.dataset = LeRobotDataset.create(
            repo_id=self.repo_id,
            root=self.root_path,
            features=features,
            use_videos=True,
            fps=fps,
            robot_type="omy_f3m",
            image_writer_threads=4,
            image_writer_processes=2
        )
        return f"✅ 데이터셋 초기화 완료"

    def start_timer(self):
        """Recording Loop 타이머 시작"""
        if self.timer_thread is not None and self.timer_thread.is_alive():
            return # 이미 타이머가 실행 중인 경우 무시

        self.stop_event.clear()
        self.timer_thread = threading.Thread(target=self._timer_loop, daemon=True)
        self.timer_thread.start()

        # 데이터 수집이 가능한 상태
        self.status = "ready"

    def _timer_loop(self):
        """Recording Loop(메인 루프)"""
        interval = 1.0 / self.fps
        next_time = time.time()

        # 메인 루프
        while not self.stop_event.is_set():
            # 최신 메세지 토픽 받기
            k_msg, w_msg, f_msg, l_msg = self.subscriber_hub.get_latest_msg()

            # 녹화 함수
            self._record_loop(k_msg, w_msg, f_msg, l_msg)

            # 타이머 속도 조절
            next_time += interval
            sleep_time = next_time - time.time()

            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_time = time.time()

    def _record_loop(self, kinect_msg, wrist_msg, follower_msg, leader_msg):
        if self.dataset is None: return
        with self.lock:
            # 데이터 녹화중
            if self.status == "record":
                if self.recording_start_time == 0:
                    self.recording_start_time = time.time() # 에피소드 녹화 시간 측정

                # 데이터 변환
                k_img = decode_image(kinect_msg) # 디코딩
                w_img = decode_image(wrist_msg)  # 디코딩
                f_joint = jointState_to_nparray(follower_msg, self.joint_names) # 넘파이로 변환
                l_joint = jointState_to_nparray(leader_msg, self.joint_names)   # 넘파이로 변환

                # 프레임 생성
                frame = {}
                frame['observation.images.cam_top'] = k_img
                frame['observation.images.cam_wrist'] = w_img
                frame['observation.state'] = f_joint
                frame['action'] = l_joint
                frame['task'] = self.task_name

                # 프레임 추가
                self.dataset.add_frame(frame)
                self.num_frames += 1

            # 재시도중
            elif self.status == "retry":
                self.dataset.clear_episode_buffer()
                self.status = "ready"
                self.recording_start_time, self.num_frames = 0, 0

    def _save_episode_threaded(self):
        try:
            self.dataset.save_episode()
            # finalize()는 여기서 호출하지 않음
            message = f"✅ 에피소드 저장 완료 (누적 에피소드: {self.dataset.num_episodes})"
        except Exception as e:
            message = f"❌ 에피소드 저장 실패: {e}"
        with self.lock:
            self.status = "ready"
            self.recording_start_time, self.num_frames = 0, 0
            self.last_save_result_message = message

    def _finalize_threaded(self):
        try:
            self.dataset.finalize()
            message = "🏁 데이터셋 파이널라이즈 완료! (업로드/학습 준비 완료)"
        except Exception as e:
            message = f"❌ 파이널라이즈 실패: {e}"
        with self.lock:
            self.status = "ready"
            self.last_save_result_message = message

    def record(self):
        with self.lock:
            if self.status == "record": return "✅ 이미 녹화 중입니다."
            if self.status in ["saving", "finalizing"]: return "⏳ 작업 중... 잠시 기다려주세요"
            self.status = "record"
            self.last_save_result_message = ""
            return "🔴 녹화 시작됨"

    def save(self):
        with self.lock:
            if self.status == "saving": return "⏳ 이미 에피소드 저장 중입니다."
            if self.num_frames == 0: return "❌ 저장할 데이터가 없습니다."
            self.status = "saving"
            threading.Thread(target=self._save_episode_threaded).start()
            return "⏳ 에피소드 저장 중..."

    def retry(self):
        with self.lock:
            if self.status in ["saving", "finalizing"]: return "❌ 현재 작업 중에 재시도 불가"
            self.status = "retry"
            return "⏳ 현재 녹화된 에피소드 제거 중..."

    def finalize_dataset(self):
        with self.lock:
            if self.status in ["record", "saving", "finalizing"]: return "❌ 녹화/저장 중에는 파이널라이즈 불가"
            if self.dataset is None: return "❌ 초기화 필요"
            self.status = "finalizing"
            threading.Thread(target=self._finalize_threaded).start()
            return "⏳ 데이터셋 최종 확정 중 (비디오 인코딩)..."

    def get_display_status(self):
        with self.lock:
            if self.last_save_result_message:
                msg = self.last_save_result_message
                self.last_save_result_message = ""
                return msg
            if self.status == "record":
                return f"🔴 녹화 중: {time.time()-self.recording_start_time:.1f}초 ({self.num_frames}f)"
            elif self.status == "saving": return "⏳ 에피소드 저장 중..."
            elif self.status == "finalizing": return "⏳ 파이널라이즈 중 (종료 대기)..."
            elif self.status == "ready": return "✅ 대기 중 (준비 완료)"
            return "초기화되지 않음"