import gradio as gr
import cv2
import numpy as np
from pathlib import Path
import time
import os
import threading
import shutil
from pynput import keyboard


# 허깅페이스 오프라인 모드 ON
os.environ["HF_HUB_OFFLINE"] = "1"

# NumPy 2.x 호환성 경고 방지를 위한 설정
os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"

# ros2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, JointState

# lerobot
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_FEATURES


# 내 모듈
from subscriber_hub import SubscriberHub


# ------------------------------------------
# 유틸리티
# ------------------------------------------
def decode_image(msg: CompressedImage):
    """압축된 이미지 메시지를 OpenCV 이미지로 변환"""
    try:
        if msg is None:
            return None
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if cv_image is None:
            return None
        cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        return cv_image_rgb
    except Exception as e:
        print(f"이미지 디코딩 오류: {e}")
        return None

def jointState_to_nparray(msg: JointState, target_names: list) -> np.ndarray:
    """JointState -> np.array"""
    if msg is None:
        return np.zeros(len(target_names), dtype=np.float32)

    # 메시지의 {이름: 위치값} 딕셔너리 생성
    name_to_pos_map = dict(zip(msg.name, msg.position))

    # target_names 순서대로 리스트 생성
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

        # 시간 및 프레임 추적을 위한 변수 추가
        self.recording_start_time = 0
        self.num_frames = 0
        self.fps = 30
        self.status = ""
        self.joint_names = [
            'right_joint1',
            'right_joint2',
            'right_joint3',
            'right_joint4',
            'right_joint5',
            'right_joint6',
            'right_rh_r1_joint'
        ]

        # 정밀 타이머 스레드 관련
        self.stop_event = threading.Event()
        self.timer_thread = None

    def init_dataset(self, repo_id="my_dataset", root_dir="data", task_name="teleop", fps=30) -> str:
        """데이터셋 초기화 및 생성"""
        self.repo_id = repo_id
        self.root_path = Path(root_dir).absolute()
        self.task_name = task_name
        self.fps = fps

        dataset_path = self.root_path / self.repo_id

        # 기존 데이터셋 폴더가 있으면 사용
        if dataset_path.exists():
            self.dataset = LeRobotDataset(repo_id=self.repo_id, root=dataset_path)
            return f"✅ 기존 데이터셋 초기화 완료"

        # 새 데이터셋 생성
        features = DEFAULT_FEATURES.copy()
        features[f'observation.images.cam_top'] = {
            'dtype': 'video',
            'names': ['height', 'width', 'channels'],
            'shape': (720, 1280, 3)
        }
        features[f'observation.images.cam_wrist'] = {
            'dtype': 'video',
            'names': ['height', 'width', 'channels'],
            'shape': (480, 848, 3)
        }
        features[f'observation.state'] = {
            'dtype': 'float32',
            'names': self.joint_names,
            'shape': (7,)
        }
        features[f'action'] = {
            'dtype': 'float32',
            'names': self.joint_names,
            'shape': (7,)
        }

        self.dataset = LeRobotDataset.create(
            repo_id=self.repo_id,
            root=dataset_path,
            features=features,
            use_videos=True,
            fps=fps,
            robot_type="omy_f3m",
        )

        print(f"[Info] 데이터셋 초기화 성공 {self.repo_id}")
        return f"✅ 데이터셋 초기화 완료"

    def start_timer(self):
        """녹화 타이머 시작"""
        if self.timer_thread is not None and self.timer_thread.is_alive():
            return

        self.stop_event.clear()
        self.timer_thread = threading.Thread(target=self._timer_loop, daemon=True)
        self.timer_thread.start()

        self.status = "ready"

        print("[Info] 녹화 준비 완료")

    def stop_timer(self):
        """녹화 타이머 스레드 중지"""
        self.stop_event.set()
        if self.timer_thread:
            self.timer_thread.join()

        self.status = ""

        print("[Info] 녹화 타이머 중지")

    def _timer_loop(self):
        """녹화 타이머 루프 (별도 스레드)"""
        interval = 1.0 / self.fps
        next_time = time.time()

        while not self.stop_event.is_set():
            # record 함수 실행
            kinect_msg, wrist_msg, follower_msg, leader_msg = self.subscriber_hub.get_latest_msg()
            self._record_loop(kinect_msg, wrist_msg, follower_msg, leader_msg)

            # 정밀 타이머 대기
            next_time += interval
            sleep_time = next_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # 루프가 너무 느려진 경우 현재 시간으로 보정
                next_time = time.time()

    def _record_loop(self, kinect_msg, wrist_msg, follower_msg, leader_msg):
        """데이터 녹화 및 상태 처리"""
        if self.dataset is None:
            print(f"[Warn ] 데이터 녹화 실패: 데이터셋 초기화 필요")
            return

        if self.status == "record":
            # 녹화 시작 시점 시간 기록
            if self.recording_start_time == 0:
                self.recording_start_time = time.time()

            # 데이터 변환
            kinect_img = decode_image(kinect_msg)
            wrist_img = decode_image(wrist_msg)
            follower_joint_data = jointState_to_nparray(follower_msg, self.joint_names)
            leader_joint_data = jointState_to_nparray(leader_msg, self.joint_names)

            # 프레임 생성 및 추가
            frame = {}
            frame['observation.images.cam_top'] = kinect_img
            frame['observation.images.cam_wrist'] = wrist_img
            frame['observation.state'] = follower_joint_data
            frame['action'] = leader_joint_data
            frame['task'] = self.task_name

            # 데이터셋에 프레임 추가 (Thread-safe 고려 필요 시 Lock 사용)
            self.dataset.add_frame(frame)
            self.num_frames += 1 # 프레임 카운트 증가

            # 경과 시간 계산
            elapsed_time = time.time() - self.recording_start_time
            if self.num_frames % 10 == 0: # 너무 잦은 출력 방지
                print(f"🔴 녹화 중: {elapsed_time:.1f}초 ({self.num_frames} 프레임)")

        elif self.status == "save":
            print(f"[Info] 에피소드 저장 중... ({self.num_frames} 프레임)")
            self.dataset.save_episode()
            self.dataset.finalize()
            self.status = "save complete"
            self.recording_start_time = 0
            self.num_frames = 0
            print("[Info] 에피소드 저장 완료")

        elif self.status == "retry":
            print(f"[Info] 에피소드 저장 재시도 중...")
            self.dataset.clear_episode_buffer()
            self.status = "ready"
            self.recording_start_time = 0
            self.num_frames = 0
            print("[Info] 에피소드 저장 재시도 완료")
            return f"✅ 에피소드 저장 재시도 완료!"

    def record(self):
        """녹화 상태로 변경"""

        if self.status == "":
            print(f"[Warn ] 데이터 녹화 실패: 초기화 필요")
            return f"❌ 데이터 녹화 실패: 초기화 필요"

        elif self.status == "record":
            print(f"[Warn ] 데이터 녹화 실패: 이미 녹화 중")
            return f"이미 녹화 중..."

        elif self.status == "save":
            print(f"[Warn ] 데이터 녹화 실패: 에피소드 저장 중")
            return f"에피소드 저장 중..."

        self.status = "record"
        return f"✅ 데이터 녹화 중..."

    def save(self):
        """에피소드 저장 상태로 변경"""
        if self.status == "":
            print(f"[Warn ] 에피소드 저장 실패: 초기화 필요")
            return f"❌ 에피소드 저장 실패: 초기화 필요"

        elif self.status == "save":
            print(f"[Warn ] 에피소드 저장 실패: 이미 저장 중")
            return f"에피소드 저장 중..."

        elif self.num_frames == 0:
            print(f"[Warn ] 에피소드 저장 실패: 녹화된 프레임 없음")
            return f"❌ 에피소드 저장 실패: 녹화된 프레임 없음"

        self.status = "save"
        print(f"[Info] 에피소드 저장 요청")
        return f"✅ 에피소드 저장 중..."

    def retry(self):
        self.status = "retry"
        print(f"[Info] 에피소드 저장 재시도 요청")
        return f"✅ 에피소드 저장 재시도 중..."




# ------------------------------------------
# 웹 인터페이스
# ------------------------------------------
class GradioWeb:
    def __init__(self, hub: SubscriberHub):
        self.hub = hub
        self.dataset_manager = LerobotDatasetManager(hub)
        self.interface = self.build_interface()

        # 채터링 방지 및 키 상태 관리를 위한 변수
        self.last_key_time = 0
        self.chatter_threshold = 0.2
        self.right_pressed = False
        self.left_pressed = False

        # 키보드 리스너 시작
        self.listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self.listener.start()


    def _format_joint_state(self, msg: JointState):
        """JointState 메시지를 텍스트로 포맷팅"""
        if msg is None:
            return "No Data"

        lines = []
        for name, pos in zip(msg.name, msg.position):
            pos_degrees = pos * 180.0 / 3.14159265359 # deg
            lines.append(f"{name}: {pos_degrees:.4f}°")
        return "\n".join(lines)

    def update_tick(self, current_status_text: str):
        """UI 업데이트를 위한 Timer 틱 함수"""
        kinect_msg, wrist_msg, follower_msg, leader_msg = self.hub.get_latest_msg()

        # 이미지 디코딩 (UI 표시용)
        kinect_img = decode_image(kinect_msg)
        wrist_img = decode_image(wrist_msg)

        # 조인트 데이터 텍스트 변환
        follower_text = self._format_joint_state(follower_msg)
        leader_text = self._format_joint_state(leader_msg)

        # 상태 텍스트 업데이트
        new_status_text = current_status_text

        # 백그라운드에서 'save complete' 상태가 되면
        if self.dataset_manager.status == "save complete":
            self.status_output = "✅ 에피소드 저장 완료!"
            self.dataset_manager.status = "ready"

        return kinect_img, wrist_img, follower_text, leader_text, self.status_output

    def _on_press(self, key):
        # 데이터셋이 초기화된 상태에서만 키 입력 처리
        if self.dataset_manager.dataset is None:
            return

        # 채터링 방지
        current_time = time.time()
        if current_time - self.last_key_time < self.chatter_threshold:
            return

        # 오른쪽 방향키: 녹화 시작/저장
        if key == keyboard.Key.right:
            if not self.right_pressed:
                self.right_pressed = True
                self.last_key_time = current_time

                # 데이터 메니저 상태
                status = self.dataset_manager.status

                if status == "":
                    print(f"[Warn ] 데이터 녹화 실패: 초기화 필요")
                    self.status_output = "❌ 데이터 녹화 실패: 초기화 필요"
                    return

                elif status == "ready":
                    result = self.dataset_manager.record()
                    self.status_output = result

                elif status == "record" or status == "save":
                    result = self.dataset_manager.save()
                    self.status_output = result





    def _on_release(self, key):
        if key == keyboard.Key.right:
            self.right_pressed = False
        elif key == keyboard.Key.left:
            self.left_pressed = False


    def handle_init(self, repo_id, root_dir, task_name, fps):
        """Init 버튼 클릭 시 데이터셋 초기화 실행"""
        result = self.dataset_manager.init_dataset(repo_id, root_dir, task_name, int(fps))
        # 데이터셋 초기화 후 타이머 스레드 시작
        self.dataset_manager.start_timer()
        return result

    def handle_record(self):
        """Record 버튼 클릭시 이벤트"""
        result = self.dataset_manager.record()
        return result

    def handle_save_episode(self):
        """Save 버튼 클릭시 이벤트"""
        result = self.dataset_manager.save()
        return result

    def handle_retry(self):
        """Retry 버튼 클릭시 이벤트"""
        result = self.dataset_manager.retry()
        return result

    def build_interface(self):
        """Gradio 인터페이스 구성"""
        with gr.Blocks(title="Robot Teleoperation Monitor") as demo:
            gr.Markdown("# Robot Teleoperation Monitor")

            timer = gr.Timer(0.1)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Dataset Configuration")
                    repo_id_input = gr.Textbox(label="Repo ID", value="my_dataset")
                    root_dir_input = gr.Textbox(label="Root Directory", value="data")
                    task_name_input = gr.Textbox(label="Task Name", value="teleop")
                    fps_input = gr.Number(label="FPS", value=30)
                    init_btn = gr.Button("Init Dataset", variant="primary")
                    record_btn = gr.Button("Record", variant="primary")
                    save_btn = gr.Button("Save", variant="primary")
                    retry_btn = gr.Button("Retry", variant="primary")
                    self.status_output = gr.Textbox(label="Status", interactive=False)

                with gr.Column(scale=2):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Kinect Camera")
                            kinect_view = gr.Image(label="Top View")
                        with gr.Column():
                            gr.Markdown("### Wrist Camera")
                            wrist_view = gr.Image(label="Wrist View")
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Follower Joint States")
                            follower_view = gr.Textbox(label="Follower Joints", lines=7)
                        with gr.Column():
                            gr.Markdown("### Leader Joint States")
                            leader_view = gr.Textbox(label="Leader Joints", lines=7)

            # 버튼 클릭 이벤트 연결
            init_btn.click(
                self.handle_init,
                [repo_id_input, root_dir_input, task_name_input, fps_input],
                self.status_output
            )
            record_btn.click(
                self.handle_record,
                None,
                self.status_output
            )
            save_btn.click(
                self.handle_save_episode,
                None,
                self.status_output
            )
            retry_btn.click(
                self.handle_retry,
                None,
                self.status_output
            )

            # UI 업데이트 타이머 연결
            timer.tick(
                self.update_tick,
                inputs=[self.status_output],
                outputs=[kinect_view, wrist_view, follower_view, leader_view, self.status_output]
            )

        return demo

    def launch(self):
        """Gradio 앱 실행"""
        self.interface.launch(server_name="0.0.0.0", share=False)


def main():
    # ROS2 초기화
    rclpy.init()

    # 허브 노드 생성
    hub = SubscriberHub()

    # ROS2 스핀을 별도 스레드에서 실행
    ros_thread = threading.Thread(target=rclpy.spin, args=(hub,), daemon=True)
    ros_thread.start()

    try:
        # Gradio 웹 인터페이스 실행
        web = GradioWeb(hub)
        web.launch()
    except KeyboardInterrupt:
        pass
    finally:
        hub.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
