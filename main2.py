import gradio as gr
import cv2
import numpy as np
from pathlib import Path
import threading
import time
import os
import shutil
import subprocess
import signal

# 허깅페이스 오프라인 모드 ON
os.environ["HF_HUB_OFFLINE"] = "1"

# NumPy 2.x 호환성 경고 방지를 위한 설정
os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"

# ros2
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, JointState

# lerobot
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_FEATURES

# 내가 만든 모듈
from subscriber_hub import SubscriberHub

def decode_image(msg: CompressedImage):
    """압축된 이미지 메시지를 OpenCV 이미지로 변환 및 실행 시간 출력"""
    start_time = time.perf_counter()  # 측정 시작

    try:
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        # 실행 시간 계산 (초 단위 -> 밀리초 단위로 변환)
        end_time = time.perf_counter()
        elapsed_time = (end_time - start_time) * 1000
        print(f"이미지 디코딩 소요 시간: {elapsed_time:.2f} ms")

        return cv_image_rgb
    except Exception as e:
        print(f"이미지 디코딩 오류: {e}")
        return None

def decode_image_for_rendering(msg: CompressedImage):
    """UI 출력용 디코딩"""
    try:
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        return cv_image_rgb
    except Exception as e:
        print(f"이미지 디코딩 오류: {e}")
        return None


class Dataset_manager:
    def __init__(self, subscriber_hub: SubscriberHub):
        self.subscriber_hub = subscriber_hub
        self.dataset = None
        self.is_recording = False
        self.running = True
        self.lock = threading.Lock()

        self.max_record_time = 0
        self.start_time = 0
        self.fps = 30

        # 에피소드 구분을 위한 ID 관리
        self.current_episode_id = 0
        self.is_canceled = False # 현재 에피소드 취소 여부

        self.joint_names = [
            'right_joint1', 'right_joint2', 'right_joint3',
            'right_joint4', 'right_joint5', 'right_joint6',
            'right_rh_r1_joint'
        ]

        # 단일 녹화 스레드: 데이터를 수집하고 즉시 처리
        self.record_thread = threading.Thread(target=self._recording_loop, daemon=True)
        self.record_thread.start()

        print("[Info ] 녹화 스레드가 시작되었습니다 (직접 처리 방식)")

    def init_dataset(self, repo_id, root_dir, task_name, fps) -> str:
        """데이터셋 초기화 및 생성"""
        with self.lock:
            self.repo_id = repo_id
            self.root_path = Path(root_dir).absolute()
            self.task_name = task_name
            self.fps = fps


            dataset_path = self.root_path / self.repo_id
            info_json = dataset_path / "meta" / "info.json"

            if info_json.exists():
                self.dataset = LeRobotDataset(repo_id=self.repo_id, root=dataset_path)
                print(f"[Info ] 기존 데이터셋을 로드함")
                return "📂 기존 데이터셋을 로드했습니다."
            else:
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

                print(f"[Info ] 데이터셋이 성공적으로 초기화되었습니다.")
                return "✅ 데이터셋 생성"

    def _recording_loop(self):
        """데이터 수집 및 즉시 디코딩/저장 루프"""
        next_time = time.time()

        while self.running:
            if self.is_recording and self.dataset is not None:
                frame_interval = 1.0 / self.fps

                if self.max_record_time > 0:
                    if time.time() - self.start_time >= self.max_record_time:
                        self.stop_recording()
                        continue

                # 1. 데이터 수집
                raw_data = self.subscriber_hub.get_latest_msg()

                # 2. 즉시 처리 (디코딩 및 데이터셋 추가)
                if not self.is_canceled:
                    try:
                        kinect_msg, wrist_msg, follow_msg, leader_msg = raw_data

                        # 이미지 디코딩
                        kinect_img = decode_image(kinect_msg)
                        wrist_img = decode_image(wrist_msg)

                        # 팔로워(State) 데이터 정렬
                        follow_map = dict(zip(follow_msg.name, follow_msg.position))
                        follower_joint_data = np.array([follow_map[name] for name in self.joint_names], dtype=np.float32)

                        # 리더(Action) 데이터 정렬
                        leader_map = dict(zip(leader_msg.name, leader_msg.position))
                        leader_joint_data = np.array([leader_map[name] for name in self.joint_names], dtype=np.float32)

                        # 데이터셋 추가
                        if kinect_img is not None and wrist_img is not None:
                            with self.lock:
                                self.dataset.add_frame({
                                    'observation.images.cam_top': kinect_img,
                                    'observation.images.cam_wrist': wrist_img,
                                    'observation.state': follower_joint_data,
                                    'action': leader_joint_data,
                                    'task': self.task_name
                                })
                    except Exception as e:
                        print(f"[Error] 데이터 처리 중 오류: {e}")

                # 3. 정밀 타이밍 제어
                next_time += frame_interval
                sleep_time = next_time - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    next_time = time.time()
            else:
                time.sleep(0.1)
                next_time = time.time()

    def start_recording(self, max_time=0):
        if self.dataset is None:
            return "❌ 오류: 데이터셋이 초기화되지 않았습니다."

        if self.is_recording:
            return "⚠️ 시스템: 이미 녹화 중입니다."

        self.max_record_time = max_time
        self.start_time = time.time()
        self.is_canceled = False

        # 새로운 에피소드 시작 시 ID 증가
        self.current_episode_id += 1
        self.is_recording = True

        msg = "녹화 중..."
        print(f"[Info ] {msg} (Episode ID: {self.current_episode_id})")
        return msg

    def stop_recording(self):
        if not self.is_recording:
            return "⚠️ 시스템: 현재 녹화 중이 아닙니다."

        self.is_recording = False

        # 즉시 저장 (이미 add_frame이 완료된 상태이므로)
        try:
            with self.lock:
                self.dataset.save_episode()
            print(f"[Info ] 에피소드 {self.current_episode_id} 저장 완료")
            return "✅ 에피소드 저장 완료"
        except Exception as e:
            print(f"[Error] 에피소드 저장 중 오류: {e}")
            return f"❌ 저장 오류: {e}"

    def cancel_recording(self):
        """현재 녹화를 취소"""
        if not self.is_recording:
            return "⚠️ 시스템: 현재 녹화 중이 아닙니다."

        self.is_recording = False
        self.is_canceled = True

        msg = "현재 에피소드 녹화 취소됨"
        print(f"[Info ] {msg} (ID: {self.current_episode_id})")
        return msg

    def finalize_dataset(self):
        """데이터 수집 완료 및 데이터셋 최종화"""
        if self.dataset is None:
            return "❌ 오류: 데이터셋이 초기화되지 않았습니다."

        if self.is_recording:
            return "❌ 오류: 녹화 중에는 데이터셋을 완료할 수 없습니다."

        try:
            self.dataset.finalize()
            # 최종화 후 데이터셋 객체를 None으로 설정하여 키 리스너가 동작하지 않게 함
            self.dataset = None
            msg = "✅ 데이터 수집 완료 및 데이터셋 최종화 성공"
            print(msg)
            return msg
        except Exception as e:
            msg = f"❌ 시스템: 데이터셋 최종화 중 오류 발생: {e}"
            print(msg)
            return msg

    def close(self):
        self.running = False
        if self.record_thread.is_alive():
            self.record_thread.join()
        print("시스템: 모든 쓰레드가 종료되었습니다.")


from pynput import keyboard
class GradioVisualizer:
    def __init__(self, subscriber_hub: SubscriberHub):
        self.subscriber_hub = subscriber_hub
        self.update_interval = 1/30
        self.dataset_manager = Dataset_manager(self.subscriber_hub)

        # 채터링 방지 및 키 상태 관리를 위한 변수
        self.last_key_time = 0
        self.chatter_threshold = 0.2
        self.right_pressed = False
        self.left_pressed = False

        # 상태 메시지 관리
        self.current_status = "✅ 대기 중"

        # 키보드 리스너 시작
        self.listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self.listener.start()

    def _on_press(self, key):
        try:
            # 데이터셋이 초기화된 상태에서만 키 입력 처리
            if self.dataset_manager.dataset is None:
                return

            current_time = time.time()
            if current_time - self.last_key_time < self.chatter_threshold:
                return

            # 오른쪽 방향키: 녹화 토글
            if key == keyboard.Key.right:
                if not self.right_pressed:
                    self.right_pressed = True
                    self.last_key_time = current_time
                    self._toggle_recording()
            # 왼쪽 방향키: 녹화 취소
            elif key == keyboard.Key.left:
                if not self.left_pressed:
                    self.left_pressed = True
                    self.last_key_time = current_time
                    self._re_record()
        except Exception as e:
            print(f"[Error] Key press handling error: {e}")

    def _on_release(self, key):
        try:
            if key == keyboard.Key.right:
                self.right_pressed = False
            elif key == keyboard.Key.left:
                self.left_pressed = False
        except Exception as e:
            pass

    def _toggle_recording(self):
        """녹화 상태를 토글"""
        if self.dataset_manager.is_recording:
            self.current_status = self.dataset_manager.stop_recording()
        else:
            self.current_status = self.dataset_manager.start_recording(max_time=0)

    def _re_record(self):
        """현재 녹화를 취소"""
        if self.dataset_manager.is_recording:
            self.current_status = self.dataset_manager.cancel_recording()
        else:
            self.current_status = "⚠️ 시스템: 현재 녹화 중이 아닙니다."

    def ui_timer_callback(self):
        (k_msg, w_msg, f_joint, l_joint) = self.subscriber_hub.get_latest_msg()
        k_img = decode_image_for_rendering(k_msg)
        w_img = decode_image_for_rendering(w_msg)

        desired_names = ['right_joint1', 'right_joint2', 'right_joint3', 'right_joint4', 'right_joint5', 'right_joint6', 'right_rh_r1_joint']

        follower_text = "N/A"
        if f_joint is not None:
            f_vals = [np.rad2deg(f_joint.position[f_joint.name.index(n)]) if n in f_joint.name else np.nan for n in desired_names]
            follower_text = f"J1: {f_vals[0]:.1f} J2: {f_vals[1]:.1f} J3: {f_vals[2]:.1f} J4: {f_vals[3]:.1f} J5: {f_vals[4]:.1f} J6: {f_vals[5]:.1f} G: {f_vals[6]:.1f}"

        leader_text = "N/A"
        if l_joint is not None:
            l_vals = [np.rad2deg(l_joint.position[l_joint.name.index(n)]) if n in l_joint.name else np.nan for n in desired_names]
            leader_text = f"J1: {l_vals[0]:.1f} J2: {l_vals[1]:.1f} J3: {l_vals[2]:.1f} J4: {l_vals[3]:.1f} J5: {l_vals[4]:.1f} J6: {l_vals[5]:.1f} G: {l_vals[6]:.1f}"

        display_status = self.current_status

        if self.dataset_manager.is_recording:
            elapsed = time.time() - self.dataset_manager.start_time
            display_status = f"{self.current_status} ({elapsed:.1f}s)"

        return k_img, w_img, follower_text, leader_text, display_status

    def handle_init(self, repo_id, root_dir, task_name, fps):
        res = self.dataset_manager.init_dataset(repo_id, root_dir, task_name, fps)
        self.current_status = res
        return res

    def handle_record(self, max_time):
        res = self.dataset_manager.start_recording(max_time)
        self.current_status = res
        return res

    def handle_re_record(self):
        """재녹화 버튼 핸들러 (취소만 수행하도록 수정)"""
        if self.dataset_manager.is_recording:
            res = self.dataset_manager.cancel_recording()
        else:
            res = "⚠️ 시스템: 현재 녹화 중이 아닙니다."
        self.current_status = res
        return res

    def handle_stop(self):
        res = self.dataset_manager.stop_recording()
        self.current_status = res
        return res

    def handle_finalize(self):
        res = self.dataset_manager.finalize_dataset()
        self.current_status = res
        return res

    def create_interface(self):
        default_root_dir = os.path.join(os.getcwd(), "dataset")

        with gr.Blocks(title="Robot Data Collector") as demo:
            gr.Markdown("# 🤖 Robot Data Collector")

            with gr.Row():
                kinect_image = gr.Image(label="Kinect", type="numpy")
                wrist_image = gr.Image(label="Wrist", type="numpy")

            with gr.Row():
                follower_joint_output = gr.Textbox(label="Follower Arm Joints")
                leader_joint_output = gr.Textbox(label="Leader Arm Joints")

            with gr.Row():
                repo_id_input = gr.Textbox(label="Repo ID", value="test_dataset")
                root_dir_input = gr.Textbox(label="Root Path", value=default_root_dir)
                task_name_input = gr.Textbox(label="Task", value="test_task")
                fps_input = gr.Number(label="FPS", value=30)
                max_time_input = gr.Number(label="Max Time", value=0)

            init_btn = gr.Button("Initialize")
            status_output = gr.Textbox(label="Status", value=self.current_status)

            with gr.Row():
                record_btn = gr.Button("Record (Right Arrow)", variant="primary")
                re_record_btn = gr.Button("Cancel Recording (Left Arrow)", variant="secondary")
                stop_btn = gr.Button("Stop", variant="stop")
                finalize_btn = gr.Button("데이터 수집 완료 (Finalize)", variant="secondary")

            init_btn.click(self.handle_init, [repo_id_input, root_dir_input, task_name_input, fps_input], status_output)
            record_btn.click(self.handle_record, [max_time_input], status_output)
            re_record_btn.click(self.handle_re_record, outputs=status_output)
            stop_btn.click(self.handle_stop, outputs=status_output)
            finalize_btn.click(self.handle_finalize, outputs=status_output)

            timer = gr.Timer(value=self.update_interval)
            timer.tick(
                self.ui_timer_callback,
                outputs=[
                    kinect_image, wrist_image, follower_joint_output, leader_joint_output,
                    status_output
                ]
            )

        return demo

    def launch(self):
        demo = self.create_interface()
        demo.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    import rclpy
    rclpy.init()
    hub = SubscriberHub()
    threading.Thread(target=lambda: rclpy.spin(hub), daemon=True).start()

    visualizer = GradioVisualizer(hub)
    try:
        visualizer.launch()
    except KeyboardInterrupt:
        visualizer.dataset_manager.close()
        rclpy.shutdown()
