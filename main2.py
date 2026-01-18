import gradio as gr
import cv2
import numpy as np
from pathlib import Path
import threading
import time
import queue
import os
import shutil

# NumPy 2.x 호환성 경고 방지를 위한 설정
os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"

# ros2
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, JointState

# lerobot
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_FEATURES

from data_converter import decode_image
from subscriber_hub import SubscriberHub

class Dataset_manager:
    def __init__(self, subscriber_hub: SubscriberHub):
        self.subscriber_hub = subscriber_hub
        self.dataset = None
        self.is_recording = False
        self.running = True
        self.lock = threading.Lock()

        # 비동기 처리를 위한 큐 추가
        self.data_queue = queue.Queue()

        self.max_record_time = 0
        self.start_time = 0
        self.fps = 30

        # 1. 생산자 쓰레드: 데이터를 수집하여 큐에 넣음
        self.record_thread = threading.Thread(target=self._recording_loop, daemon=True)
        self.record_thread.start()

        # 2. 소비자 쓰레드: 큐에서 데이터를 꺼내 디코딩 및 저장
        self.consumer_thread = threading.Thread(target=self._consumer_loop, daemon=True)
        self.consumer_thread.start()

        print("[Info ] 녹화 및 소비자 쓰레드가 시작되었습니다.")

    def init_dataset(self, repo_id, root_dir, task_name, fps) -> str:
        """데이터셋 초기화 및 생성"""
        with self.lock:
            self.repo_id = repo_id
            self.root_path = Path(root_dir).absolute()
            self.task_name = task_name
            self.fps = fps

            dataset_path = self.root_path / self.repo_id

            joint_names = [
                'right_joint1', 'right_joint2', 'right_joint3',
                'right_joint4', 'right_joint5', 'right_joint6',
                'right_rh_r1_joint'
            ]

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
                'names': joint_names,
                'shape': (7,)
            }
            features[f'action'] = {
                'dtype': 'float32',
                'names': joint_names,
                'shape': (7,)
            }

            self.dataset = LeRobotDataset.create(
                repo_id=self.repo_id,
                root=dataset_path,
                features=features,
                use_videos=True,
                fps=fps,
                robot_type="omy_f3m",
                image_writer_processes=2,
                image_writer_threads=4,
            )

            print(f"[Info ] 데이터셋이 성공적으로 초기화되었습니다.")
            return "데이터셋 초기화 성공"

    def _recording_loop(self):
        """생산자: 정밀한 타이밍에 맞춰 데이터만 수집하여 큐에 삽입"""
        next_time = time.time()

        while self.running:
            if self.is_recording and self.dataset is not None:
                frame_interval = 1.0 / self.fps

                if self.max_record_time > 0:
                    if time.time() - self.start_time >= self.max_record_time:
                        self.stop_recording()
                        continue

                # 1. 데이터 수집
                raw_data = self.subscriber_hub.get_latest_data()

                # 2. 큐에 삽입 (에피소드 구분을 위해 현재 에피소드 인덱스 포함 가능)
                self.data_queue.put(raw_data)

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

    def _consumer_loop(self):
        """소비자: 큐에서 데이터를 꺼내 무거운 작업 수행"""
        while self.running:
            try:
                raw_data = self.data_queue.get(timeout=0.1)

                if self.dataset is not None:
                    kinect_msg, wrist_msg, follow_msg, leader_msg = raw_data
                    kinect_img = decode_image(kinect_msg)
                    wrist_img = decode_image(wrist_msg)
                    follower_joint_data = np.array(follow_msg.position, dtype=np.float32)
                    leader_joint_data = np.array(leader_msg.position, dtype=np.float32)

                    if kinect_img is not None and wrist_img is not None:
                        with self.lock:
                            if self.dataset is not None:
                                self.dataset.add_frame({
                                    f'observation.images.cam_top': kinect_img,
                                    f'observation.images.cam_wrist': wrist_img,
                                    f'observation.state': follower_joint_data,
                                    f'action': leader_joint_data,
                                    f'task': self.task_name
                                })

                self.data_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Error] 소비자 루프 오류: {e}")

    def start_recording(self, max_time=0):
        if self.dataset is None:
            return "오류: 데이터셋이 초기화되지 않았습니다."

        # 즉시 다음 녹화가 가능하도록 큐를 비우지 않고 상태만 변경
        # (이전 녹화 데이터는 소비자 쓰레드가 백그라운드에서 계속 처리 중)
        self.max_record_time = max_time
        self.start_time = time.time()
        self.is_recording = True

        msg = "시스템: 녹화를 시작합니다."
        print(msg)
        return msg

    def stop_recording(self):
        if not self.is_recording:
            return "시스템: 현재 녹화 중이 아닙니다."

        self.is_recording = False

        # 즉시 저장을 호출하는 대신, 별도 쓰레드에서 큐가 비워지면 저장하도록 함
        threading.Thread(target=self._wait_and_save, daemon=True).start()

        print("시스템: 녹화가 중단되었습니다. 백그라운드에서 저장을 진행합니다.")
        return "시스템: 녹화 중단 (백그라운드 저장 중)"

    def _wait_and_save(self):
        """백그라운드에서 큐가 비워질 때까지 기다린 후 저장"""
        # 현재 시점의 큐 작업이 끝날 때까지 대기
        self.data_queue.join()

        with self.lock:
            if self.dataset is not None:
                self.dataset.save_episode()
                self.dataset.finalize()
                print("시스템: 에피소드 저장 완료")

    def close(self):
        self.running = False
        self.record_thread.join()
        self.consumer_thread.join()
        print("시스템: 모든 쓰레드가 종료되었습니다.")

class GradioVisualizer:
    def __init__(self, subscriber_hub: SubscriberHub):
        self.subscriber_hub = subscriber_hub
        self.update_interval = 1/30
        self.dataset_manager = Dataset_manager(self.subscriber_hub)

    def ui_timer_callback(self):
        (k_msg, w_msg, f_joint, l_joint) = self.subscriber_hub.get_latest_data()
        k_img = decode_image(k_msg)
        w_img = decode_image(w_msg)

        desired_names = ['right_joint1', 'right_joint2', 'right_joint3', 'right_joint4', 'right_joint5', 'right_joint6', 'right_rh_r1_joint']

        follower_text = "N/A"
        if f_joint is not None:
            f_vals = [np.rad2deg(f_joint.position[f_joint.name.index(n)]) if n in f_joint.name else np.nan for n in desired_names]
            follower_text = f"J1: {f_vals[0]:.1f} J2: {f_vals[1]:.1f} J3: {f_vals[2]:.1f} J4: {f_vals[3]:.1f} J5: {f_vals[4]:.1f} J6: {f_vals[5]:.1f} G: {f_vals[6]:.1f}"

        leader_text = "N/A"
        if l_joint is not None:
            l_vals = [np.rad2deg(l_joint.position[l_joint.name.index(n)]) if n in l_joint.name else np.nan for n in desired_names]
            leader_text = f"J1: {l_vals[0]:.1f} J2: {l_vals[1]:.1f} J3: {l_vals[2]:.1f} J4: {l_vals[3]:.1f} J5: {l_vals[4]:.1f} J6: {l_vals[5]:.1f} G: {l_vals[6]:.1f}"

        # 상태 및 프로세스 표시
        q_size = self.dataset_manager.data_queue.qsize()
        if self.dataset_manager.is_recording:
            elapsed = time.time() - self.dataset_manager.start_time
            status = f"🔴 녹화 중... {elapsed:.1f}s | 대기 큐: {q_size}"
        else:
            if q_size > 0:
                status = f"⏳ 백그라운드 저장 중... (남은 작업: {q_size})"
            else:
                status = "✅ 대기 중 (모든 작업 완료)"

        return k_img, w_img, follower_text, leader_text, status

    def create_interface(self):
        default_root_dir = os.path.join(os.getcwd(), "dataset")
        with gr.Blocks(title="Robot Data Collector") as demo:
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
            status_output = gr.Textbox(label="Status")

            with gr.Row():
                record_btn = gr.Button("Record", variant="primary")
                stop_btn = gr.Button("Stop", variant="stop")

            init_btn.click(self.dataset_manager.init_dataset, [repo_id_input, root_dir_input, task_name_input, fps_input], status_output)
            record_btn.click(self.dataset_manager.start_recording, [max_time_input], status_output)
            stop_btn.click(self.dataset_manager.stop_recording, outputs=status_output)

            timer = gr.Timer(value=self.update_interval)
            timer.tick(self.ui_timer_callback, outputs=[kinect_image, wrist_image, follower_joint_output, leader_joint_output, status_output])
        return demo

    def launch(self):
        self.create_interface().launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    import rclpy
    rclpy.init()
    hub = SubscriberHub()
    threading.Thread(target=lambda: rclpy.spin(hub), daemon=True).start()
    GradioVisualizer(hub).launch()
