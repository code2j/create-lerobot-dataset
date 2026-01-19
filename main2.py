import gradio as gr
import cv2
import numpy as np
from pathlib import Path
import threading
import time
import queue
import os
import shutil
import subprocess
import signal

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

        self.joint_names = [
            'right_joint1', 'right_joint2', 'right_joint3',
            'right_joint4', 'right_joint5', 'right_joint6',
            'right_rh_r1_joint'
        ]

        # 1. 생산자 쓰레드: 데이터를 수집하여 큐에 넣음
        self.record_thread = threading.Thread(target=self._recording_loop, daemon=True)
        self.record_thread.start()

        # 2. 소비자 쓰레드: 큐에서 데이터를 꺼내 디코딩 및 저장
        self.consumer_thread = threading.Thread(target=self._consumer_loop, daemon=True)
        self.consumer_thread.start()

        print("[Info ] 녹화 및 소비자 쓰레드가 시작")

    def init_dataset(self, repo_id, root_dir, task_name, fps) -> str:
        """데이터셋 초기화 및 생성"""
        with self.lock:
            self.repo_id = repo_id
            self.root_path = Path(root_dir).absolute()
            self.task_name = task_name
            self.fps = fps

            dataset_path = self.root_path / self.repo_id

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
                raw_data = self.subscriber_hub.get_latest_msg()

                # 2. 큐에 삽입
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
        """큐에서 데이터를 꺼내 작업 수행(디코딩 및 변환)"""
        while self.running:
            try:
                raw_data = self.data_queue.get(timeout=0.1)

                if self.dataset is not None:
                    kinect_msg, wrist_msg, follow_msg, leader_msg = raw_data

                    # 1. 이미지 처리 (디코딩)
                    kinect_img = decode_image(kinect_msg)
                    wrist_img = decode_image(wrist_msg)

                    # 2. 팔로워(State) 데이터 정렬
                    follow_map = dict(zip(follow_msg.name, follow_msg.position))
                    follower_joint_data = np.array([follow_map[name] for name in self.joint_names], dtype=np.float32)

                    # 3. 리더(Action) 데이터 정렬
                    leader_map = dict(zip(leader_msg.name, leader_msg.position))
                    leader_joint_data = np.array([leader_map[name] for name in self.joint_names], dtype=np.float32)

                    # 4. 데이터셋 추가
                    if kinect_img is not None and wrist_img is not None:
                        with self.lock:
                            self.dataset.add_frame({
                                'observation.images.cam_top': kinect_img,
                                'observation.images.cam_wrist': wrist_img,
                                'observation.state': follower_joint_data,
                                'action': leader_joint_data,
                                'task': self.task_name
                            })

                self.data_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Error] 소비자 루프 오류: {e}")


    def toggle_recording(self, max_time=0):
        """스페이스바 단축키를 위한 토글 기능: 상태에 따라 분기"""
        if self.is_recording:
            return self.stop_recording()
        else:
            return self.start_recording(max_time)

    def start_recording(self, max_time=0):
        if self.dataset is None:
            return "오류: 데이터셋이 초기화되지 않았습니다."

        if self.is_recording:
            return "시스템: 이미 녹화 중입니다."

        self.max_record_time = max_time
        self.start_time = time.time()
        self.is_recording = True

        msg = "시스템: 녹화를 시작합니다. (Space 키로 중단 가능)"
        print(msg)
        return msg

    def stop_recording(self):
        if not self.is_recording:
            return "시스템: 현재 녹화 중이 아닙니다."

        self.is_recording = False

        # 백그라운드에서 큐가 비워지면 저장하도록 함
        threading.Thread(target=self._wait_and_save, daemon=True).start()

        msg = "시스템: 녹화 중단 (백그라운드 저장 중)"
        print(msg)
        return msg

    def _wait_and_save(self):
        """백그라운드에서 큐가 비워질 때까지 기다린 후 저장"""
        print("시스템: 남은 데이터를 처리 중입니다...")
        self.data_queue.join()

        with self.lock:
            if self.dataset is not None:
                try:
                    self.dataset.save_episode()
                    print("시스템: 에피소드 저장 완료")
                except Exception as e:
                    print(f"시스템: 에피소드 저장 중 오류: {e}")

    def finalize_dataset(self):
        """데이터 수집 완료 및 데이터셋 최종화"""
        if self.dataset is None:
            return "오류: 데이터셋이 초기화되지 않았습니다."

        if self.is_recording:
            return "오류: 녹화 중에는 데이터셋을 완료할 수 없습니다."

        if not self.data_queue.empty():
            return "시스템: 아직 처리 중인 데이터가 있습니다. 잠시 후 다시 시도해주세요."

        with self.lock:
            try:
                self.dataset.finalize()
                msg = "시스템: 데이터 수집 완료 및 데이터셋 최종화 성공"
                print(msg)
                return msg
            except Exception as e:
                msg = f"시스템: 데이터셋 최종화 중 오류 발생: {e}"
                print(msg)
                return msg

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
        (k_msg, w_msg, f_joint, l_joint) = self.subscriber_hub.get_latest_msg()
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

        # 스페이스바 감지를 위한 JavaScript
        js_code = """
        function() {
            document.addEventListener('keydown', function(e) {
                if (e.code === 'Space') {
                    const active = document.activeElement;
                    if (active.tagName === 'INPUT' || active.tagName === 'TEXTAREA' || active.isContentEditable) {
                        return;
                    }
                    e.preventDefault();
                    // 숨겨진 토글 버튼을 클릭하게 함
                    const btn = document.getElementById('toggle_btn');
                    if (btn) btn.click();
                }
            });
        }
        """

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
            status_output = gr.Textbox(label="Status")

            with gr.Row():
                # 버튼 이름을 명확하게 수정
                record_btn = gr.Button("Record", variant="primary")
                stop_btn = gr.Button("Stop", variant="stop")
                finalize_btn = gr.Button("데이터 수집 완료 (Finalize)", variant="secondary")

            # 스페이스바 전용 숨겨진 버튼 (UI에는 보이지 않음)
            toggle_btn = gr.Button("Toggle Recording", visible=False, elem_id="toggle_btn")

            init_btn.click(self.dataset_manager.init_dataset, [repo_id_input, root_dir_input, task_name_input, fps_input], status_output)

            # 명시적 버튼 이벤트
            record_btn.click(self.dataset_manager.start_recording, [max_time_input], status_output)
            stop_btn.click(self.dataset_manager.stop_recording, outputs=status_output)
            finalize_btn.click(self.dataset_manager.finalize_dataset, outputs=status_output)

            # 스페이스바 토글 이벤트 (상태에 따라 자동 분기)
            toggle_btn.click(self.dataset_manager.toggle_recording, [max_time_input], status_output)

            timer = gr.Timer(value=self.update_interval)
            timer.tick(
                self.ui_timer_callback,
                outputs=[
                    kinect_image, wrist_image, follower_joint_output, leader_joint_output,
                    status_output
                ]
            )

            self.js_code = js_code

        return demo

    def launch(self):
        demo = self.create_interface()
        demo.launch(server_name="0.0.0.0", server_port=7860, js=self.js_code)

if __name__ == "__main__":
    import rclpy
    rclpy.init()
    hub = SubscriberHub()
    # ROS2 Spin을 별도 쓰레드에서 실행
    threading.Thread(target=lambda: rclpy.spin(hub), daemon=True).start()

    visualizer = GradioVisualizer(hub)
    try:
        visualizer.launch()
    except KeyboardInterrupt:
        visualizer.dataset_manager.close()
        rclpy.shutdown()