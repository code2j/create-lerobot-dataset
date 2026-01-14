import threading
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, JointState
import cv2
import numpy as np
import torch
import gradio as gr
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import time
import os
import queue

# 허브 접속 차단 (로컬 우선)
os.environ["HF_HUB_OFFLINE"] = "1"

KINECT_TOPIC        = "/kinect/color/compressed"
KINECT_DICT         = "kinect_camera"
RIGHT_WRIST_TOPIC   = "/right/camera/cam_wrist/color/image_rect_raw/compressed"
RIGHT_WRIST_DICT    = "right_wrist_camera"
RIGHT_STATE_TOPIC   = "/right/joint_states"

class GradioLeRobotVideoRecorder(Node):
    def __init__(self):
        super().__init__('gradio_lerobot_video_recorder')
        self.lock = threading.Lock()

        self.dataset = None
        self.repo_id = ""
        self.root_path = None
        self.task_name = "default_task"

        self.is_recording = False
        self.is_saving = False
        self.frame_count = 0
        self.current_frame_for_ui = None
        self.current_frame_secondary_for_ui = None

        self.fps = 30
        self.frame_duration = 1.0 / self.fps
        self.max_time = 10.0
        self.start_time = 0.0
        self.elapsed_time = 0.0
        self.next_frame_time = 0.0
        self.status_msg = "대기 중"

        # 데이터 큐 및 처리 스레드
        self.data_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._data_worker, daemon=True)
        self.worker_thread.start()

        # 해상도 저장용
        self.res_main = "0x0"
        self.res_sub = "0x0"

        # 최신 압축 데이터 및 상태 저장
        self.latest_compressed_kinect = None
        self.latest_compressed_wrist = None
        self.latest_state = torch.zeros(7)
        self.latest_action = torch.zeros(7)

        self.joint_names = [
            'right_joint1', 'right_joint2', 'right_joint3',
            'right_joint4', 'right_joint5', 'right_joint6',
            'right_rh_r1_joint'
        ]

        # 1번 카메라 (Kinect)
        self.subscription = self.create_subscription(
            CompressedImage, KINECT_TOPIC, self._kinect_callback, 10)

        # 2번 카메라 (Right Wrist)
        self.subscription_wrist = self.create_subscription(
            CompressedImage, RIGHT_WRIST_TOPIC, self._right_wrist_callback, 10)

        # 오른쪽 로봇 조인트 상태
        self.joint_subscription = self.create_subscription(
            JointState, RIGHT_STATE_TOPIC, self._joint_state_callback, 10)

        # 타이머 주기 (200Hz)
        self.create_timer(0.005, self._recording_loop)

    def _data_worker(self):
        """백그라운드에서 두 카메라의 압축 해제 및 add_frame 수행"""
        while True:
            try:
                item = self.data_queue.get(timeout=1)
                if item is None: continue

                # 1. Kinect 압축 해제 및 변환
                np_kinect = np.frombuffer(item['kinect_img'], np.uint8)
                img_kinect = cv2.imdecode(np_kinect, cv2.IMREAD_COLOR)

                # 2. Wrist 압축 해제 및 변환
                np_wrist = np.frombuffer(item['wrist_img'], np.uint8)
                img_wrist = cv2.imdecode(np_wrist, cv2.IMREAD_COLOR)

                if img_kinect is not None and img_wrist is not None and self.dataset is not None:
                    rgb_kinect = cv2.cvtColor(img_kinect, cv2.COLOR_BGR2RGB)
                    rgb_wrist = cv2.cvtColor(img_wrist, cv2.COLOR_BGR2RGB)

                    # LeRobotDataset 형식에 맞춰 텐서 변환 (C, H, W)
                    kinect_tensor = torch.from_numpy(rgb_kinect).permute(2, 0, 1)
                    wrist_tensor = torch.from_numpy(rgb_wrist).permute(2, 0, 1)

                    self.dataset.add_frame({
                        "observation.images.cam_top": kinect_tensor,
                        "observation.images.right_cam_wrist": wrist_tensor,
                        "observation.state": item['state'],
                        "action": item['action'],
                        "task": item['task']
                    })

                    with self.lock:
                        self.frame_count += 1

                self.data_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Worker error: {e}")

    def get_ep_count(self):
        return self.dataset.num_episodes if self.dataset is not None else 0

    def init_dataset(self, repo_id, root_dir, task_name):
        with self.lock:
            try:
                self.repo_id = repo_id
                self.root_path = Path(root_dir).absolute()
                self.task_name = task_name
                dataset_path = self.root_path / self.repo_id
                info_json = dataset_path / "meta" / "info.json"

                if info_json.exists():
                    self.dataset = LeRobotDataset(repo_id=self.repo_id, root=dataset_path)
                    self.status_msg = "📂 기존 데이터셋을 로드했습니다."
                else:
                    self.dataset = LeRobotDataset.create(
                        repo_id=self.repo_id,
                        root=dataset_path,
                        fps=self.fps,
                        robot_type= "omy_f3m",
                        features={
                            "timestamp": {"dtype": "float32", "shape": (1,), "names": None, "fps": self.fps},
                            "frame_index": {"dtype": "int64", "shape": (1,), "names": None, "fps": self.fps},
                            "episode_index": {"dtype": "int64", "shape": (1,), "names": None, "fps": self.fps},
                            "index": {"dtype": "int64", "shape": (1,), "names": None, "fps": self.fps},
                            "task_index": {"dtype": "int64", "shape": (1,), "names": None, "fps": self.fps},
                            "observation.images.cam_top": {
                                "dtype": "video",
                                "shape": (3, 720, 1280),
                                "names": ["channels", "height", "width"],
                                "info": {
                                    "video.height": 720,
                                    "video.width": 1280,
                                    "video.channels": 3,
                                    "video.codec": "libx264",
                                    "video.pix_fmt": "yuv420p"
                                }
                            },
                            "observation.images.right_cam_wrist": {
                                "dtype": "video",
                                "shape": (3, 480, 848),
                                "names": ["channels", "height", "width"],
                                "info": {
                                    "video.height": 480,
                                    "video.width": 848,
                                    "video.channels": 3,
                                    "video.codec": "libx264",
                                    "video.pix_fmt": "yuv420p"
                                }
                            },
                            "observation.state": {
                                "dtype": "float32",
                                "shape": (7,),
                                "names": self.joint_names
                            },
                            "action": {
                                "dtype": "float32",
                                "shape": (7,),
                                "names": self.joint_names
                            },
                        },
                        use_videos=True,
                    )
                    self.status_msg = f"📂 새로운 데이터셋을 생성했습니다."
                return self.status_msg, self.get_ep_count()

            except Exception as e:
                self.status_msg = f"❌ 오류: {str(e)}"
                return self.status_msg, 0

    def _joint_state_callback(self, msg):
        current_joints = []
        for name in self.joint_names:
            if name in msg.name:
                idx = msg.name.index(name)
                current_joints.append(msg.position[idx])

        if len(current_joints) == 7:
            with self.lock:
                joint_tensor = torch.tensor(current_joints, dtype=torch.float32)
                self.latest_state = joint_tensor
                self.latest_action = joint_tensor

    def _kinect_callback(self, msg):
        with self.lock:
            self.latest_compressed_kinect = msg.data
        np_arr = np.frombuffer(msg.data, np.uint8)
        img_raw = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img_raw is not None:
            self.res_main = f"{img_raw.shape[1]}x{img_raw.shape[0]}"
            with self.lock:
                self.current_frame_for_ui = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

    def _right_wrist_callback(self, msg):
        with self.lock:
            self.latest_compressed_wrist = msg.data
        np_arr = np.frombuffer(msg.data, np.uint8)
        img_raw = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img_raw is not None:
            self.res_sub = f"{img_raw.shape[1]}x{img_raw.shape[0]}"
            with self.lock:
                self.current_frame_secondary_for_ui = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

    def _recording_loop(self):
        if not self.is_recording or self.dataset is None:
            return

        now = time.time()
        self.elapsed_time = now - self.start_time

        if now >= self.next_frame_time:
            with self.lock:
                if self.latest_compressed_kinect is None or self.latest_compressed_wrist is None:
                    return

                if self.elapsed_time >= self.max_time:
                    self.is_recording = False
                    threading.Thread(target=self._wait_and_save, daemon=True).start()
                    return

                self.data_queue.put({
                    "kinect_img": self.latest_compressed_kinect,
                    "wrist_img": self.latest_compressed_wrist,
                    "state": self.latest_state.clone(),
                    "action": self.latest_action.clone(),
                    "task": self.task_name
                })

                self.next_frame_time += self.frame_duration

    def _wait_and_save(self):
        self.status_msg = "⏳ 데이터 처리 중..."
        self.data_queue.join()
        self._save_episode_internal()

    def _save_episode_internal(self):
        with self.lock:
            self.is_saving = True
            self.status_msg = "💾 저장 중... 잠시만 기다려주세요"

            try:
                self.dataset.save_episode()
                gr.Info(f"✅ 에피소드 {self.dataset.num_episodes - 1} 저장 완료! (총 {self.frame_count} 프레임)")
                self.status_msg = "✅ 저장 완료"
            except Exception as e:
                print(f"Save error: {e}")
                self.status_msg = "❌ 저장 오류"
            finally:
                self.is_saving = False

    def start_rec(self):
        if self.dataset is None:
            self.status_msg = "⚠️ 데이터셋 초기화/불러오기를 먼저 해주세요"
            return self.status_msg, 0, 0
        if self.is_recording:
            return self.status_msg, self.get_ep_count(), self.frame_count

        with self.lock:
            self._clear_buffer_internal()
            while not self.data_queue.empty():
                try: self.data_queue.get_nowait(); self.data_queue.task_done()
                except: break

            self.is_recording = True
            self.frame_count = 0
            self.start_time = time.time()
            self.next_frame_time = self.start_time
            self.elapsed_time = 0.0
            self.status_msg = "🔴 녹화 중..."
        return self.status_msg, self.get_ep_count(), 0

    def _clear_buffer_internal(self):
        if self.dataset is not None:
            if hasattr(self.dataset, 'clear_episode_buffer'):
                self.dataset.clear_episode_buffer()
            else:
                self.dataset._frames = []

    def update_ui_components(self):
        progress_val = 0
        bar_label = f"준비 완료: 최대 {self.max_time:.1f}s"
        if self.is_recording:
            progress_val = min(100, (self.elapsed_time / self.max_time) * 100)
            bar_label = f"⌛ 녹화 중: {self.elapsed_time:.1f}s / {self.max_time:.1f}s"
        elif self.is_saving:
            progress_val, bar_label = 100, "💾 저장 중..."

        with self.lock:
            joints_deg = [np.rad2deg(val.item()) for val in self.latest_state]
            joint_str = (f"J1:{joints_deg[0]:>6.1f}° | J2:{joints_deg[1]:>6.1f}° | J3:{joints_deg[2]:>6.1f}°\n"
                         f"J4:{joints_deg[3]:>6.1f}° | J5:{joints_deg[4]:>6.1f}° | J6:{joints_deg[5]:>6.1f}°\n"
                         f"Gripper: {joints_deg[6]:.1f}°")
            main_label = f"Kinect camera | {self.res_main}"
            sub_label = f"Right Wrist Camera | {self.res_sub}"
            current_frames = self.frame_count

        return (
            gr.update(value=self.current_frame_for_ui, label=main_label),
            gr.update(value=self.current_frame_secondary_for_ui, label=sub_label),
            gr.update(value=progress_val, label=bar_label),
            self.status_msg,
            self.get_ep_count(),
            joint_str,
            current_frames
        )

def launch_ui(server_name:str, port:int, dt:float):
    if not rclpy.ok(): rclpy.init()
    recorder = GradioLeRobotVideoRecorder()
    threading.Thread(target=lambda: rclpy.spin(recorder), daemon=True).start()

    with gr.Blocks(title="LeRobot Collector - Multi Cam") as demo:
        gr.Markdown("# 🤖 LeRobot 데이터 수집기 (Kinect + Wrist)")

        with gr.Accordion("⚙️ 설정", open=True):
            with gr.Row():
                repo_id_input = gr.Textbox(label="Repo ID", value="uon/HERE_DATASET_NAME")
                root_path_input = gr.Textbox(label="Root Path", value="outputs")
                task_name_input = gr.Textbox(label="Task Name", value="HERE_TASK_NAME")
            with gr.Row():
                max_time_input = gr.Number(label="최대 시간(초)", value=10.0)
            init_btn = gr.Button("🔄 데이터셋 초기화/불러오기")

        with gr.Row():
            with gr.Column(scale=2):
                with gr.Row():
                    image_output = gr.Image(label="Kinect camera")
                    image_secondary_output = gr.Image(label="Right wrist camera")
                joint_info_display = gr.Textbox(label="현재 로봇 조인트 각도 (Degree)", lines=3, interactive=False)
                progress_bar = gr.Slider(label="준비 완료", minimum=0, maximum=100, value=0, interactive=False)

            with gr.Column(scale=1):
                ep_count_display = gr.Label(value="0", label="현재 에피소드 수")
                frame_count_display = gr.Label(value="0", label="현재 녹화된 프레임 수 (add_frame)")
                status_text = gr.Label(value="대기 중", label="현재 상태")
                start_btn = gr.Button("🔴 녹화 시작", variant="primary")

        gr.Timer(dt).tick(
            recorder.update_ui_components,
            outputs=[image_output, image_secondary_output, progress_bar, status_text, ep_count_display, joint_info_display, frame_count_display]
        )

        init_btn.click(
            recorder.init_dataset,
            inputs=[repo_id_input, root_path_input, task_name_input],
            outputs=[status_text, ep_count_display]
        )

        start_btn.click(recorder.start_rec, outputs=[status_text, ep_count_display, frame_count_display])

    demo.launch(server_name=server_name, server_port=port)
    rclpy.shutdown()

if __name__ == "__main__":
    launch_ui(server_name="127.0.0.1", port=7890, dt=1/60)
