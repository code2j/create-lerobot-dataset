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
import shutil
import time
import os

# 허브 접속 차단 (로컬 우선)
os.environ["HF_HUB_OFFLINE"] = "1"

KINECT_TOPIC        = "/kinect/color/compressed"
KINECT_DICT         = "kinect_camera"

RIGHT_WRIST_TOPIC   = "/right/camera/cam_wrist/color/image_rect_raw/compressed"
RIGHT_STATE_TOPIC   = "/right/joint_states"
RIGHT_WRIST_DICT    = "right_wrist_camera"


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

        self.max_time = 10.0
        self.start_time = 0.0
        self.elapsed_time = 0.0
        self.status_msg = "대기 중"

        # 해상도 저장용
        self.res_main = "0x0" # 키넥트 해상도
        self.res_sub = "0x0"  # 오른쪽 카메라 해상도

        # 상태 및 액션 (6 joints + 1 gripper = 7)
        self.latest_data = {
            KINECT_DICT: None,
            RIGHT_WRIST_DICT: None,
            "state": torch.zeros(7),
            "action": torch.zeros(7)
        }

        self.joint_names = [
            'right_joint1', 'right_joint2', 'right_joint3',
            'right_joint4', 'right_joint5', 'right_joint6',
            'right_rh_r1_joint'
        ]

        # 1번 카메라 (Kinect)
        self.subscription = self.create_subscription(
            CompressedImage, KINECT_TOPIC, self._kinect_callback, 10)

        # 2번 카메라 (Right Wrist)
        self.subscription_secondary = self.create_subscription(
            CompressedImage, RIGHT_WRIST_TOPIC, self._right_wrist_callback, 10)

        # 오른쪽 로봇 조인트 상태 (7 joints)
        self.joint_subscription = self.create_subscription(
            JointState, RIGHT_STATE_TOPIC, self._joint_state_callback, 10)

        self.create_timer(1.0 / 30.0, self._recording_loop)

    def get_ep_count(self):
        return self.dataset.num_episodes if self.dataset is not None else 0

    def init_dataset(self, repo_id, root_dir, task_name): # task_name 인자 추가
        with self.lock:
            try:
                self.repo_id = repo_id
                self.root_path = Path(root_dir).absolute()
                self.task_name = task_name # 입력받은 테스크 이름 저장
                dataset_path = self.root_path / self.repo_id
                info_json = dataset_path / "meta" / "info.json"

                print("root path: ", self.root_path)
                print("dataset_path: ", dataset_path)

                if info_json.exists():
                    self.dataset = LeRobotDataset(repo_id=self.repo_id, root=dataset_path)
                    self.status_msg = "📂 기존 데이터셋을 로드했습니다."
                else:
                    # TODO: Features 설정
                    self.dataset = LeRobotDataset.create(
                        repo_id=self.repo_id,
                        root=dataset_path,
                        fps=30,
                        robot_type= "omy_f3m",
                        features={
                            "timestamp": {
                                "dtype": "float32",
                                "shape": (1),
                                "names": None,
                                "fps": 30
                            },
                            "frame_index": {
                                "dtype": "int64",
                                "shape": (1),
                                "names": None,
                                "fps": 30
                            },
                            "episode_index": {
                                "dtype": "int64",
                                "shape": (1),
                                "names": None,
                                "fps": 30
                            },
                            "index": {
                                "dtype": "int64",
                                "shape": (1),
                                "names": None,
                                "fps": 30
                            },
                            "task_index": {
                                "dtype": "int64",
                                "shape": (1),
                                "names": None,
                                "fps": 30
                            },

                            "observation.images.cam_top": {
                                "dtype": "video",
                                "shape": (1280, 720, 3),
                                "names": ["width", "height", "channels"],
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
                                "shape": (848, 480, 3),
                                "names": ["width", "height", "channels"],
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
                                "names": [
                                    "right_joint1",
                                    "right_joint2",
                                    "right_joint3",
                                    "right_joint4",
                                    "right_joint5",
                                    "right_joint6",
                                    "right_rh_r1_joint"
                                ]
                            },
                            "action": {
                                "dtype": "float32",
                                "shape": (7,),
                                "names": [
                                    "right_joint1",
                                    "right_joint2",
                                    "right_joint3",
                                    "right_joint4",
                                    "right_joint5",
                                    "right_joint6",
                                    "right_rh_r1_joint"
                                ]
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
        """조인트 데이터 수신 콜백"""
        current_joints = []

        for name in self.joint_names:
            if name in msg.name:
                idx = msg.name.index(name)
                current_joints.append(msg.position[idx])

        if len(current_joints) == 7:
            with self.lock:
                joint_tensor = torch.tensor(current_joints, dtype=torch.float32)
                self.latest_data["state"] = joint_tensor
                self.latest_data["action"] = joint_tensor

    def _kinect_callback(self, msg):
        """키넥트 데이터 수신 콜백"""
        np_arr = np.frombuffer(msg.data, np.uint8)
        img_raw = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img_raw is not None:
            self.res_main = f"{img_raw.shape[1]}x{img_raw.shape[0]}"
            with self.lock:
                rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
                self.latest_data[KINECT_DICT] = rgb
                self.current_frame_for_ui = rgb

    def _right_wrist_callback(self, msg):
        """오른쪽 손목 카메라 데이터 수신 콜백"""
        np_arr = np.frombuffer(msg.data, np.uint8)
        img_raw = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img_raw is not None:
            self.res_sub = f"{img_raw.shape[1]}x{img_raw.shape[0]}"
            with self.lock:
                rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
                self.latest_data[RIGHT_WRIST_DICT] = rgb
                self.current_frame_secondary_for_ui = rgb

    def _recording_loop(self):
        """데이터 녹화 루프"""
        if not self.is_recording or self.dataset is None:
            return
        with self.lock:
            if self.latest_data[KINECT_DICT] is None:
                self.status_msg = "⚠️ 키넥트 데이터 없음"
                return

            if self.latest_data[RIGHT_WRIST_DICT] is None:
                self.status_msg = "⚠️ 오른쪽 손목 카메라 데이터 없음"
                return


            self.elapsed_time = time.time() - self.start_time
            if self.elapsed_time >= self.max_time:
                self.is_recording = False
                threading.Thread(target=self._save_episode_internal, daemon=True).start()
                return

            img_tensor = torch.from_numpy(self.latest_data[KINECT_DICT]).permute(1, 0, 2)
            img_secondary_tensor = torch.from_numpy(self.latest_data[RIGHT_WRIST_DICT]).permute(1, 0, 2)

            # TODO: 데이터 추가
            self.dataset.add_frame({
                "observation.images.cam_top": img_tensor,
                "observation.images.right_cam_wrist": img_secondary_tensor,
                "observation.state": self.latest_data["state"],
                "action": self.latest_data["action"],
                "task": self.task_name
            })
            self.frame_count += 1

    def _save_episode_internal(self):
        with self.lock:
            if self.frame_count == 0:
                self.status_msg = "⚠️ 데이터 없음"
                self.is_saving = False
                return

            self.is_saving = True
            self.is_recording = False
            self.status_msg = "💾 저장 중... 잠시만 기다려주세요"

            try:
                self.dataset.save_episode()
                gr.Info(f"✅ 에피소드 {self.dataset.num_episodes - 1} 저장 완료!")
                self.status_msg = "✅ 저장 완료"
            except Exception as e:
                self.status_msg = "❌ 저장 오류"
            finally:
                self.frame_count = 0
                self.elapsed_time = 0.0
                self.is_saving = False

    def start_rec(self):
        """녹화 시작 버튼"""
        if self.dataset is None:
            self.status_msg = "⚠️ 데이터셋 초기화/불러오기를 먼저 해주세요"
            return self.status_msg, 0, ""
        if self.is_recording:
            self.status_msg = "⚠️ 이미 녹화 중입니다"
            return self.status_msg
        with self.lock:
            self._clear_buffer_internal()
            self.is_recording = True
            self.frame_count = 0
            self.start_time = time.time()
            self.elapsed_time = 0.0
            self.status_msg = "🔴 녹화 중..."
        return self.status_msg, self.get_ep_count(), ""

    def retry_rec(self):
        """재시작 버튼"""
        if self.dataset is None: return self.status_msg, 0, ""
        with self.lock:
            self._clear_buffer_internal()
            self.is_recording = True
            self.frame_count = 0
            self.start_time = time.time()
            self.status_msg = "🔄 재시도 중"
        return self.status_msg, self.get_ep_count(), ""

    def _clear_buffer_internal(self):
        """녹화중인 데이터 버퍼 초기화"""
        if self.dataset is not None:
            if hasattr(self.dataset, 'clear_episode_buffer'):
                self.dataset.clear_episode_buffer()
            else:
                self.dataset._frames = []

    def next_episode(self):
        """에피소드 완료 버튼"""
        if self.dataset is None: return self.status_msg, 0, ""
        if self.is_recording:
            self.is_recording = False
            threading.Thread(target=self._save_episode_internal, daemon=True).start()
        return self.status_msg, self.get_ep_count(), ""

    def finalize_dataset(self):
        """데이터셋 수집 종료 버튼"""
        if self.dataset is None: return "⚠️ 미설정", 0, ""
        with self.lock:
            self.dataset.finalize()
            self.status_msg = "🏁 수집 종료"
            return self.status_msg, self.get_ep_count(), ""

    def update_ui_components(self):
        """UI 컴포넌트 업데이트"""
        progress_val = 0
        bar_label = f"준비 완료: 최대 {self.max_time:.1f}s"
        if self.is_recording:
            progress_val = min(100, (self.elapsed_time / self.max_time) * 100)
            bar_label = f"⌛ 녹화 중: {self.elapsed_time:.1f}s / {self.max_time:.1f}s"
        elif self.is_saving:
            progress_val, bar_label = 100, "💾 저장 중..."

        with self.lock:
            joints_deg = [np.rad2deg(val.item()) for val in self.latest_data["state"]]
            joint_str = (f"J1:{joints_deg[0]:>6.1f}° | J2:{joints_deg[1]:>6.1f}° | J3:{joints_deg[2]:>6.1f}°\n"
                         f"J4:{joints_deg[3]:>6.1f}° | J5:{joints_deg[4]:>6.1f}° | J6:{joints_deg[5]:>6.1f}°\n"
                         f"Gripper: {joints_deg[6]:.1f}°")

            main_label = f"Kinect camera | {self.res_main}"
            sub_label = f"Right Wrist Camera | {self.res_sub}"

        return (
            gr.update(value=self.current_frame_for_ui, label=main_label),
            gr.update(value=self.current_frame_secondary_for_ui, label=sub_label),
            gr.update(value=progress_val, label=bar_label),
            self.status_msg,
            self.get_ep_count(),
            joint_str
        )

# --- UI 함수 ---
def launch_ui(server_name:str, port:int, dt:float):
    if not rclpy.ok(): rclpy.init()
    recorder = GradioLeRobotVideoRecorder()
    threading.Thread(target=lambda: rclpy.spin(recorder), daemon=True).start()

    with gr.Blocks(title="LeRobot Collector v3.4") as demo:
        gr.Markdown("# 🤖 LeRobot v3.4 멀티캠 수집기 (Task 설정 추가)")

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
                status_text = gr.Label(value="대기 중", label="현재 상태")
                start_btn = gr.Button("🔴 녹화 시작", variant="primary")
                next_btn = gr.Button("💾 에피소드 완료", variant="secondary")
                finish_btn = gr.Button("🏁 데이터 수집 종료", variant="stop")

        gr.Timer(dt).tick(
            recorder.update_ui_components,
            outputs=[image_output, image_secondary_output, progress_bar, status_text, ep_count_display, joint_info_display]
        )

        # 초기화 버튼 클릭 시 task_name_input도 함께 전달
        init_btn.click(
            recorder.init_dataset,
            inputs=[repo_id_input, root_path_input, task_name_input],
            outputs=[status_text, ep_count_display]
        )

        start_btn.click(recorder.start_rec, outputs=[status_text, ep_count_display])
        next_btn.click(lambda: (recorder.next_episode()[0], recorder.get_ep_count()), outputs=[status_text, ep_count_display])
        finish_btn.click(recorder.finalize_dataset, outputs=[status_text, ep_count_display])

    demo.launch(server_name=server_name, server_port=port)
    rclpy.shutdown()

if __name__ == "__main__":
    launch_ui(
        server_name="127.0.0.1",
        port=7890,
        dt=1/60
    )