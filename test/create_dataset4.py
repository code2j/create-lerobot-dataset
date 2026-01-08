import threading
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
import torch
import gradio as gr
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import shutil
import time

class GradioLeRobotVideoRecorder(Node):
    def __init__(self):
        super().__init__('gradio_lerobot_video_recorder')
        self.lock = threading.Lock()

        # 1. 경로 설정
        self.repo_id = "uon/triple-cam-task-video"
        self.root_path = Path("../outputs")
        self.dataset_path = self.root_path / self.repo_id

        # 2. 데이터셋 존재 여부 확인
        if (self.dataset_path / "meta" / "info.json").exists():
            print(f"📂 [기존 데이터셋 발견] 경로: {self.dataset_path}")
            print("🔄 기존 데이터셋에 에피소드를 이어서 저장합니다.")

            self.dataset = LeRobotDataset(
                repo_id=self.repo_id,
                root=self.dataset_path
            )
        else:
            print(f"✨ [새 데이터셋 생성] 경로: {self.dataset_path}")
            # 이 단계에서만 .create()가 호출되어 폴더를 새로 만듭니다.
            self.dataset = LeRobotDataset.create(
                repo_id=self.repo_id,
                root=self.dataset_path,
                fps=30,
                features={
                    "observation.image": {
                        "dtype": "video",
                        "shape": (3, 480, 640),
                        "names": ["channels", "height", "width"],
                        "info": {"fps": 30, "video_backend": "pyav"}
                    },
                    "observation.state": {"dtype": "float32", "shape": (6,)},
                    "action": {"dtype": "float32", "shape": (6,)},
                },
                use_videos=True,
            )

        print(f"📊 현재 수집된 총 에피소드: {self.dataset.num_episodes}")

        # 2. 최신 데이터 버퍼
        self.latest_data = {
            "image": None,
            "state": torch.zeros(6),
            "action": torch.zeros(6)
        }

        self.is_recording = False
        self.frame_count = 0
        self.current_frame_for_ui = None

        # 3. ROS2 구독
        self.subscription = self.create_subscription(
            CompressedImage, '/kinect/color/compressed', self._kinect_callback, 10)

        # 4. 고정 주기 타이머 (30Hz)
        self.timer_period = 1.0 / 30.0
        self.record_timer = self.create_timer(self.timer_period, self._recording_loop)

    # 수집하 데이터가 추가 될
    def _kinect_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.resize(img, (640, 480))
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            with self.lock:
                self.latest_data["image"] = rgb_img
                self.current_frame_for_ui = rgb_img

    def _recording_loop(self):
        """데이터 저장 루프"""
        if not self.is_recording:
            return

        with self.lock:
            if self.latest_data["image"] is None:
                return

            img_tensor = torch.from_numpy(self.latest_data["image"]).permute(2, 0, 1)

            self.dataset.add_frame({
                "observation.image": img_tensor,
                "observation.state": self.latest_data["state"],
                "action": self.latest_data["action"],
                "task": "kinect_video_task"
            })
            self.frame_count += 1

    def start_rec(self):
        with self.lock:
            self.is_recording = True
            self.frame_count = 0
        return "🔴 녹화 시작됨..."

    def next_episode(self):
        with self.lock:
            if self.frame_count > 0:
                self.is_recording = False
                print(f"🎬 에피소드 저장 중 ({self.frame_count} 프레임)...")
                self.dataset.save_episode()
                msg = f"✅ 에피소드 저장 완료! ({self.frame_count} 프레임)"
                self.frame_count = 0
                return msg
            return "⚠️ 저장할 데이터가 없습니다."

    def finalize_dataset(self):
        with self.lock:
            if self.is_recording: return "⚠️ 녹화 중에는 확정할 수 없습니다."
            print("🚀 최종 데이터셋 확정 중...")
            self.dataset.finalize()
            return "🏁 데이터셋 확정 완료!"

# --- UI 함수 ---
def launch_ui():
    global recorder
    if not rclpy.ok(): rclpy.init()
    recorder = GradioLeRobotVideoRecorder()

    ros_thread = threading.Thread(target=lambda: rclpy.spin(recorder), daemon=True)
    ros_thread.start()

    with gr.Blocks(title="LeRobot Collector") as demo:
        gr.Markdown("# 🤖 LeRobot 확장형 수집기 (Fixed)")
        with gr.Row():
            with gr.Column(scale=2):
                image_output = gr.Image(label="Kinect Live Feed")
                # 타이머를 통해 UI 업데이트
                gr.Timer(0.1).tick(lambda: recorder.current_frame_for_ui, outputs=image_output)

            with gr.Column(scale=1):
                status_text = gr.Textbox(label="상태", value="대기 중")
                start_btn = gr.Button("🔴 녹화 시작", variant="primary")
                next_btn = gr.Button("💾 에피소드 완료", variant="secondary")
                finish_btn = gr.Button("🏁 최종 확정", variant="stop")

        start_btn.click(recorder.start_rec, outputs=status_text)
        next_btn.click(recorder.next_episode, outputs=status_text)
        finish_btn.click(recorder.finalize_dataset, outputs=status_text)

    demo.launch(server_name="0.0.0.0", server_port=7860)
    rclpy.shutdown()

if __name__ == "__main__":
    launch_ui()