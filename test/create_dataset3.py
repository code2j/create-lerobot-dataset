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

        # 1. 데이터셋 설정
        self.repo_id = "uon/triple-cam-task-video"
        self.root_path = Path("../outputs/dataset")
        self.dataset_path = self.root_path / self.repo_id

        if self.dataset_path.exists():
            print(f"🗑️ 기존 데이터 삭제 중: {self.dataset_path}")
            shutil.rmtree(self.dataset_path)

        self.dataset = LeRobotDataset.create(
            repo_id=self.repo_id,
            root=self.root_path,
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

        self.current_frame = None
        self.is_recording = False
        self.frame_count = 0

        # 2. ROS2 구독
        self.subscription = self.create_subscription(
            CompressedImage, '/kinect/color/compressed', self.image_callback, 10)

    def image_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.resize(img, (640, 480))
            self.current_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if self.is_recording:
                with self.lock:
                    img_tensor = torch.from_numpy(self.current_frame).permute(2, 0, 1)
                    self.dataset.add_frame({
                        "observation.image": img_tensor,
                        "observation.state": torch.zeros(6),
                        "action": torch.zeros(6),
                        "task": "kinect_video_task"
                    })
                    self.frame_count += 1

    def start_rec(self):
        with self.lock:
            self.is_recording = True
            self.frame_count = 0
        return "🔴 비디오 녹화 중..."

    def next_episode(self):
        with self.lock:
            if self.frame_count > 0:
                self.is_recording = False
                print(f"🎬 에피소드 저장 중 ({self.frame_count} 프레임)...")
                self.dataset.save_episode()
                msg = f"✅ 에피소드 저장 완료! (총 {self.frame_count} 프레임)"
                self.frame_count = 0
                return msg
            return "⚠️ 저장할 데이터가 없습니다."

    def finalize_dataset(self):
        """[추가] UI 종료 없이 데이터셋을 확정하는 메서드"""
        with self.lock:
            if self.is_recording:
                return "⚠️ 녹화 중에는 확정할 수 없습니다. 먼저 에피소드를 완료하세요."

            print("🚀 최종 데이터셋 확정(Finalize) 시작...")
            start_time = time.time()
            self.dataset.finalize()
            duration = time.time() - start_time
            msg = f"🏁 최종 확정 완료! (소요 시간: {duration:.2f}초)"
            print(msg)
            return msg

# --- 글로벌 노드 인스턴스 ---
recorder = None

def get_live_image():
    if recorder is not None and recorder.current_frame is not None:
        return recorder.current_frame
    return np.zeros((480, 640, 3), dtype=np.uint8)

def launch_ui():
    global recorder
    if not rclpy.ok():
        rclpy.init()

    recorder = GradioLeRobotVideoRecorder()

    # ROS2 스레드 분리
    ros_thread = threading.Thread(target=lambda: rclpy.spin(recorder), daemon=True)
    ros_thread.start()

    with gr.Blocks(title="LeRobot Video Collector") as demo:
        gr.Markdown("# 🤖 LeRobot v3.0 비디오 데이터 수집 GUI")

        with gr.Row():
            with gr.Column(scale=2):
                image_output = gr.Image(label="Kinect Live Feed")
                timer = gr.Timer(0.1)
                timer.tick(get_live_image, outputs=image_output)

            with gr.Column(scale=1):
                status_text = gr.Textbox(label="상태", value="대기 중")
                start_btn = gr.Button("🔴 녹화 시작 (Start)", variant="primary")
                next_btn = gr.Button("💾 에피소드 완료 (Next)", variant="secondary")
                finish_btn = gr.Button("🏁 전체 종료 및 확정 (Finalize)", variant="stop")

        # --- 이벤트 연결 ---
        # 1. 녹화 시작
        start_btn.click(recorder.start_rec, outputs=status_text)

        # 2. 에피소드 단위 저장 (MP4 인코딩)
        next_btn.click(recorder.next_episode, outputs=status_text)

        # 3. 최종 확정 (UI를 닫지 않고 상태만 업데이트)
        finish_btn.click(recorder.finalize_dataset, outputs=status_text)

    # UI 실행 (메인 스레드를 점유하여 웹 페이지 유지)
    print("🌐 Gradio 서버를 시작합니다...")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

    # 사용자가 터미널에서 Ctrl+C를 누르거나 프로세스를 종료하면 아래가 실행됩니다.
    print("\n🛑 서버가 종료되었습니다.")
    rclpy.shutdown()

if __name__ == "__main__":
    launch_ui()