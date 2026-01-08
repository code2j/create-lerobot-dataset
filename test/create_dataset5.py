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
import os

# 허브 접속 차단 (로컬 우선)
os.environ["HF_HUB_OFFLINE"] = "1"

class GradioLeRobotVideoRecorder(Node):
    def __init__(self):
        super().__init__('gradio_lerobot_video_recorder')
        self.lock = threading.Lock()

        self.dataset = None
        self.repo_id = ""
        self.root_path = None

        self.is_recording = False
        self.is_saving = False
        self.frame_count = 0
        self.current_frame_for_ui = None

        self.max_time = 10.0
        self.start_time = 0.0
        self.elapsed_time = 0.0
        self.status_msg = "대기 중"

        self.latest_data = {
            "image": None,
            "state": torch.zeros(6),
            "action": torch.zeros(6)
        }

        self.subscription = self.create_subscription(
            CompressedImage, '/kinect/color/compressed', self._kinect_callback, 10)

        self.create_timer(1.0 / 30.0, self._recording_loop)

    def get_ep_count(self):
        return self.dataset.num_episodes if self.dataset is not None else 0

    def init_dataset(self, repo_id, root_dir):
        with self.lock:
            try:
                self.repo_id = repo_id
                self.root_path = Path(root_dir).absolute()
                dataset_path = self.root_path / self.repo_id
                info_json = dataset_path / "meta" / "info.json"

                if info_json.exists():
                    self.dataset = LeRobotDataset(repo_id=self.repo_id, root=dataset_path)
                    print(f"\n[INFO] 기존 데이터셋 로드 완료")
                    gr.Info("📂 기존 데이터셋을 성공적으로 불러왔습니다.") # 토스트 알림
                else:
                    self.dataset = LeRobotDataset.create(
                        repo_id=self.repo_id,
                        root=dataset_path,
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
                    print(f"\n[INFO] 새 데이터셋 생성 완료")
                    gr.Info("✨ 새 데이터셋이 생성되었습니다.") # 토스트 알림

                self.status_msg = "✅ 초기화 완료"
                return self.status_msg, self.get_ep_count()
            except Exception as e:
                print(f"\n[ERROR] 초기화 실패: {e}")
                gr.Error(f"❌ 초기화 실패: {e}") # 에러 토스트
                self.status_msg = "❌ 초기화 실패"
                return self.status_msg, 0

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
        if not self.is_recording or self.dataset is None:
            return
        with self.lock:
            if self.latest_data["image"] is None:
                return
            self.elapsed_time = time.time() - self.start_time
            if self.elapsed_time >= self.max_time:
                self.is_recording = False
                threading.Thread(target=self._save_episode_internal).start()
                return

            img_tensor = torch.from_numpy(self.latest_data["image"]).permute(2, 0, 1)
            self.dataset.add_frame({
                "observation.image": img_tensor,
                "observation.state": self.latest_data["state"],
                "action": self.latest_data["action"],
                "task": "kinect_video_task"
            })
            self.frame_count += 1

    def _save_episode_internal(self):
        """에피소드 저장 및 인코딩 프로세스 (Lock 강화 버전)"""
        # 저장 시작 전, 녹화 상태를 확실히 끔
        with self.lock:
            if self.frame_count == 0:
                self.status_msg = "⚠️ 데이터 없음"
                self.is_saving = False
                return

            self.is_saving = True
            self.is_recording = False # 확실히 녹화 중단
            self.status_msg = "💾 저장 중... 잠시만 기다려주세요"
            print(f"[SAVE] 저장 및 인코딩 중... ({self.frame_count} 프레임)")

            try:
                # [핵심] 저장하는 동안 다른 스레드가 add_frame을 하지 못하도록
                # Lock 안에서 save_episode를 호출합니다.
                self.dataset.save_episode()

                print(f"[SUCCESS] 저장 완료 (총 에피소드: {self.dataset.num_episodes})")
                gr.Info(f"✅ 에피소드 {self.dataset.num_episodes - 1} 저장 완료!")
                self.status_msg = "✅ 저장 완료"
            except Exception as e:
                print(f"[ERROR] 저장 중 오류 발생: {e}")
                self.status_msg = "❌ 저장 오류"
            finally:
                self.frame_count = 0
                self.elapsed_time = 0.0
                self.is_saving = False

    def start_rec(self):
        if self.dataset is None:
            gr.Warning("⚠️ 먼저 데이터셋 초기화/불러오기를 완료해 주세요!")
            return self.status_msg, 0

        # [개선] 이미 녹화 중이라면 '시작' 버튼은 아무 역할도 하지 않게 보호
        if self.is_recording:
            gr.Info("이미 녹화가 진행 중입니다.")
            return self.status_msg, self.get_ep_count()

        if self.is_saving:
            gr.Warning("⏳ 현재 저장 작업이 진행 중입니다.")
            return self.status_msg, self.get_ep_count()

        with self.lock:
            # 시작 전 버퍼를 확실히 비워 에러 방지 (핵심!)
            self._clear_buffer_internal()

            self.is_recording = True
            self.frame_count = 0
            self.start_time = time.time()
            self.elapsed_time = 0.0
            self.status_msg = "🔴 녹화 중..."
            print(f"\n[REC] 녹화 시작 (최대 {self.max_time}초)")
        return self.status_msg, self.get_ep_count()

    def retry_rec(self):
        if self.dataset is None:
            gr.Warning("⚠️ 먼저 데이터셋 초기화/불러오기를 완료해 주세요!")
            return self.status_msg, 0

        with self.lock:
            # 리트라이는 현재 진행 상황을 무조건 폐기하고 다시 시작
            self._clear_buffer_internal()

            self.is_recording = True
            self.frame_count = 0
            self.start_time = time.time()
            self.elapsed_time = 0.0
            self.status_msg = "🔄 재시도 중"
            print(f"\n[RETRY] 데이터 폐기 및 재시작")
            gr.Info("🔄 처음부터 다시 녹화합니다.")
        return self.status_msg, self.get_ep_count()

        with self.lock:
            if hasattr(self.dataset, 'clear_episode_buffer'): self.dataset.clear_episode_buffer()
            else: self.dataset._frames = []
            self.is_recording = True
            self.frame_count = 0
            self.start_time = time.time()
            self.elapsed_time = 0.0
            self.status_msg = "🔄 재시도 중"
            gr.Info("🔄 현재 녹화 프레임을 폐기하고 다시 시작합니다.")
        return self.status_msg, self.get_ep_count()
    def _clear_buffer_internal(self):
        """데이터셋 내부 버퍼를 안전하게 비우는 공통 함수"""
        if self.dataset is not None:
            if hasattr(self.dataset, 'clear_episode_buffer'):
                self.dataset.clear_episode_buffer()
            else:
                self.dataset._frames = [] # 구버전 호환용

    def next_episode(self):
        if self.dataset is None:
            gr.Warning("⚠️ 먼저 데이터셋 초기화/불러오기를 완료해 주세요!")
            return self.status_msg, 0

        if self.is_recording:
            # 1. 먼저 녹화 플래그를 꺼서 타이머 루프가 진입하지 못하게 함
            self.is_recording = False
            # 2. 저장 스레드 실행
            threading.Thread(target=self._save_episode_internal, daemon=True).start()

        return self.status_msg, self.get_ep_count()

    def finalize_dataset(self):
        if self.dataset is None:
            gr.Warning("⚠️ 먼저 데이터셋 초기화/불러오기를 완료해 주세요!")
            return "⚠️ 미설정", 0

        with self.lock:
            if self.is_recording or self.is_saving:
                gr.Warning("⚠️ 작업 중에는 확정할 수 없습니다.")
                return "⚠️ 작업 중", self.get_ep_count()

            print("\n[FINALIZE] 데이터셋 최종 확정 시작...")
            self.dataset.finalize()
            print(f"[SUCCESS] 모든 작업 완료")
            gr.Info("🏁 데이터셋 최종 확정이 완료되었습니다!") # 최종 완료 알림
            self.status_msg = "🏁 수집 종료"
            return self.status_msg, self.get_ep_count()

    def update_ui_components(self):
        progress_val = 0
        bar_label = "진행 시간: 0.0s / 0.0s"

        if self.is_recording:
            progress_val = min(100, (self.elapsed_time / self.max_time) * 100)
            bar_label = f"⌛ 녹화 중: {self.elapsed_time:.1f}s / {self.max_time:.1f}s"
        elif self.is_saving:
            progress_val = 100
            bar_label = "💾 저장 중... 인코딩 중입니다. 잠시만 기다려주세요."
        else:
            bar_label = f"준비 완료: 최대 {self.max_time:.1f}s"

        return (
            self.current_frame_for_ui,
            gr.update(value=progress_val, label=bar_label),
            self.status_msg,
            self.get_ep_count()
        )

# --- UI 함수 ---
def launch_ui():
    global recorder
    if not rclpy.ok(): rclpy.init()
    recorder = GradioLeRobotVideoRecorder()

    ros_thread = threading.Thread(target=lambda: rclpy.spin(recorder), daemon=True)
    ros_thread.start()

    with gr.Blocks(title="LeRobot Collector") as demo:
        gr.Markdown("# 🤖 LeRobot v3.0 수집기")

        with gr.Accordion("⚙️ 설정", open=True):
            with gr.Row():
                repo_id_input = gr.Textbox(label="Repo ID", value="uon/triple-cam-task-video")
                root_path_input = gr.Textbox(label="Root Path", value="outputs/dataset")
            max_time_input = gr.Number(label="최대 녹화 시간 (초)", value=10.0, precision=1)
            init_btn = gr.Button("🔄 데이터셋 초기화/불러오기")

        with gr.Row():
            with gr.Column(scale=2):
                image_output = gr.Image(label="Live Feed")
                progress_bar = gr.Slider(label="준비 완료", minimum=0, maximum=100, value=0, interactive=False)

            with gr.Column(scale=1):
                ep_count_display = gr.Label(value="0", label="현재 에피소드 수")
                status_text = gr.Label(value="대기 중", label="현재 상태")

                with gr.Row():
                    start_btn = gr.Button("🔴 시작", variant="primary")
                    retry_btn = gr.Button("🔄 리트라이", variant="secondary")

                next_btn = gr.Button("💾 완료 및 저장", variant="secondary")
                finish_btn = gr.Button("🏁 전체 확정", variant="stop")

        gr.Timer(1/60).tick(
            recorder.update_ui_components,
            outputs=[image_output, progress_bar, status_text, ep_count_display]
        )

        init_btn.click(recorder.init_dataset, inputs=[repo_id_input, root_path_input], outputs=[status_text, ep_count_display])
        max_time_input.change(lambda v: setattr(recorder, 'max_time', float(v if v else 0)), inputs=max_time_input)
        start_btn.click(recorder.start_rec, outputs=[status_text, ep_count_display])
        retry_btn.click(recorder.retry_rec, outputs=[status_text, ep_count_display])
        next_btn.click(recorder.next_episode, outputs=[status_text, ep_count_display])
        finish_btn.click(recorder.finalize_dataset, outputs=[status_text, ep_count_display])

    demo.launch(server_name="0.0.0.0", server_port=7860)
    rclpy.shutdown()

if __name__ == "__main__":
    launch_ui()