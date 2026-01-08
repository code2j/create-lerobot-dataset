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
        self.current_frame_secondary_for_ui = None # 두 번째 UI용 프레임

        self.max_time = 10.0
        self.start_time = 0.0
        self.elapsed_time = 0.0
        self.status_msg = "대기 중"

        self.latest_data = {
            "image": None,
            "image_secondary": None, # 두 번째 이미지 버퍼 추가
            "state": torch.zeros(6),
            "action": torch.zeros(6)
        }

        # 1번 카메라 (Kinect)
        self.subscription = self.create_subscription(
            CompressedImage, '/kinect/color/compressed', self._kinect_callback, 10)

        # 2번 카메라 (플레이스홀더 토픽)
        self.subscription_secondary = self.create_subscription(
            CompressedImage, '/right/camera/cam_wrist/color/image_rect_raw/compressed', self._secondary_callback, 10)

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
                    print(f"\n[INFO] 기존 데이터셋 발견 및 로드 완료")
                    gr.Info("📂 기존 데이터셋을 성공적으로 불러왔습니다.")
                else:
                    self.dataset = LeRobotDataset.create(
                        repo_id=self.repo_id,
                        root=dataset_path,
                        fps=30,
                        features={
                            "observation.image": { # 메인 카메라
                                "dtype": "video",
                                "shape": (3, 480, 640),
                                "names": ["channels", "height", "width"],
                                "info": {"fps": 30, "video_backend": "pyav"}
                            },
                            "observation.image_secondary": { # 보조 카메라 추가
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
                    print(f"\n[INFO] 새 데이터셋 생성 완료 (2-Cam 설정)")
                    gr.Info("✨ 새 데이터셋(멀티캠)이 생성되었습니다.")

                self.status_msg = "✅ 초기화 완료"
                return self.status_msg, self.get_ep_count()
            except Exception as e:
                print(f"\n[ERROR] 초기화 실패: {e}")
                gr.Error(f"❌ 초기화 실패: {e}")
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

    def _secondary_callback(self, msg):
        """두 번째 카메라 콜백 함수"""
        np_arr = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.resize(img, (640, 480))
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            with self.lock:
                self.latest_data["image_secondary"] = rgb_img
                self.current_frame_secondary_for_ui = rgb_img

    def _recording_loop(self):
        if not self.is_recording or self.dataset is None:
            return
        with self.lock:
            # 두 카메라 데이터 중 하나라도 없으면 기록 스킵 (데이터 정렬 유지)
            if self.latest_data["image"] is None:
                return

            # 보조 카메라 데이터가 아직 없다면 검은색 화면으로 대체 (에러 방지용)
            if self.latest_data["image_secondary"] is None:
                img_secondary = np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                img_secondary = self.latest_data["image_secondary"]

            self.elapsed_time = time.time() - self.start_time
            if self.elapsed_time >= self.max_time:
                self.is_recording = False
                threading.Thread(target=self._save_episode_internal, daemon=True).start()
                return

            img_tensor = torch.from_numpy(self.latest_data["image"]).permute(2, 0, 1)
            img_secondary_tensor = torch.from_numpy(img_secondary).permute(2, 0, 1)

            self.dataset.add_frame({
                "observation.image": img_tensor,
                "observation.image_secondary": img_secondary_tensor, # 추가된 필드 저장
                "observation.state": self.latest_data["state"],
                "action": self.latest_data["action"],
                "task": "multi_cam_task"
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
            print(f"[SAVE] 멀티캠 데이터 인코딩 중... ({self.frame_count} 프레임)")

            try:
                self.dataset.save_episode()
                print(f"[SUCCESS] 저장 완료 (총: {self.dataset.num_episodes})")
                gr.Info(f"✅ 에피소드 {self.dataset.num_episodes - 1} 저장 완료!")
                self.status_msg = "✅ 저장 완료"
            except Exception as e:
                print(f"[ERROR] 저장 오류: {e}")
                self.status_msg = "❌ 저장 오류"
            finally:
                self.frame_count = 0
                self.elapsed_time = 0.0
                self.is_saving = False

    def start_rec(self):
        if self.dataset is None:
            gr.Warning("⚠️ 먼저 데이터셋 초기화/불러오기를 완료해 주세요!")
            return self.status_msg, 0
        if self.is_recording:
            gr.Info("이미 녹화가 진행 중입니다.")
            return self.status_msg, self.get_ep_count()
        if self.is_saving:
            gr.Warning("⏳ 현재 저장 작업이 진행 중입니다.")
            return self.status_msg, self.get_ep_count()

        with self.lock:
            self._clear_buffer_internal()
            self.is_recording = True
            self.frame_count = 0
            self.start_time = time.time()
            self.elapsed_time = 0.0
            self.status_msg = "🔴 녹화 중..."
            print(f"\n[REC] 멀티캠 녹화 시작 (최대 {self.max_time}초)")
        return self.status_msg, self.get_ep_count()

    def retry_rec(self):
        if self.dataset is None:
            gr.Warning("⚠️ 먼저 데이터셋 초기화/불러오기를 완료해 주세요!")
            return self.status_msg, 0
        with self.lock:
            self._clear_buffer_internal()
            self.is_recording = True
            self.frame_count = 0
            self.start_time = time.time()
            self.elapsed_time = 0.0
            self.status_msg = "🔄 재시도 중"
            gr.Info("🔄 처음부터 다시 녹화합니다.")
        return self.status_msg, self.get_ep_count()

    def _clear_buffer_internal(self):
        if self.dataset is not None:
            if hasattr(self.dataset, 'clear_episode_buffer'):
                self.dataset.clear_episode_buffer()
            else:
                self.dataset._frames = []

    def next_episode(self):
        if self.dataset is None:
            gr.Warning("⚠️ 먼저 데이터셋 초기화/불러오기를 완료해 주세요!")
            return self.status_msg, 0
        if self.is_recording:
            self.is_recording = False
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
            self.dataset.finalize()
            gr.Info("🏁 데이터셋 최종 확정이 완료되었습니다!")
            self.status_msg = "🏁 수집 종료"
            return self.status_msg, self.get_ep_count()

    def update_ui_components(self):
        progress_val = 0
        bar_label = f"준비 완료: 최대 {self.max_time:.1f}s"

        if self.is_recording:
            progress_val = min(100, (self.elapsed_time / self.max_time) * 100)
            bar_label = f"⌛ 녹화 중: {self.elapsed_time:.1f}s / {self.max_time:.1f}s"
        elif self.is_saving:
            progress_val = 100
            bar_label = "💾 저장 중... 잠시만 기다려주세요."

        return (
            self.current_frame_for_ui,
            self.current_frame_secondary_for_ui, # 두 번째 영상 UI 갱신
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
        gr.Markdown("# 🤖 LeRobot v3.0 멀티캠 수집기")

        with gr.Accordion("⚙️ 설정", open=True):
            with gr.Row():
                repo_id_input = gr.Textbox(label="Repo ID", value="uon/triple-cam-task-video")
                root_path_input = gr.Textbox(label="Root Path", value="outputs/dataset")
            max_time_input = gr.Number(label="최대 녹화 시간 (초)", value=10.0, precision=1)
            init_btn = gr.Button("🔄 데이터셋 초기화/불러오기")

        with gr.Row():
            with gr.Column(scale=2):
                with gr.Row(): # 영상 피드 두 개 나란히 배치
                    image_output = gr.Image(label="Main Camera (Kinect)")
                    image_secondary_output = gr.Image(label="Secondary Camera")

                progress_bar = gr.Slider(label="준비 완료", minimum=0, maximum=100, value=0, interactive=False)

            with gr.Column(scale=1):
                ep_count_display = gr.Label(value="0", label="현재 에피소드 수")
                status_text = gr.Label(value="대기 중", label="현재 상태")

                with gr.Row():
                    start_btn = gr.Button("🔴 시작", variant="primary")
                    retry_btn = gr.Button("🔄 리트라이", variant="secondary")

                next_btn = gr.Button("💾 완료 및 저장", variant="secondary")
                finish_btn = gr.Button("🏁 전체 확정", variant="stop")

        gr.Timer(0.1).tick(
            recorder.update_ui_components,
            outputs=[image_output, image_secondary_output, progress_bar, status_text, ep_count_display]
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