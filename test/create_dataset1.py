import gradio as gr
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import shutil
import threading
import time
import os
from queue import Queue


import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, JointState, Image as ROSImage
import cv2


try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    import lerobot.datasets.lerobot_dataset as lr_ds
    LeRobotDataset = lr_ds.LeRobotDataset


class DatasetCollector(Node):
    def __init__(self):
        super().__init__('lerobot_data_collector')

        # --- 초기 설정값 ---
        self.record_sec = 10
        self.wait_sec = 5
        self.repo_id = "uon"
        self.dataset_name = "triple-cam-dataset"
        self.root_path = "../outputs/dataset"  # 기본 루트 경로

        self.dataset = None
        self.is_recording = False
        self.is_waiting = False
        self.saved_episodes_count = 0
        self.fps = 30
        self.lock = threading.Lock()


        self.total_progress = 0
        self.status_msg = "대기 중"

        # 데이터 저장 변수
        self.latest_image_top = None
        self.latest_image_wrist_right = None
        self.latest_image_wrist_left = None
        self.latest_joints_right = np.zeros(7, dtype=np.float32)
        self.latest_joints_left = np.zeros(7, dtype=np.float32)

        self.save_queue = Queue()
        self.is_saving_background = False

        # --- 구독자 설정 (카메라 3대 + 관절 2개) ---
        self.sub_top = self.create_subscription(CompressedImage, '/right/camera/cam_top/color/image_rect_raw/compressed', self._top_image_callback, 10)
        self.sub_wrist_right = self.create_subscription(CompressedImage, '/right/camera/cam_wrist/color/image_rect_raw/compressed', self._wrist_right_image_callback, 10)
        self.sub_wrist_left = self.create_subscription(CompressedImage, '/left/camera/cam_wrist/color/image_rect_raw/compressed', self._wrist_left_image_callback, 10)
        self.sub_joints_right = self.create_subscription(JointState, '/joint_states', self._joint_right_callback, 10)
        self.sub_joints_left = self.create_subscription(JointState, '/left_robot/leader/joint_states', self._joint_left_callback, 10)


        # LeRobot은 일반적으로 (C, H, W) 형식을 사용합니다.
        self.features_config = {
            "observation.state": (14,),
            "action": (14,),
            "observation.images.top": (3, 480, 640),
            "observation.images.wrist_right": (3, 480, 640),
            "observation.images.wrist_left": (3, 480, 640)
        }


        threading.Thread(target=lambda: rclpy.spin(self), daemon=True).start()
        threading.Thread(target=self._background_save_worker, daemon=True).start()


    # --- 이미지 디코딩 및 콜백 ---
    def _decode_image(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
        except: return None


    def _top_image_callback(self, msg):
        img = self._decode_image(msg)
        if img:
            with self.lock: self.latest_image_top = img
    def _wrist_right_image_callback(self, msg):
        img = self._decode_image(msg)
        if img:
            with self.lock: self.latest_image_wrist_right = img
    def _wrist_left_image_callback(self, msg):
        img = self._decode_image(msg)
        if img:
            with self.lock: self.latest_image_wrist_left = img
    def _joint_right_callback(self, msg):
        with self.lock:
            if len(msg.position) >= 7: self.latest_joints_right = np.array(msg.position[:7], dtype=np.float32)
    def _joint_left_callback(self, msg):
        with self.lock:
            if len(msg.position) >= 7: self.latest_joints_left = np.array(msg.position[:7], dtype=np.float32)


    # --- 데이터셋 초기화 ---
    def setup_dataset(self, root_path, repo_id, dataset_name, task_label, wait_sec, record_sec):
        self.root_path = root_path
        self.repo_id = repo_id
        self.dataset_name = dataset_name
        self.task_label = task_label
        self.wait_sec = wait_sec
        self.record_sec = record_sec


        # 경로 생성 규칙: [루트]/[Repo ID]/[데이터셋 이름]
        full_save_dir = Path(self.root_path) / self.repo_id / self.dataset_name
        self.local_dir = full_save_dir.absolute()


        if self.local_dir.exists():
            shutil.rmtree(self.local_dir)

        # LeRobot 데이터셋 생성 시 repo_id는 [ID]/[Name] 형식으로 전달
        lerobot_repo_id = f"{self.repo_id}/{self.dataset_name}"

        # 핵심 수정: features에 "task"를 추가해야 add_frame 시 오류가 발생하지 않습니다.
        # 또한 이미지 shape를 (C, H, W) 형식으로 맞추는 것이 안전합니다.
        features = {
            "observation.state": {"dtype": "float32", "shape": self.features_config["observation.state"]},
            "action": {"dtype": "float32", "shape": self.features_config["action"]},
            "observation.images.top": {"dtype": "video", "shape": self.features_config["observation.images.top"], "names": ["color"], "video_backend": "pyav"},
            "observation.images.wrist_right": {"dtype": "video", "shape": self.features_config["observation.images.wrist_right"], "names": ["color"], "video_backend": "pyav"},
            "observation.images.wrist_left": {"dtype": "video", "shape": self.features_config["observation.images.wrist_left"], "names": ["color"], "video_backend": "pyav"},
            "task": {"dtype": "string", "shape": (1,)},
        }

        self.dataset = LeRobotDataset.create(repo_id=lerobot_repo_id, root=self.local_dir, fps=self.fps, features=features, use_videos=True)
        return f"✅ 초기화 성공: {self.local_dir}"


    def start_workflow(self):
        if self.dataset is None: return "❌ 초기화 먼저 하세요."
        threading.Thread(target=self._workflow_loop, daemon=True).start()
        return "⏳ 준비 중..."


    def _workflow_loop(self):
        total_time = self.wait_sec + self.record_sec
        self.is_waiting = True
        start_wait = time.time()
        while (time.time() - start_wait) < self.wait_sec:
            elapsed = time.time() - start_wait
            self.total_progress = (elapsed / total_time) * 100
            self.status_msg = f"⏳ 대기 중... ({self.wait_sec - elapsed:.1f}s)"
            time.sleep(0.05)
        self.is_waiting = False


        self.is_recording = True
        frames_to_save = []
        start_rec = time.time()
        while self.is_recording:
            loop_start = time.time()
            elapsed_rec = loop_start - start_rec
            if elapsed_rec >= self.record_sec: break
            with self.lock:
                if all([self.latest_image_top, self.latest_image_wrist_right, self.latest_image_wrist_left]):
                    combined_state = np.concatenate([self.latest_joints_right, self.latest_joints_left])
                    frames_to_save.append({
                        "state": combined_state.copy(), "action": combined_state.copy(),
                        "img_top": self.latest_image_top.copy(), "img_wrist_r": self.latest_image_wrist_right.copy(), "img_wrist_l": self.latest_image_wrist_left.copy()
                    })
            self.total_progress = ((self.wait_sec + elapsed_rec) / total_time) * 100
            self.status_msg = f"🔴 녹화 중... ({elapsed_rec:.1f}s)"
            time.sleep(max(0, (1.0 / self.fps) - (time.time() - loop_start)))


        self.is_recording = False
        if frames_to_save:
            self.save_queue.put(frames_to_save)
            self.status_msg = "✅ 녹화 완료! (저장 중)"
        self.total_progress = 0


    def _background_save_worker(self):
        while True:
            frames = self.save_queue.get()
            self.is_saving_background = True
            try:
                for f in frames:
                    # PIL Image를 그대로 넣어도 LeRobotDataset이 처리하지만,
                    # features에 정의된 shape와 일치하는지 내부적으로 검증합니다.
                    self.dataset.add_frame({
                        "observation.state": torch.from_numpy(f["state"]).float(),
                        "action": torch.from_numpy(f["action"]).float(),
                        "observation.images.top": f["img_top"],
                        "observation.images.wrist_right": f["img_wrist_r"],
                        "observation.images.wrist_left": f["img_wrist_l"],
                        "task": self.task_label,
                    })
                self.dataset.save_episode()
                self.saved_episodes_count += 1
                print(f"에피소드 저장 완료! 현재 총: {self.saved_episodes_count}")
            except Exception as e:
                print(f"저장 중 오류 발생: {e}")
                import traceback
                traceback.print_exc() # 상세 오류 출력
            self.is_saving_background = False
            self.save_queue.task_done()


    def get_ui_data(self):
        display_status = self.status_msg
        if self.is_saving_background:
            display_status += f" 💾 [저장 중... 큐:{self.save_queue.qsize()}]"

        total_time = self.wait_sec + self.record_sec
        marker_pos = (self.wait_sec / total_time) * 100
        bar_color = '#FFE0B2' if self.is_waiting else '#C8E6C9'

        html_bar = f"""
       <div style="width: 100%; background-color: #f5f5f5; border-radius: 6px; height: 35px; position: relative; overflow: hidden; border: 1px solid #ddd;">
           <div style="width: {self.total_progress}%; background-color: {bar_color}; height: 100%; transition: width 0.1s linear;"></div>
           <div style="position: absolute; left: {marker_pos}%; top: 0; width: 3px; height: 100%; background-color: #555; z-index: 10;"></div>
           <div style="position: absolute; width: 100%; text-align: center; top: 0; line-height: 35px; font-weight: bold; color: #444; pointer-events: none;">{display_status}</div>
       </div>
       """
        with self.lock:
            def fmt(joints, name):
                deg = [np.degrees(val) for val in joints]
                return f"[{name}] J1-6: {', '.join([f'{d:.1f}' for d in deg[:6]])} | G: {deg[6]:.1f}°"
            full_text = fmt(self.latest_joints_right, "Right") + "\n" + fmt(self.latest_joints_left, "Left")
            return html_bar, self.saved_episodes_count, self.latest_image_top, self.latest_image_wrist_right, self.latest_image_wrist_left, full_text


# --- 메인 실행 ---
def main():
    rclpy.init()
    collector = DatasetCollector()

    with gr.Blocks() as demo:
        gr.Markdown("# 🤖 Dual-Arm LeRobot Multi-Path Collector")

        with gr.Row():
            camera_top = gr.Image(label="Top Camera (Kinect)", streaming=True, interactive=False)
            camera_wrist_r = gr.Image(label="Right Wrist", streaming=True, interactive=False)
            camera_wrist_l = gr.Image(label="Left Wrist", streaming=True, interactive=False)

        joint_display = gr.Textbox(label="Robot Status (Degrees)", lines=2, interactive=False)

        with gr.Row():
            with gr.Column(scale=2):
                root_path = gr.Textbox(label="1. 루트 저장 경로", value="outputs/dataset")
                repo_id = gr.Textbox(label="2. Repo ID (폴더명)", value="uon")
                dataset_name = gr.Textbox(label="3. 데이터셋 이름", value="triple-cam-task")
            with gr.Column(scale=1):
                task_label = gr.Textbox(label="태스크 라벨", value="pick_up")
                wait_duration = gr.Number(label="대기 시간(초)", value=5)
                record_duration = gr.Number(label="녹화 시간(초)", value=10)
                ep_count_display = gr.Number(label="저장 완료 에피소드", value=0, interactive=False)
                init_btn = gr.Button("⚙️ 경로 설정 및 초기화", variant="secondary")

        progress_html = gr.HTML()
        start_btn = gr.Button("🔴 녹화 시작", variant="primary")

        timer = gr.Timer(0.1)
        timer.tick(collector.get_ui_data, outputs=[progress_html, ep_count_display, camera_top, camera_wrist_r, camera_wrist_l, joint_display])

        init_btn.click(collector.setup_dataset,
                       [root_path, repo_id, dataset_name, task_label, wait_duration, record_duration],
                       ep_count_display)

        start_btn.click(collector.start_workflow)


    demo.launch(css=".gradio-container {max-width: 1400px !important}")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
