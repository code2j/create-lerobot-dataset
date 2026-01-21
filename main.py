import gradio as gr
import cv2
import numpy as np
import time
import threading
from pynput import keyboard

# ROS2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, JointState

# 내 모듈
from subscriber_hub import SubscriberHub
from data_manager import LerobotDatasetManager



# ------------------------------------------
# 웹 인터페이스
# ------------------------------------------
class GradioWeb:
    def __init__(self, hub: SubscriberHub):
        self.hub = hub
        self.dataset_manager = LerobotDatasetManager(hub)
        self.interface = self._build_interface()
        self.last_key_time = 0
        self.chatter_threshold = 0.2
        self.page_down_pressed = False
        self.delete_pressed = False
        self.listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self.listener.start()

    def _timer_callback(self):
        """UI 업데이트 타이머 콜백"""
        k_msg, w_msg, f_msg, l_msg = self.hub.get_latest_msg()
        return (
            self._decode_image(k_msg), \
            self._decode_image(w_msg), \
            self._format_joint_state(f_msg), \
            self._format_joint_state(l_msg), \
            self.dataset_manager.get_display_status()
        )

    def _format_joint_state(self, msg: JointState):
        """JointState를 디그리 문자열로 변환(UI 렌더링용)"""
        if msg is None:
            return "No Data"
        return "\n".join([f"{n}: {p*180/3.14159:.4f}°" for n, p in zip(msg.name, msg.position)])

    def _decode_image(self, msg: CompressedImage):
        """CompressedImage를 OpenCV 이미지로 디코딩"""
        if msg is None:
            return None

        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if cv_image is None:
            return None

        return cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

    def _on_press(self, key):
        """키보드 입력 콜백"""
        if self.dataset_manager.dataset is None:
            return # 데이터셋 초기화 필요

        current_time = time.time()
        if current_time - self.last_key_time < self.chatter_threshold:
            return # 체터링 방지

        # 키입력: Page Down
        if key == keyboard.Key.page_down and not self.page_down_pressed:
            self.page_down_pressed = True
            self.last_key_time = current_time

            # 녹화 가능 상태면 녹화 시작
            if self.dataset_manager.status == "ready":
                self.dataset_manager.record()

            # 이미 녹화 중이면 에피소드 저장
            elif self.dataset_manager.status == "record":
                self.dataset_manager.save()

        # 키입력: Delete
        if key == keyboard.Key.delete and not self.delete_pressed:
            self.delete_pressed = True
            self.last_key_time = current_time

            # 녹화 중이면 에피소드 재시도
            if self.dataset_manager.status == "record":
                self.dataset_manager.retry()

    def _on_release(self, key):
        """키보드 입력 콜백"""
        if key == keyboard.Key.page_down:
            self.page_down_pressed = False
        elif key == keyboard.Key.delete:
            self.delete_pressed = False

    def _build_interface(self):
        with gr.Blocks(title="Robot Teleoperation Monitor", css="""
        .status-output textarea {
            font-size: 44px !important;
            font-weight: bold !important;
            min-height: 30px !important;
            max-height: 70px !important;
            padding: 8px !important;
        }
    """) as demo:
            gr.Markdown("# Robot Teleoperation Monitor")
            timer = gr.Timer(0.1)

            # 영상 데이터 상단 배치
            with gr.Row():
                kinect_view = gr.Image(label="Top View")
                wrist_view = gr.Image(label="Wrist View")

            # 조인트 정보 중간 배치
            with gr.Row():
                with gr.Column(scale=1):
                    follower_view = gr.Textbox(label="Follower Joints", lines=7)
                with gr.Column(scale=1):
                    leader_view = gr.Textbox(label="Leader Joints", lines=7)

            # 상태 출력 크게 표시
            status_output = gr.Textbox(
                label="Status",
                interactive=False,
                lines=3,
                elem_classes="status-output"
            )

            # 설정 및 버튼 하단 배치
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Dataset Configuration")
                    repo_id_input = gr.Textbox(label="Repo ID", value="my_dataset")
                    root_dir_input = gr.Textbox(label="Root Directory", value="data")
                    task_name_input = gr.Textbox(label="Task Name", value="teleop")
                    fps_input = gr.Number(label="FPS", value=30)

                with gr.Column(scale=1):
                    gr.Markdown("### Controls")
                    init_btn = gr.Button("1. Init Dataset", variant="primary", scale=1)
                    record_btn = gr.Button("2. Record (PgDn)", variant="secondary", scale=1)
                    save_btn = gr.Button("3. Save Episode (PgDn)", variant="secondary", scale=1)
                    retry_btn = gr.Button("Retry (Del)", variant="stop", scale=1)
                    finalize_btn = gr.Button("4. Finalize Dataset (Finish)", variant="primary", scale=1)

            # 데이터셋 초기화 버튼
            init_btn.click(self.dataset_manager.init_dataset,
                           [repo_id_input, root_dir_input, task_name_input, fps_input],
                           status_output).then(self.dataset_manager.start_timer)

            # 녹화 버튼
            record_btn.click(self.dataset_manager.record,
                             None,
                             status_output)

            # 에피소드 저장 버튼
            save_btn.click(self.dataset_manager.save,
                           None,
                           status_output)

            # 에피소드 재시도 버튼
            retry_btn.click(self.dataset_manager.retry,
                            None,
                            status_output)

            # 데이터셋 최종 확정 버튼
            finalize_btn.click(self.dataset_manager.finalize_dataset,
                               None,
                               status_output)

            # 타이머 콜백
            timer.tick(self._timer_callback,
                       None,
                       [kinect_view, wrist_view, follower_view, leader_view, status_output])

        return demo


    def launch(self):
        self.interface.launch(server_name="0.0.0.0", share=False)

def main():
    rclpy.init()
    hub = SubscriberHub()
    threading.Thread(target=rclpy.spin, args=(hub,), daemon=True).start()
    try:
        GradioWeb(hub).launch()
    finally:
        rclpy.shutdown()

if __name__ == "__main__":
    main()