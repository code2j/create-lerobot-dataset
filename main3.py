import gradio as gr
import cv2
import numpy as np
from pathlib import Path
import time
import os
import threading
import shutil
from pynput import keyboard


# 허깅페이스 오프라인 모드 ON
os.environ["HF_HUB_OFFLINE"] = "1"

# NumPy 2.x 호환성 경고 방지를 위한 설정
os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"

# ros2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, JointState

# lerobot
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_FEATURES


# 내 모듈
from subscriber_hub import SubscriberHub

STATUS = ""

# ------------------------------------------
# 유틸리티
# ------------------------------------------
def decode_image(msg: CompressedImage):
    """압축된 이미지 메시지를 OpenCV 이미지로 변환"""
    try:
        if msg is None:
            return None
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if cv_image is None:
            return None
        cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        return cv_image_rgb
    except Exception as e:
        print(f"이미지 디코딩 오류: {e}")
        return None

def jointState_to_nparray(msg: JointState, target_names: list) -> np.ndarray:
    """JointState -> np.array"""
    if msg is None:
        return np.zeros(len(target_names), dtype=np.float32)

    # 메시지의 {이름: 위치값} 딕셔너리 생성
    name_to_pos_map = dict(zip(msg.name, msg.position))

    # target_names 순서대로 리스트 생성
    ordered_values = [name_to_pos_map.get(name, 0.0) for name in target_names]

    return np.array(ordered_values, dtype=np.float32)


# ------------------------------------------
# Lerobot 데이터 매니저
# ------------------------------------------
class LerobotDatasetManager:
    def __init__(self, subscriber_hub: SubscriberHub):
        self.subscriber_hub = subscriber_hub
        self.dataset = None
        self.lock = threading.Lock() # 스레드 간 안전한 상태 접근을 위한 Lock

        # 시간 및 프레임 추적을 위한 변수 추가
        self.recording_start_time = 0
        self.num_frames = 0
        self.fps = 30
        # --- 상태 관리 강화 ---
        # "ready", "record", "saving", "retry" 등의 상태를 가짐
        self.status = ""
        self.last_save_result_message = "" # 저장 완료 메시지를 전달하기 위한 변수

        self.joint_names = [
            'right_joint1',
            'right_joint2',
            'right_joint3',
            'right_joint4',
            'right_joint5',
            'right_joint6',
            'right_rh_r1_joint'
        ]

        # 정밀 타이머 스레드 관련
        self.stop_event = threading.Event()
        self.timer_thread = None

    def init_dataset(self, repo_id="my_dataset", root_dir="data", task_name="teleop", fps=30) -> str:
        """데이터셋 초기화 및 생성"""
        self.repo_id = repo_id
        self.root_path = Path(root_dir).absolute() / self.repo_id
        self.task_name = task_name
        self.fps = fps

        dataset_path = self.root_path / self.repo_id
        meta_info_path = self.root_path  / "meta" / "info.json"

        if meta_info_path.exists():
            self.dataset = LeRobotDataset(repo_id=self.repo_id, root=self.root_path)
            self.dataset.start_image_writer(num_processes=2, num_threads=4)
            print(f"[Info ] 기존 데이터셋 불러오기 성공")
            return f"✅ 기존 데이터셋 불러오기 성공"

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
            root=self.root_path,
            features=features,
            use_videos=True,
            fps=fps,
            robot_type="omy_f3m",
            image_writer_threads=4,
            image_writer_processes=2
        )

        print(f"[Info ] 데이터셋 초기화 성공 {self.repo_id}")
        return f"✅ 데이터셋 초기화 완료"

    def start_timer(self):
        """녹화 타이머 시작"""
        if self.timer_thread is not None and self.timer_thread.is_alive():
            return

        self.stop_event.clear()
        self.timer_thread = threading.Thread(target=self._timer_loop, daemon=True)
        self.timer_thread.start()

        self.status = "ready"
        print("[Info ] 녹화 준비 완료")

    def stop_timer(self):
        """녹화 타이머 스레드 중지"""
        self.stop_event.set()
        if self.timer_thread:
            self.timer_thread.join()
        self.status = ""
        print("[Info ] 녹화 타이머 중지")

    def _timer_loop(self):
        """녹화 타이머 루프 (별도 스레드)"""
        interval = 1.0 / self.fps
        next_time = time.time()

        while not self.stop_event.is_set():
            kinect_msg, wrist_msg, follower_msg, leader_msg = self.subscriber_hub.get_latest_msg()
            self._record_loop(kinect_msg, wrist_msg, follower_msg, leader_msg)

            next_time += interval
            sleep_time = next_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_time = time.time()

    def _record_loop(self, kinect_msg, wrist_msg, follower_msg, leader_msg):
        """데이터 녹화 및 상태 처리"""
        if self.dataset is None:
            return

        # --- 상태에 따른 분기 처리 ---
        with self.lock: # 스레드 동시 접근 방지
            if self.status == "record":
                if self.recording_start_time == 0:
                    self.recording_start_time = time.time()

                kinect_img = decode_image(kinect_msg)
                wrist_img = decode_image(wrist_msg)
                follower_joint_data = jointState_to_nparray(follower_msg, self.joint_names)
                leader_joint_data = jointState_to_nparray(leader_msg, self.joint_names)

                frame = {
                    'observation.images.cam_top': kinect_img,
                    'observation.images.cam_wrist': wrist_img,
                    'observation.state': follower_joint_data,
                    'action': leader_joint_data,
                    'task': self.task_name
                }
                self.dataset.add_frame(frame)
                self.num_frames += 1

                elapsed_time = time.time() - self.recording_start_time
                if self.num_frames % 10 == 0:
                    print(f"🔴 녹화 중: {elapsed_time:.1f}초 ({self.num_frames} 프레임)")

            elif self.status == "retry":
                print(f"[Info ] 현재 녹화한 에피소드 버퍼 지우는 중...")
                self.dataset.clear_episode_buffer()
                self.status = "ready"
                self.recording_start_time = 0
                self.num_frames = 0
                print("[Info ] 녹화한 에피소드 버퍼 지우기 완료")

    # --- 핵심 수정: 저장 로직을 별도 스레드에서 실행 ---
    def _save_episode_threaded(self):
        """(백그라운드 스레드) 에피소드 저장 및 완료 처리"""
        print(f"[Info ] 에피소드 저장 중... ({self.num_frames} 프레임)")
        try:
            self.dataset.save_episode()
            self.dataset.finalize()
            message = f"✅ 에피소드 저장 완료 ({self.num_frames} 프레임)"
            print(f"[Info ] {message}")
        except Exception as e:
            message = f"❌ 에피소드 저장 실패: {e}"
            print(f"[Error] {message}")

        with self.lock:
            self.status = "ready"
            self.recording_start_time = 0
            self.num_frames = 0
            self.last_save_result_message = message # 완료 또는 실패 메시지 저장

    def record(self):
        """녹화 상태로 변경"""
        with self.lock:
            if self.status == "": return "❌ 데이터 녹화 실패: 초기화 필요"
            if self.status == "record": return "✅ 이미 녹화 중입니다."
            if self.status == "saving": return "⏳ 에피소드 저장 중... 잠시 기다려주세요"
            if self.status == "retry": return "⏳ 현재 에피소드 데이터 제거 중... 잠시 기다려주세요"

            print(f"[Info ] 데이터 매니저 상태 변경: record")
            self.status = "record"
            self.last_save_result_message = "" # 이전 완료 메시지 초기화
            return "🔴 녹화 시작됨"

    def save(self):
        """에피소드 저장 상태로 변경"""
        with self.lock:
            if self.status == "": return "❌ 에피소드 저장 실패: 초기화 필요"
            if self.status == "saving": return "⏳ 이미 에피소드 저장 중입니다."
            if self.num_frames == 0: return "❌ 에피소드 저장 실패: 녹화된 프레임 없음"
            if self.status == "retry": return "⏳ 현재 에피소드 데이터 제거 중... 잠시 기다려주세요"

            print(f"[Info ] 데이터 매니저 상태 변경: saving")
            self.status = "saving"
            # --- 별도 스레드에서 저장 함수 실행 ---
            save_thread = threading.Thread(target=self._save_episode_threaded)
            save_thread.start()
            return "⏳ 에피소드 저장 중..."

    def retry(self):
        with self.lock:
            if self.status == "saving": return "❌ 에피소드 저장 중에 재시도 불가"
            if self.status == "retry": return "⏳ 이미 데이터 제거 중입니다."

            print(f"[Info ] 데이터 매니저 상태 변경: retry")
            self.status = "retry"
            self.last_save_result_message = ""
            return "⏳ 현재 녹화된 에피소드 제거 중..."

    # --- UI 업데이트를 위한 상태 확인 메서드 추가 ---
    def get_display_status(self):
        """UI에 표시할 현재 상태 메시지를 반환"""
        with self.lock:
            # 저장 완료 메시지가 있으면 우선적으로 표시
            if self.last_save_result_message:
                msg = self.last_save_result_message
                self.last_save_result_message = "" # 메시지는 한 번만 표시
                return msg

            # 현재 상태에 따른 메시지 반환
            if self.status == "record":
                elapsed_time = time.time() - self.recording_start_time
                return f"🔴 녹화 중: {elapsed_time:.1f}초 ({self.num_frames} 프레임)"
            elif self.status == "saving":
                return "⏳ 에피소드 저장 중..."
            elif self.status == "ready":
                return "✅ 녹화 준비 완료"
            elif self.status == "retry":
                return "⏳ 현재 녹화된 에피소드 제거 중..."
            else:
                return "초기화되지 않음"




# ------------------------------------------
# 웹 인터페이스
# ------------------------------------------
class GradioWeb:
    def __init__(self, hub: SubscriberHub):
        self.hub = hub
        self.dataset_manager = LerobotDatasetManager(hub)
        self.interface = self.build_interface()

        # 채터링 방지 및 키 상태 관리를 위한 변수
        self.last_key_time = 0
        self.chatter_threshold = 0.2
        self.page_down_pressed = False
        self.delete_pressed = False

        # 키보드 리스너 시작
        self.listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self.listener.start()


    def _format_joint_state(self, msg: JointState):
        """JointState 메시지를 텍스트로 포맷팅"""
        if msg is None:
            return "No Data"

        lines = []
        for name, pos in zip(msg.name, msg.position):
            pos_degrees = pos * 180.0 / 3.14159265359 # deg
            lines.append(f"{name}: {pos_degrees:.4f}°")
        return "\n".join(lines)

    def update_tick(self): # current_status_text 인자 제거
        """UI 업데이트를 위한 Timer 틱 함수"""
        kinect_msg, wrist_msg, follower_msg, leader_msg = self.hub.get_latest_msg()

        kinect_img = decode_image(kinect_msg)
        wrist_img = decode_image(wrist_msg)

        follower_text = self._format_joint_state(follower_msg)
        leader_text = self._format_joint_state(leader_msg)

        # --- 데이터 매니저에서 직접 상태 메시지를 가져옴 ---
        status_text = self.dataset_manager.get_display_status()

        return kinect_img, wrist_img, follower_text, leader_text, status_text

    def _on_press(self, key):
        # 데이터셋이 초기화된 상태에서만 키 입력 처리
        if self.dataset_manager.dataset is None:
            return

        # 채터링 방지
        current_time = time.time()
        if current_time - self.last_key_time < self.chatter_threshold:
            return

        # Page Down: 녹화 시작/저장
        if key == keyboard.Key.page_down:
            if not self.page_down_pressed:
                self.page_down_pressed = True
                self.last_key_time = current_time

                # 데이터 메니저 상태
                status = self.dataset_manager.status

                if status == "":
                    print(f"[Warn ] 데이터 녹화 실패: 초기화 필요")
                    STATUS = "❌ 데이터 녹화 실패: 초기화 필요"
                    return

                elif status == "ready":
                    result = self.dataset_manager.record()

                elif status == "record":
                    result = self.dataset_manager.save()
                else:
                    return

        # Delete: 재시도
        if key == keyboard.Key.delete:
            if not self.delete_pressed:
                self.delete_pressed = True
                self.last_key_time = current_time

                # 데이터 메니저 상태
                status = self.dataset_manager.status

                if status == "record":
                    result = self.dataset_manager.retry()
                else:
                    return



    def _on_release(self, key):
        if key == keyboard.Key.page_down:
            self.page_down_pressed = False
        elif key == keyboard.Key.delete:
            self.delete_pressed = False

    def handle_init(self, repo_id, root_dir, task_name, fps):
        """Init 버튼 클릭 시 데이터셋 초기화 실행"""
        print("[Info ] 데이터셋 초기화 버튼 클릭")
        result = self.dataset_manager.init_dataset(repo_id, root_dir, task_name, int(fps))
        self.dataset_manager.start_timer() # 녹화 쓰레드 시작
        return result

    def handle_record(self):
        """Record 버튼 클릭시 이벤트"""
        print(f"[Info ] 녹화 버튼 클릭")
        result = self.dataset_manager.record()
        return result

    def handle_save_episode(self):
        """Save 버튼 클릭시 이벤트"""
        print(f"[Info ] 저장 버튼 클릭")
        result = self.dataset_manager.save()
        return result

    def handle_retry(self):
        """Retry 버튼 클릭시 이벤트"""
        print(f"[Info ] 재시도 버튼 클릭")
        result = self.dataset_manager.retry()
        return result

    def build_interface(self):
        """Gradio 인터페이스 구성"""
        with gr.Blocks(title="Robot Teleoperation Monitor") as demo:
            gr.Markdown("# Robot Teleoperation Monitor")

            timer = gr.Timer(0.1)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Dataset Configuration")
                    repo_id_input = gr.Textbox(label="Repo ID", value="my_dataset")
                    root_dir_input = gr.Textbox(label="Root Directory", value="data")
                    task_name_input = gr.Textbox(label="Task Name", value="teleop")
                    fps_input = gr.Number(label="FPS", value=30)
                    init_btn = gr.Button("Init Dataset", variant="primary")
                    record_btn = gr.Button("Record", variant="primary")
                    save_btn = gr.Button("Save", variant="primary")
                    retry_btn = gr.Button("Retry", variant="primary")
                    status_output = gr.Textbox(label="Status", interactive=False)

                with gr.Column(scale=2):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Kinect Camera")
                            kinect_view = gr.Image(label="Top View")
                        with gr.Column():
                            gr.Markdown("### Wrist Camera")
                            wrist_view = gr.Image(label="Wrist View")
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Follower Joint States")
                            follower_view = gr.Textbox(label="Follower Joints", lines=7)
                        with gr.Column():
                            gr.Markdown("### Leader Joint States")
                            leader_view = gr.Textbox(label="Leader Joints", lines=7)

            # 버튼 클릭 이벤트 연결
            init_btn.click(
                self.handle_init,
                [repo_id_input, root_dir_input, task_name_input, fps_input],
                status_output
            )
            record_btn.click(
                self.handle_record,
                inputs=None,
                outputs=status_output
            )
            save_btn.click(
                self.handle_save_episode,
                inputs=None,
                outputs=status_output
            )
            retry_btn.click(
                self.handle_retry,
                inputs=None,
                outputs=status_output
            )

            # UI 업데이트 타이머 연결
            timer.tick(
                self.update_tick,
                inputs=None, # 입력 제거
                outputs=[kinect_view, wrist_view, follower_view, leader_view, status_output] # status_output 추가
            )

        return demo

    def launch(self):
        """Gradio 앱 실행"""
        self.interface.launch(server_name="0.0.0.0", share=False)


def main():
    # ROS2 초기화
    rclpy.init()

    # 허브 노드 생성
    hub = SubscriberHub()

    # ROS2 스핀을 별도 스레드에서 실행
    ros_thread = threading.Thread(target=rclpy.spin, args=(hub,), daemon=True)
    ros_thread.start()

    try:
        # Gradio 웹 인터페이스 실행
        web = GradioWeb(hub)
        web.launch()
    except KeyboardInterrupt:
        pass
    finally:
        hub.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
