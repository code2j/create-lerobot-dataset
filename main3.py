import gradio as gr
import cv2
import numpy as np
from pathlib import Path
import time
import os
import threading
import shutil


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



# ------------------------------------------
# 유틸리티
# ------------------------------------------
def decode_image(msg: CompressedImage):
    """압축된 이미지 메시지를 OpenCV 이미지로 변환"""
    try:
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        return cv_image_rgb
    except Exception as e:
        print(f"이미지 디코딩 오류: {e}")
        return None

def jointState_to_nparray(msg: JointState, target_names: list) -> np.ndarray:
    """JointState -> np.array"""
    # 메시지의 {이름: 위치값} 딕셔너리 생성
    name_to_pos_map = dict(zip(msg.name, msg.position))

    # target_names 순서대로 리스트 생성
    ordered_values = [name_to_pos_map.get(name, 0.0) for name in target_names]

    return np.array(ordered_values, dtype=np.float32)


# ------------------------------------------
# 서브스크라이버 허브
# ------------------------------------------
class SubscriberHub(Node):
    def __init__(self, node_name='Subscriber_hub'):
        super().__init__(node_name)

        self.kinect_topic_msg = None
        self.right_wristCame_topic_msg = None
        self.right_follower_topic_msg = None
        self.right_leader_topic_msg = None


        self.init_sub()
        print(f'노드 시작: {node_name}')

    def init_sub(self):
        """서브스크라이버 등록"""
        # 키넥트
        self.create_subscription(
            CompressedImage,
            '/right/camera/cam_top/color/image_rect_raw/compressed',
            self.kinect_callback,
            10
        )

        # 오른쪽 손목 카메라
        self.create_subscription(
            CompressedImage,
            '/right/camera/cam_wrist/color/image_rect_raw/compressed',
            self.right_wrisCam_callback,
            10
        )

        # 오른쪽 로봇 조인트
        self.create_subscription(
            JointState,
            '/right/joint_states',
            self.right_flower_callback,
            10
        )

        # 오른쪽 리더암 조인트
        self.create_subscription(
            JointState,
            '/right_robot/leader/joint_states',
            self.right_leader_callback,
            10
        )

    def kinect_callback(self, msg: CompressedImage) -> None:
        """키넥트 카메라 토픽 콜백"""
        self.kinect_topic_msg = msg

    def right_wrisCam_callback(self, msg: CompressedImage) -> None:
        """오른쪽 손목 카메라 토픽 콜백"""
        self.right_wristCame_topic_msg = msg

    def right_flower_callback(self, msg: JointState) -> None:
        """오른쪽 팔로우 로봇 조인트 토픽 콜백"""
        self.right_follower_topic_msg = msg

    def right_leader_callback(self, msg:JointState) -> None:
        """오른쪽 리더 로봇 조인트 토픽 콜백"""
        self.right_leader_topic_msg = msg

    def get_latest_msg(self):
        """가장 최신의 데이터 리턴"""
        return (self.kinect_topic_msg,
                self.right_wristCame_topic_msg,
                self.right_follower_topic_msg,
                self.right_leader_topic_msg)

    def clear_latest_msg(self):
        """모든 토픽 데이터 초기화"""
        self.kinect_topic_msg = None
        self.right_wristCame_topic_msg = None
        self.right_follower_topic_msg = None
        self.right_leader_topic_msg = None


# ------------------------------------------
# Lerobot 데이터 매니저
# ------------------------------------------
class LerobotDatasetManager:
    def __init__(self, subscriber_hub: SubscriberHub):
        self.subscriber_hub = subscriber_hub
        self.dataset = None
        self.lock = threading.Lock()

        # 시간 및 프레임 추적을 위한 변수 추가
        self.recording_start_time = 0
        self.num_frames = 0
        self.fps = 30
        self.status = ""
        self.joint_names = [
            'right_joint1', 'right_joint2', 'right_joint3',
            'right_joint4', 'right_joint5', 'right_joint6',
            'right_rh_r1_joint'
        ]

    def init_dataset(self, repo_id="my_dataset", root_dir="data", task_name="teleop", fps=30) -> str:
        """데이터셋 초기화 및 생성 (기존 데이터 삭제 후 재생성)"""
        with self.lock:
            self.repo_id = repo_id
            self.root_path = Path(root_dir).absolute()
            self.task_name = task_name
            self.fps = fps

            dataset_path = self.root_path / self.repo_id

            # 기존 데이터셋 폴더가 있으면 삭제 (초기화)
            if dataset_path.exists():
                print(f"[Info] 기존 데이터셋 삭제: {dataset_path}")
                shutil.rmtree(dataset_path)

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

            print(f"[Info] 데이터셋 초기화 성공 {self.repo_id}")
            return f"✅ 데이터셋 초기화 완료"

    def create_frame(self, kinect_msg, r_wrist_msg, r_follower_msg, r_leader_msg):
        """프레임 생성"""
        frame = {}

        # 이미지 디코딩
        kinect_img = decode_image(kinect_msg)
        r_wrist_img = decode_image(r_wrist_msg)

        # 조인트 데이터 변환
        r_follower_joint_data = jointState_to_nparray(r_follower_msg)
        r_leader_joint_data = jointState_to_nparray(r_leader_msg)

        # 프레임 추가
        frame['observation.images.cam_top'] = kinect_img
        frame['observation.images.cam_wrist'] = r_wrist_img
        frame['observation.state'] = r_follower_joint_data
        frame['action'] = r_leader_joint_data
        frame['task'] = self.task_name

        return frame

    def record(self, kinect_msg, wrist_msg, follower_msg, leader_msg):
        """데이터 녹화 및 상태 문자열 반환"""
        if self.dataset is None:
            return

        if self.status == "record":
            # 녹화 시작 시점 시간 기록
            if self.recording_start_time == 0:
                self.recording_start_time = time.time()

            # 프레임 생성 및 추가 (float32 변환 포함)
            follower_joint_data = jointState_to_nparray(follower_msg, self.joint_names)
            leader_joint_data = jointState_to_nparray(leader_msg, self.joint_names)

            frame = {
                'observation.images.cam_top': decode_image(kinect_msg),
                'observation.images.cam_wrist': decode_image(wrist_msg),
                'observation.state': follower_joint_data,
                'action': leader_joint_data,
                'task': self.task_name
            }

            self.dataset.add_frame(frame)
            self.num_frames += 1 # 프레임 카운트 증가

            # 경과 시간 계산
            elapsed_time = time.time() - self.recording_start_time
            print(f"🔴 녹화 중: {elapsed_time:.1f}초 ({self.num_frames} 프레임)")

        if self.status == "save":
            self.dataset.save_episode()
            self.status = ""
            # 저장 후 카운터 초기화
            self.recording_start_time = 0
            self.num_frames = 0

        if self.status == "done":
            self.dataset.finalize()
            self.status = ""






# ------------------------------------------
# 웹 인터페이스
# ------------------------------------------
class GradioWeb:
    def __init__(self, hub: SubscriberHub):
        self.hub = hub
        self.dataset_manager = LerobotDatasetManager(hub)
        self.interface = self.build_interface()

        self.ui_status = ""
        self.update_flag_ui_status = False

    def _format_joint_state(self, msg: JointState):
        """JointState 메시지를 텍스트로 포맷팅"""
        if msg is None:
            return "No Data"

        lines = []
        for name, pos in zip(msg.name, msg.position):
            pos_degrees = pos * 180.0 / 3.14159265359 # deg
            lines.append(f"{name}: {pos_degrees:.4f}°")
        return "\n".join(lines)

    def _decode_image(self, msg: CompressedImage):
        """압축된 이미지 메시지를 OpenCV 이미지로 변환"""
        if msg is None:
            return None
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if cv_image is None:
            return None
        cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        return cv_image_rgb

    def update_tick(self):
        """Timer 틱마다 호출되어 데이터를 업데이트하는 함수"""
        kinect_msg, wrist_msg, follower_msg, leader_msg = self.hub.get_latest_msg()

        # 이미지 디코딩
        kinect_img = self._decode_image(kinect_msg)
        wrist_img = self._decode_image(wrist_msg)

        # 조인트 데이터 텍스트 변환
        follower_text = self._format_joint_state(follower_msg)
        leader_text = self._format_joint_state(leader_msg)

        #
        if self.update_flag_ui_status:
            self.dataset_manager.status = self.ui_status
            self.update_flag_ui_status = False

        self.dataset_manager.record(kinect_msg, wrist_msg, follower_msg, leader_msg);


        return kinect_img, wrist_img, follower_text, leader_text

    def handle_init(self, repo_id, root_dir, task_name, fps):
        """Init 버튼 클릭 시 데이터셋 초기화 실행"""
        result = self.dataset_manager.init_dataset(repo_id, root_dir, task_name, int(fps))
        return result

    def hangle_record(self):
        """Record 버튼 클릭시 이벤트"""
        self.ui_status = "record"
        self.update_flag_ui_status = True
        print(f"[Info ] 데이터 녹화 시작")
        return f"✅ 데이터 녹화 시작"

    def handle_save_episode(self):
        """Save 버튼 클릭시 이벤트"""
        self.ui_status = "save"
        self.update_flag_ui_status = True
        print(f"[Info ] 에피소드 저장")
        return f"✅ 에피소드 저장"

    def handle_done(self):
        """Done 버튼 클릭시 이벤트"""
        self.ui_status = "done"
        self.update_flag_ui_status = True
        print(f"[Info ] 데이터셋 저장 완료")
        return f"✅ 데이터셋 저장 완료"

    def build_interface(self):
        """Gradio 인터페이스 구성 (Timer 사용)"""
        with gr.Blocks(title="Robot Teleoperation Monitor") as demo:
            gr.Markdown("# Robot Teleoperation Monitor")

            # Timer 컴포넌트 추가 (30FPS)
            timer = gr.Timer(1/30)

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
                    done_btn = gr.Button("Done", variant="primary")
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

            # Init 버튼 클릭 이벤트 연결
            init_btn.click(
                self.handle_init,
                inputs=[repo_id_input, root_dir_input, task_name_input, fps_input],
                outputs=status_output
            )

            # 녹화 버튼 클릭 이벤트 연결
            record_btn.click(
                self.hangle_record,
                inputs=None,
                outputs=status_output
            )

            # Save 버튼 클릭 이벤트 연결
            save_btn.click(
                self.handle_save_episode,
                inputs=None,
                outputs=status_output
            )

            # Done 버튼 클릭 이벤트 연결
            done_btn.click(
                self.handle_done,
                inputs=None,
                outputs=status_output
            )

            # Timer의 tick 이벤트를 update_tick 함수에 연결
            timer.tick(
                self.update_tick,
                inputs=None,
                outputs=[kinect_view, wrist_view, follower_view, leader_view]
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
