import gradio as gr
import cv2
import numpy as np
from pathlib import Path
import threading
import time
from io import BytesIO

# ros2
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, JointState


# lerobot
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_FEATURES

from data_converter import decode_image

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

    def get_latest_data(self):
        """가장 최신의 데이터 리턴"""
        return (self.kinect_topic_msg,
                self.right_wristCame_topic_msg,
                self.right_follower_topic_msg,
                self.right_leader_topic_msg)

    def clear_latest_data(self):
        """모든 토픽 데이터 초기화"""
        self.kinect_topic_msg = None
        self.right_wristCame_topic_msg = None
        self.right_follower_topic_msg = None
        self.right_leader_topic_msg = None





class Dataset_manager:
    def __init__(self):
        self.dataset = None
        self.is_recording = False  # 실제 녹화 여부 플래그
        self.running = True        # 쓰레드 유지 플래그

        # 녹화용 쓰레드 설정 및 시작
        self.record_thread = threading.Thread(target=self._recording_loop, daemon=True)
        self.record_thread.start()
        print("시스템: 녹화 쓰레드가 시작되었습니다.")

    def init_dataset(self, repo_id, root_dir, task_name, fps) -> bool:
        """데이터셋 초기화 및 생성"""
        try:
            self.repo_id = repo_id
            self.root_path = Path(root_dir).absolute()
            self.task_name = task_name
            dataset_path = self.root_path / self.repo_id
            info_json = dataset_path / "meta" / "info.json"

            # TODO: 임시 조인트 이름
            joint_names = [
                'right_joint1', 'right_joint2', 'right_joint3',
                'right_joint4', 'right_joint5', 'right_joint6',
                'right_rh_r1_joint'
            ]

            if info_json.exists():
                # 이미 데이터셋이 존재하는 경우에 기존 데이터셋을 가져옴
                self.dataset = LeRobotDataset(repo_id=self.repo_id, root=dataset_path)
            else:
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
                    'names': joint_names,
                    'shape': (7,)
                }
                features[f'action'] = {
                    'dtype': 'float32',
                    'names': joint_names,
                    'shape': (7,)
                }


                # 새로운 데이터셋을 생성
                self.dataset = LeRobotDataset.create(
                    repo_id=self.repo_id,
                    root=dataset_path,
                    fps=fps,
                    robot_type= "omy_f3m",
                    features={
                        "timestamp": {"dtype": "float32", "shape": (1,), "names": None, "fps": fps},
                        "frame_index": {"dtype": "int64", "shape": (1,), "names": None, "fps": fps},
                        "episode_index": {"dtype": "int64", "shape": (1,), "names": None, "fps": fps},
                        "index": {"dtype": "int64", "shape": (1,), "names": None, "fps": fps},
                        "task_index": {"dtype": "int64", "shape": (1,), "names": None, "fps": fps},
                        # 키넥트 카메라
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
                        # 오른쪽 손목 카메라
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
                        # 로봇의 현재 상태
                        "observation.state": {
                            "dtype": "float32",
                            "shape": (7,),
                            "names": joint_names
                        },
                        # 액션
                        "action": {
                            "dtype": "float32",
                            "shape": (7,),
                            "names": joint_names
                        },
                    },
                    use_videos=True,
                )
            return True # 데이터셋 생성 성공
        except Exception as e:
            print(f"데이터셋 초기화 오류: {e}")
            return False # 데이터셋 생성 실패


    def _recording_loop(self):
        """별도 쓰레드에서 무한히 돌아가는 루프"""
        while self.running:
            if self.is_recording and self.dataset is not None:
                # 여기에 실제 데이터 수집 및 저장 로직이 들어갑니다.
                self.record()

                # FPS에 맞게 대기 (예: 30fps라면 약 0.033초)
                time.sleep(1.0 / self.fps if hasattr(self, 'fps') else 0.1)
            else:
                # 녹화 중이 아닐 때는 CPU 점유율을 낮추기 위해 짧게 대기
                time.sleep(0.1)

    def record(self):
        """실제 프레임을 캡처하고 데이터셋에 추가하는 로직"""
        # 이 부분에 카메라 프레임 읽기, 로봇 상태 읽기, self.dataset.add_frame() 등의 로직을 구현합니다.
        # 예: frame = camera.read(), state = robot.get_state()
        # self.dataset.add_frame({"observation.image": frame, "action": state, ...})
        print(f"[{time.time():.2f}] 데이터를 기록 중...")

    def start_recording(self):
        if self.dataset is None:
            print("오류: 데이터셋이 초기화되지 않았습니다. init_dataset을 먼저 호출하세요.")
            return
        self.is_recording = True
        print("시스템: 녹화를 시작합니다.")

    def stop_recording(self):
        self.is_recording = False
        print("시스템: 녹화를 중단합니다.")

    def close(self):
        """프로그램 종료 시 쓰레드를 안전하게 종료"""
        self.running = False
        self.record_thread.join()
        print("시스템: 녹화 쓰레드가 종료되었습니다.")




class GradioVisualizer:
    def __init__(self, subscriber_hub: SubscriberHub):
        self.subscriber_hub = subscriber_hub
        self.update_interval = 1/30  # 30hz

        # 데이터셋 매니저
        self.dataset_manager = Dataset_manager()

    def get_latest_data(self):
        """가장 최신의 데이터 리턴"""
        # 가장 최신 데이터 가져오기
        (kinect_compressed_img,
         right_wrist_compressed_img,
         right_follower_joint,
         right_leader_joint) = subscriber_hub.get_latest_data()

        # 이미지 디코딩
        kinect_img = decode_image(kinect_compressed_img)
        right_wrist_img = decode_image(right_wrist_compressed_img)

        return kinect_img, right_wrist_img, right_follower_joint, right_leader_joint

    def ui_timer_callback(self):
        """UI 업데이트 콜백"""
        kinect_img, right_wrist_img, follower_joint, _ = self.get_latest_data()

        # Joint 데이터 추출
        follower_joint_text = "N/A"
        if follower_joint is not None:
            joint_positions = follower_joint.position
            joint_names = follower_joint.name

            # 원하는 조인트 이름 순서
            desired_joint_names = [
                'right_joint1', 'right_joint2', 'right_joint3',
                'right_joint4', 'right_joint5', 'right_joint6',
                'right_rh_r1_joint'
            ]

            # 원하는 순서대로 조인트 값 가져오기
            ordered_joint_values = []
            for name in desired_joint_names:
                try:
                    idx = joint_names.index(name)
                    ordered_joint_values.append(np.rad2deg(joint_positions[idx]))
                except ValueError:
                    ordered_joint_values.append(np.nan) # 해당 조인트가 없으면 NaN

            follower_joint_text = (
                f"J1: {ordered_joint_values[0]:.2f}°  J2: {ordered_joint_values[1]:.2f}°  J3: {ordered_joint_values[2]:.2f}° "
                f"J4: {ordered_joint_values[3]:.2f}°  J5: {ordered_joint_values[4]:.2f}°  J6: {ordered_joint_values[5]:.2f}°  "
                f"Gripper: {ordered_joint_values[6]:.2f}°"
            )

        # Leader joint text
        leader_joint_text = "N/A"
        if self.subscriber_hub.right_leader_topic_msg is not None:
            leader_joint_positions = self.subscriber_hub.right_leader_topic_msg.position
            leader_joint_names = self.subscriber_hub.right_leader_topic_msg.name

            ordered_leader_values = []
            for name in desired_joint_names:
                try:
                    idx = leader_joint_names.index(name)
                    ordered_leader_values.append(np.rad2deg(leader_joint_positions[idx]))
                except ValueError:
                    ordered_leader_values.append(np.nan)

            leader_joint_text = (
                f"J1: {ordered_leader_values[0]:.2f}°  J2: {ordered_leader_values[1]:.2f}°  J3: {ordered_leader_values[2]:.2f}° "
                f"J4: {ordered_leader_values[3]:.2f}°  J5: {ordered_leader_values[4]:.2f}°  J6: {ordered_leader_values[5]:.2f}°  "
                f"Gripper: {ordered_leader_values[6]:.2f}°"
            )
        return kinect_img, right_wrist_img, follower_joint_text, leader_joint_text

    def create_interface(self):
        """Gradio 인터페이스 생성"""
        with gr.Blocks(title="Test ") as demo:

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 키넥트 카메라")
                    kinect_image = gr.Image(
                        label="Kinect Camera",
                        type="numpy",
                        interactive=False
                    )

                with gr.Column():
                    gr.Markdown("### 오른쪽 손목 카메라")
                    wrist_image = gr.Image(
                        label="Right Wrist Camera",
                        type="numpy",
                        interactive=False
                    )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 로봇 조인트 상태 (Degrees)")
                    follower_joint_output
                    leader_joint_output


        return kinect_img, right_wrist_img, follower_joint_text, leader_joint_output

    def create_interface(self):
        """Gradio 인터페이스 생성"""
        with gr.Blocks(title="Test ") as demo:

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 키넥트 카메라")
                    kinect_image = gr.Image(
                        label="Kinect Camera",
                        type="numpy",
                        interactive=False
                    )

                with gr.Column():
                    gr.Markdown("### 오른쪽 손목 카메라")
                    wrist_image = gr.Image(
                        label="Right Wrist Camera",
                        type="numpy",
                        interactive=False
                    )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 로봇 조인트 상태 (Degrees)")
                    follower_joint_output = gr.Textbox(label="Follower Arm Joints", interactive=False)
                    leader_joint_output = gr.Textbox(label="Leader Arm Joints", interactive=False)

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### ⚙️ 데이터셋 설정")
                    repo_id_input = gr.Textbox(label="Repo ID", value="uon/test_dataset")
                    root_dir_input = gr.Textbox(label="Root Directory", value="./dataset")
                    task_name_input = gr.Textbox(label="Task Name", value="test_task_name")
                    # FPS 설정 추가
                    fps_input = gr.Number(label="데이터셋 FPS", value=30, precision=0)

                    init_btn = gr.Button("데이터셋 초기화 (Initialize)", variant="primary")
                    status_output = gr.Textbox(label="시스템 상태", interactive=False)

            with gr.Row():
                record_btn = gr.Button("데이터셋 녹화 (Record)", variant="primary")


            # 데이터 초기화 버튼 이벤트
            init_btn.click(
                fn=self.dataset_manager.init_dataset,
                inputs=[repo_id_input, root_dir_input, task_name_input, fps_input],
                outputs=status_output
            )

            # 녹화 시작 버튼
            record_btn.click(
                fn=self.dataset_manager.start_recording,
                inputs=[],
                outputs=[]
            )

            # 100 ms 업데이트
            timer = gr.Timer(value=self.update_interval)
            timer.tick(
                self.ui_timer_callback,
                outputs=[kinect_image, wrist_image, follower_joint_output, leader_joint_output]
            )

        return demo

    def launch(self, share=False, server_name="0.0.0.0", server_port=7860):
        """Gradio 앱 실행"""
        demo = self.create_interface()
        demo.launch(
            share=share,
            server_name=server_name,
            server_port=server_port,
            show_error=True
        )


# 사용 예시
if __name__ == "__main__":
    import rclpy

    rclpy.init()

    # SubscriberHub 노드 생성
    subscriber_hub = SubscriberHub()

    # Gradio 시각화기 생성
    visualizer = GradioVisualizer(subscriber_hub)

    # 스핀 스레드 시작 (ROS 2 메시지 수신을 위해)
    def spin_node():
        rclpy.spin(subscriber_hub)

    spin_thread = threading.Thread(target=spin_node, daemon=True)
    spin_thread.start()

    # Gradio 앱 실행
    visualizer.launch(share=False, server_port=7860)
