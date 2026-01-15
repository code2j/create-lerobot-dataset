import gradio as gr
import cv2
import numpy as np
from pathlib import Path
import threading
import time
from io import BytesIO
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

from lerobot.datasets.lerobot_dataset import LeRobotDataset



class SubscriberHub(Node):
    def __init__(self, node_name='Subscriber_hub'):
        super().__init__(node_name)

        self.camera_topic_msgs = {}      # compressed

        self.init_sub()
        print(f'노드 시작: {node_name}')

    def init_sub(self):
        # 키넥트
        self.create_subscription(
            CompressedImage,
            '/kinect/color/compressed',
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

    def kinect_callback(self, msg: CompressedImage) -> None:
        """키넥트 카메라 토픽 콜백"""
        self.camera_topic_msgs['kinect'] = msg

    def right_wrisCam_callback(self, msg: CompressedImage) -> None:
        """오른쪽 손목 카메라 토픽 콜백"""
        self.camera_topic_msgs['right_wrist'] = msg

    def get_latest_data(self):
        """가장 최신의 이미지 데이터 리턴"""
        return self.camera_topic_msgs.copy()

    def clear_latest_data(self):
        """모든 카메라 데이터 초기화"""
        self.camera_topic_msgs = {}

    def decompress_image(self, compressed_msg: CompressedImage) -> np.ndarray:
        """압축된 이미지 메시지를 OpenCV 이미지로 변환"""
        try:
            np_arr = np.frombuffer(compressed_msg.data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            return image
        except Exception as e:
            print(f"이미지 디코딩 오류: {e}")
            return None



class Dataset_manager:
    def __init__(self):
        pass

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
                    'shape': (???)
                }
                features[f'observation.images.cam_wrist'] = {
                    'dtype': 'video',
                    'names': ['height', 'width', 'channels'],
                    'shape': (???)
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


class GradioVisualizer:
    def __init__(self, subscriber_hub: SubscriberHub):
        self.subscriber_hub = subscriber_hub
        self.update_interval = 1/30  # 100ms 업데이트 간격

        # 데이터셋 매니저
        self.dataset_manager = Dataset_manager()

    def get_kinect_image(self):
        """키넥트 이미지 반환"""
        data = self.subscriber_hub.get_latest_data()
        if 'kinect' in data:
            image = self.subscriber_hub.decompress_image(data['kinect'])
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return image
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def get_right_wrist_image(self):
        """오른쪽 손목 카메라 이미지 반환"""
        data = self.subscriber_hub.get_latest_data()
        if 'right_wrist' in data:
            image = self.subscriber_hub.decompress_image(data['right_wrist'])
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return image
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def get_both_images(self):
        """두 카메라 이미지 동시에 반환"""
        kinect_img = self.get_kinect_image()
        wrist_img = self.get_right_wrist_image()
        return kinect_img, wrist_img

    def ui_timer_callback(self):
        """UI 업데이트 콜백"""
        kinect_img, wrist_img = self.get_both_images()
        return kinect_img, wrist_img

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
                    gr.Markdown("### ⚙️ 데이터셋 설정")
                    repo_id_input = gr.Textbox(label="Repo ID", value="uon/test_dataset")
                    root_dir_input = gr.Textbox(label="Root Directory", value="./dataset")
                    task_name_input = gr.Textbox(label="Task Name", value="test_task_name")
                    # FPS 설정 추가
                    fps_input = gr.Number(label="데이터셋 FPS", value=30, precision=0)

                    init_btn = gr.Button("데이터셋 초기화 (Initialize)", variant="primary")
                    status_output = gr.Textbox(label="시스템 상태", interactive=False)


            # 데이터 초기화 버튼 이벤트
            init_btn.click(
                fn=self.dataset_manager.init_dataset,
                inputs=[repo_id_input, root_dir_input, task_name_input, fps_input],
                outputs=status_output
            )


            # 100 ms 업데이트
            timer = gr.Timer(value=self.update_interval)
            timer.tick(
                self.ui_timer_callback,
                outputs=[kinect_image, wrist_image]
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
