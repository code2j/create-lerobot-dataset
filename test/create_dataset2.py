import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
import torch
import shutil
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

class LeRobotV3Recorder(Node):
    def __init__(self):
        super().__init__('lerobot_v3_recorder')

        # 1. 경로 및 데이터셋 설정
        self.repo_id = "uon/triple-cam-task"
        self.root_path = Path("../outputs/dataset")
        self.dataset_path = self.root_path / self.repo_id

        # 깨끗한 생성을 위해 기존 폴더 삭제 (필요 시 주석 처리)
        if self.dataset_path.exists():
            print(f"🗑️ 기존 데이터 삭제 중: {self.dataset_path}")
            shutil.rmtree(self.dataset_path)

        # 2. LeRobot v3.0 데이터셋 생성
        # 이 시점에는 meta/info.json만 기본적으로 생성됩니다.
        self.dataset = LeRobotDataset.create(
            repo_id=self.repo_id,
            root=self.root_path,
            fps=30,
            features={
                "observation.image": {"dtype": "image", "shape": (3, 480, 640), "names": ["channels", "height", "width"]},
                "observation.state": {"dtype": "float32", "shape": (6,)},
                "action": {"dtype": "float32", "shape": (6,)},
            },
            use_videos=False  # 이미지 기반 Parquet 저장을 위해 False
        )

        # 3. ROS2 구독자 설정
        self.subscription = self.create_subscription(
            CompressedImage,
            '/kinect/color/compressed',
            self.image_callback,
            10)

        self.current_img = None
        self.is_recording = False
        self.frame_count = 0

        print(f"✅ v3.0 노드 시작. 경로: {self.dataset_path}")
        print("⌨️  [K] 녹화 시작/중지 | [Q] 최종 저장 및 종료")

    def image_callback(self, msg):
        # 이미지 디코딩 및 리사이징 (720p -> 480p)
        np_arr = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None: return

        self.current_img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)

        # UI 렌더링
        display = self.current_img.copy()
        if self.is_recording:
            cv2.circle(display, (30, 30), 15, (0, 0, 255), -1)
            cv2.putText(display, f"REC: {self.frame_count}", (60, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        cv2.imshow("LeRobot v3.0 Collector", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('k'):
            self.toggle_recording()
        elif key == ord('q'):
            self.safe_exit()

        # 데이터 프레임 추가
        if self.is_recording and self.current_img is not None:
            self.add_to_buffer()

    def toggle_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.frame_count = 0
            print("🔴 녹화 시작...")
        else:
            self.is_recording = False
            # [핵심] 이 시점에 data/ 폴더에 parquet이 생기기 시작합니다.
            print(f"💾 에피소드 저장 중 ({self.frame_count} 프레임)...")
            self.dataset.save_episode()
            print("✅ 에피소드 데이터 저장 완료.")

    def add_to_buffer(self):
        img_rgb = cv2.cvtColor(self.current_img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1)

        self.dataset.add_frame({
            "observation.image": img_tensor,
            "observation.state": torch.zeros(6),
            "action": torch.zeros(6),
            "task": "kinect capture"
        })
        self.frame_count += 1

    def safe_exit(self):
        # [핵심] finalize가 호출되어야 meta/episodes 폴더와 stats.json이 생성됩니다.
        print("🏁 데이터셋 최종 확정(Finalizing)... 이 작업은 시간이 소요됩니다.")
        self.dataset.finalize()
        print(f"🚀 모든 파일 생성 완료! 위치: {self.dataset_path}")
        rclpy.shutdown()

def main():
    rclpy.init()
    node = LeRobotV3Recorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.safe_exit()
    finally:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()