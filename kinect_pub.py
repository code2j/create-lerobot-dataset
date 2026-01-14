# Ros2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

# cv2, numpy
import cv2
import numpy as np

# Kinect
from pyk4a import PyK4A, Config, ColorResolution, DepthMode, FPS, ImageFormat


class KinectPublisher(Node):
    def __init__(self, topic_name):
        super().__init__('kinect_publisher')

        # 퍼블리셔 설정
        self.topic_name = topic_name
        self.publisher_ = self.create_publisher(
            CompressedImage,
            topic_name,
            10
        )

        # Azure Kinect 설정
        self.config = Config(
            color_resolution=ColorResolution.RES_720P,  # 해상도
            depth_mode=DepthMode.OFF,                   # 뎁스 여부
            camera_fps=FPS.FPS_30,                      # FPS
            color_format=ImageFormat.COLOR_BGRA32,      # 포멧
            synchronized_images_only=False,             # 컬러와 뎁스가 모두 캡쳐된 경우에 반환할지 여부
        )

        self.k4a = PyK4A(config=self.config)
        self.k4a.start()

        print('=== Azure Kinect Publisher 시작 ===')
        print(f'토픽 : {topic_name}')
        print(f'해상도 : {self.config.color_resolution.name}')
        print(f'FPS : {self.config.camera_fps.name}')
        print('====================================')
        self.timer = self.create_timer(1.0 / 30.0, self.timer_callback) # 타이머 30Hz

    def timer_callback(self):
        """키넥트 퍼블리셔 타이머 콜백"""
        try:
            capture = self.k4a.get_capture()

            # 데이터가 없으면
            if capture.color is None:
                return

            # 이미지 변환 (BGRA -> BGR)
            color_image = capture.color[:, :, :3]

            # JPEG 압축 수행
            success, encoded_image = cv2.imencode('.jpg', color_image)
            if not success:
                return

            # 메시지 구성 및 발행
            msg = CompressedImage()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'kinect_color_frame'
            msg.format = "jpeg"
            msg.data = encoded_image.tobytes()

            self.publisher_.publish(msg)

        except Exception as e:
            self.get_logger().error(f'타이머 콜백 에러: {e}')

    def destroy_node(self):
        """키네틱 퍼블리셔 종료"""
        self.k4a.stop()
        super().destroy_node()



def main(args=None):
    # 1. ROS2 초기화
    if not rclpy.ok():
        rclpy.init()

    # 2. 퍼블리셔 생성
    node = KinectPublisher('/kinect/color/compressed')

    # 3. ROS2 스피너 시작
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    # 키넥트 데이터 퍼블리시
    main()