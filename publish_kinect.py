import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np
from pyk4a import PyK4A, Config, ColorResolution, DepthMode, FPS, ImageFormat

class KinectPublisher(Node):
    def __init__(self, topic_name):
        super().__init__('kinect_publisher')

        # 1. 퍼블리셔 설정 (/kinect/color/compressed)
        self.publisher_ = self.create_publisher(
            CompressedImage,
            topic_name,
            10
        )

        # 2. Azure Kinect 설정
        self.config = Config(
            color_resolution=ColorResolution.RES_720P,
            depth_mode=DepthMode.OFF,
            camera_fps=FPS.FPS_30,
            color_format=ImageFormat.COLOR_BGRA32,
            synchronized_images_only=False,
        )

        self.k4a = PyK4A(config=self.config)
        self.k4a.start()

        self.bridge = CvBridge()

        # 타이머 설정 (30 FPS에 맞춰 약 0.033초마다 실행)
        self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)
        self.get_logger().info('Kinect Publisher Node has been started.')

    def timer_callback(self):
        try:
            capture = self.k4a.get_capture()

            if capture.color is not None:
                # BGRA -> BGR 변환
                color_image = capture.color[:, :, :3]

                # OpenCV 이미지를 CompressedImage 메시지로 변환 (jpeg 압축)
                msg = CompressedImage()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = 'kinect_color_frame'
                msg.format = "jpeg"

                # JPEG 압축 수행
                success, encoded_image = cv2.imencode('.jpg', color_image)
                if success:
                    msg.data = encoded_image.tobytes()
                    self.publisher_.publish(msg)
                    # self.get_logger().info('Publishing compressed image')

        except Exception as e:
            self.get_logger().error(f'Error in timer_callback: {e}')

    def destroy_node(self):
        self.k4a.stop()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = KinectPublisher('/kinect/color/compressed')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
