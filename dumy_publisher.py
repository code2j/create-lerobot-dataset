import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, JointState
import cv2
import numpy as np

class DummyDataPublisher(Node):
    def __init__(self):
        super().__init__('dummy_data_publisher')

        # 1. 조인트 설정
        self.joint_names = [
            'right_joint1', 'right_joint2', 'right_joint3',
            'right_joint4', 'right_joint5', 'right_joint6',
            'right_rh_r1_joint'
        ]
        self.num_joints = len(self.joint_names)

        # 2. 퍼블리셔 설정
        self.pub_cam_top = self.create_publisher(CompressedImage, '/right/camera/cam_top/color/image_rect_raw/compressed', 10)
        self.pub_cam_wrist = self.create_publisher(CompressedImage, '/right/camera/cam_wrist/color/image_rect_raw/compressed', 10)
        self.pub_joint_robot = self.create_publisher(JointState, '/right/joint_states', 10)
        self.pub_joint_leader = self.create_publisher(JointState, '/right_robot/leader/joint_states', 10)

        # 3. 검은색 이미지 데이터 미리 생성 (한 번만 수행)
        self.top_image_data = self._generate_black_compressed_image(720, 1280)
        self.wrist_image_data = self._generate_black_compressed_image(480, 848)

        # 4. 타이머 설정 (30Hz)
        self.publish_freq = 30.0
        self.timer = self.create_timer(1.0 / self.publish_freq, self.timer_callback)

        self.get_logger().info(f'Simple Dummy Publisher (Black Screen & Zero Joints) started at {self.publish_freq}Hz.')

    def _generate_black_compressed_image(self, h, w):
        """검은색 이미지를 생성하고 JPEG로 미리 인코딩"""
        black_img = np.zeros((h, w, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', black_img)
        return buffer.tobytes()

    def timer_callback(self):
        now = self.get_clock().now().to_msg()

        # --- 이미지 발행 ---
        top_msg = CompressedImage()
        top_msg.header.stamp = now
        top_msg.format = "jpeg"
        top_msg.data = self.top_image_data
        self.pub_cam_top.publish(top_msg)

        wrist_msg = CompressedImage()
        wrist_msg.header.stamp = now
        wrist_msg.format = "jpeg"
        wrist_msg.data = self.wrist_image_data
        self.pub_cam_wrist.publish(wrist_msg)

        # --- 조인트 데이터 발행 (전부 0) ---
        joint_msg = JointState()
        joint_msg.header.stamp = now
        joint_msg.name = self.joint_names
        joint_msg.position = [0.0] * self.num_joints
        joint_msg.velocity = [0.0] * self.num_joints
        joint_msg.effort = [0.0] * self.num_joints

        self.pub_joint_robot.publish(joint_msg)
        self.pub_joint_leader.publish(joint_msg)

def main(args=None):
    rclpy.init(args=args)
    node = DummyDataPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()