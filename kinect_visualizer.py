import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
import gradio as gr
import threading

class KinectVisualizerNode(Node):
    def __init__(self, topic_name):
        super().__init__('gradio_timer_node')
        self.subscription = self.create_subscription(
            CompressedImage,
            topic_name,
            self.listener_callback,
            10
        )
        self.latest_frame = None

    def listener_callback(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if cv_image is not None:
                self.latest_frame = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB) # 이미지 변환
        except Exception as e:
            self.get_logger().error(f'이미지 디코딩 에러: {e}')

# 전역 노드 변수
ros_node = None

def get_frame():
    """타이머가 호출할 함수: 최신 프레임을 반환"""
    if ros_node is not None and ros_node.latest_frame is not None:
        return ros_node.latest_frame
    return None

def run_ros():
    global ros_node
    rclpy.init()
    ros_node = KinectVisualizerNode('/kinect/color/compressed')
    rclpy.spin(ros_node)
    rclpy.shutdown()

# 메인 실행부
def main():
    # ROS 2 스레드 시작
    threading.Thread(target=run_ros, daemon=True).start()

    with gr.Blocks() as demo:
        gr.Markdown(f"## Kinect Visualizer")

        with gr.Row():
            image_output = gr.Image(label="Kinect Color")

        # 1. 타이머 설정
        timer = gr.Timer(1/60) # 60 FPS

        # 2. 타이머의 tick 이벤트를 이미지 업데이트 함수에 연결
        timer.tick(fn=get_frame, outputs=image_output)

    demo.launch()

if __name__ == '__main__':
    main()