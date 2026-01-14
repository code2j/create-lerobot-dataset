import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
import threading
import gradio as gr
import argparse

class SubscriberHub(Node):
    def __init__(self, node_name='Subscriber_hub'):
        super().__init__(node_name)


        self.create_subscription(
            CompressedImage,            # 메세지 타입
            '/kinect/color/compressed', # 키넥트 토픽 이름
            self.kinect_callback,        # 키넥트 콜백
            10
        )
        print(f'노드 시작: {node_name}')

    def kinect_callback(self, msg):
        """이미지 수신 시 호출"""
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img is not None:
                # BGR(OpenCV) -> RGB 변환
                self.current_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"이미지 처리 에러: {e}")



def run(hub: SubscriberHub):
    # ROS 2 spin을 별도 스레드에서 실행하여 Gradio와 충돌 방지
    ros_thread = threading.Thread(target=rclpy.spin, args=(hub,), daemon=True)
    ros_thread.start()

    # Gradio UI 구성
    with gr.Blocks() as demo:
        gr.Markdown("## Kinect")
        image_view = gr.Image(label="Kinect View")

        # 타이머 설정
        timer = gr.Timer(value=1/60) # 60 hz

        # hub의 최신 프레임을 반환
        def get_frame():
            if hub.current_frame is not None:
                pass
            else:
                # 초기 프레임이 없을 경우 빈 이미지 반환
                return np.zeros((480, 640, 3), dtype=np.uint8) # 적절한 크기로 조정

            return hub.current_frame

        # image_view 업데이트
        timer.tick(fn=get_frame, outputs=image_view)

    try:
        demo.launch(server_name="0.0.0.0", server_port=7861)
    except KeyboardInterrupt:
        pass
    finally:
        hub.destroy_node()
        rclpy.shutdown()

def run_without_visualize(hub: SubscriberHub):
    try:
        rclpy.spin(hub)
    except KeyboardInterrupt:
        pass
    finally:
        hub.destroy_node()
        rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vis', action='store_true', default=False)
    args, unknown = parser.parse_known_args()

    if not rclpy.ok():
        rclpy.init(args=unknown)

    hub = SubscriberHub()

    if args.vis:
        print("시각화 모드: ON")
        run(hub)

    else:
        print("시각화 모드: OFF")
        run_without_visualize(hub)




if __name__ == '__main__':
    main()

