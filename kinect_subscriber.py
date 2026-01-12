import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

class KinnectSubscriber(Node):
    def __init__(self, topic_name):
        if not rclpy.ok():
            rclpy.init()

        super().__init__('kinect_subscriber')
        self.subscription = self.create_subscription(
            CompressedImage,
            topic_name,
            self.listener_callback,
            10
        )

    def listener_callback(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if cv_image is not None:
                self.latest_frame = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB) # 이미지 변환
        except Exception as e:
            self.get_logger().error(f'이미지 디코딩 에러: {e}')

    def get_frame(self):
        # 구현
        pass

if __name__ == '__main__':
    sub = KinnectSubscriber('/kinect/color/compressed')
    while True:
        color = sub.get_frame()

