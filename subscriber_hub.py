from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, JointState

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