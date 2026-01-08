import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

class OmyF3MActionPublisher(Node):
    def __init__(self):
        super().__init__('omy_f3m_action_publisher')

        # 1. 퍼블리셔 셋업 (YAML의 leader:/leader/joint_trajectory 반영)
        self.joint_publishers = {
            'leader': self.create_publisher(
                JointTrajectory,
                '/leader/joint_trajectory',
                10
            )
        }

        # YAML에 정의된 관절 순서
        self.joint_names = [
            'joint1', 'joint2', 'joint3', 'joint4',
            'joint5', 'joint6', 'rh_r1_joint'
        ]

        self.get_logger().info('Omy_F3M 액션 퍼블리셔가 시작되었습니다.')

        # 테스트용: 2초마다 동작 명령 실행
        self.create_timer(10.0, self.test_publish)

    def publish_action(self, joint_msg_datas):
        """딕셔너리 기반 메시지 발행 (원본 코드 구조)"""
        for name, joint_msg in joint_msg_datas.items():
            if name in self.joint_publishers:
                self.joint_publishers[name].publish(joint_msg)
                print(f'{name}] 토픽으로 궤적을 전송했습니다.')

    def test_publish(self):
        # AI 모델에서 나온 결과값이라고 가정 (7개 관절의 목표 각도)
        # 예: 모든 관절을 0.1 라디안으로 이동
        target_positions = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

        # 2. JointTrajectory 메시지 구성
        msg = JointTrajectory()
        msg.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = target_positions
        point.time_from_start = Duration(sec=0, nanosec=100000000) # 0.1초 안에 도달 목표

        msg.points.append(point)

        # 3. publish_action 호출을 위한 딕셔너리 포맷팅
        action_data = {'leader': msg}
        self.publish_action(action_data)

def main(args=None):
    rclpy.init(args=args)
    node = OmyF3MActionPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()