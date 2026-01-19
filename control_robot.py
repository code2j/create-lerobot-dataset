import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration

class TrajectoryActionClient(Node):
    def __init__(self):
        super().__init__('trajectory_action_client')
        # 1. 액션 클라이언트 생성
        self._action_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/arm_controller2/follow_joint_trajectory'
        )

    def send_goal(self, joint_names, positions):
        # 액션 서버가 준비될 때까지 대기
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('액션 서버를 찾을 수 없습니다.')
            return

        # 2. 목표(Goal) 메시지 구성
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = joint_names

        # 3. 궤적 포인트(Point) 설정
        point = JointTrajectoryPoint()
        point.positions = positions
        # 시작 후 2초 안에 목표 도달하도록 설정
        point.time_from_start = Duration(sec=2, nanosec=0)

        goal_msg.trajectory.points = [point]

        self.get_logger().info(f'목표 전송 중: {positions}')

        # 4. 목표 전송 (비동기)
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('목표가 거절되었습니다.')
            return

        self.get_logger().info('목표가 수락되었습니다.')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info('최종 결과 수신 완료')
        # 종료 처리를 위해 rclpy.shutdown()을 호출하거나 추가 로직 작성

    def feedback_callback(self, feedback_msg):
        # 실제 로봇의 현재 상태를 피드백으로 받을 수 있음
        # self.get_logger().info('피드백 수신 중...')
        pass

def main(args=None):
    rclpy.init(args=args)
    action_client = TrajectoryActionClient()

    # 사용 중인 로봇의 실제 조인트 이름으로 변경해야 합니다.
    joint_names = ['right_joint1', 'right_joint2', 'right_joint3', 'right_joint4', 'right_joint5', 'right_joint6']
    # 이동시키고 싶은 각도 (라디안 단위)
    target_positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    action_client.send_goal(joint_names, target_positions)

    rclpy.spin(action_client)

if __name__ == '__main__':
    main()