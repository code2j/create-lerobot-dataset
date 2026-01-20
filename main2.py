import gradio as gr
import cv2
import numpy as np
from pathlib import Path
import threading
import time
import queue
import os
import shutil
import subprocess
import signal

# 허깅페이스 오프라인 모드 ON
os.environ["HF_HUB_OFFLINE"] = "1"


# NumPy 2.x 호환성 경고 방지를 위한 설정
os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"

# ros2
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, JointState

# lerobot
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_FEATURES

from data_converter import decode_image
from subscriber_hub import SubscriberHub

class Dataset_manager:
    def __init__(self, subscriber_hub: SubscriberHub):
        self.subscriber_hub = subscriber_hub
        self.dataset = None
        self.is_recording = False
        self.running = True
        self.lock = threading.Lock()

        # 비동기 처리를 위한 큐 추가
        self.data_queue = queue.Queue()

        self.max_record_time = 0
        self.start_time = 0
        self.fps = 30

        # 에피소드 구분을 위한 ID 관리
        self.current_episode_id = 0
        self.canceled_episode_ids = set() # 취소된 에피소드 ID 목록

        self.joint_names = [
            'right_joint1', 'right_joint2', 'right_joint3',
            'right_joint4', 'right_joint5', 'right_joint6',
            'right_rh_r1_joint'
        ]

        # 1. 생산자 쓰레드: 데이터를 수집하여 큐에 넣음
        self.record_thread = threading.Thread(target=self._recording_loop, daemon=True)
        self.record_thread.start()

        # 2. 소비자 쓰레드: 큐에서 데이터를 꺼내 디코딩 및 저장
        self.consumer_thread = threading.Thread(target=self._consumer_loop, daemon=True)
        self.consumer_thread.start()

        print("[Info ] 녹화 및 소비자 쓰레드가 시작")

    def init_dataset(self, repo_id, root_dir, task_name, fps) -> str:
        """데이터셋 초기화 및 생성"""
        with self.lock:
            self.repo_id = repo_id
            self.root_path = Path(root_dir).absolute()
            self.task_name = task_name
            self.fps = fps


            dataset_path = self.root_path / self.repo_id
            info_json = dataset_path / "meta" / "info.json"

            if info_json.exists():
                self.dataset = LeRobotDataset(repo_id=self.repo_id, root=dataset_path)
                print(f"[Info ] 기존 데이터셋을 로드함")
                return "📂 기존 데이터셋을 로드했습니다."
            else:
                features = DEFAULT_FEATURES.copy()
                features[f'observation.images.cam_top'] = {
                    'dtype': 'video',
                    'names': ['height', 'width', 'channels'],
                    'shape': (720, 1280, 3)
                }
                features[f'observation.images.cam_wrist'] = {
                    'dtype': 'video',
                    'names': ['height', 'width', 'channels'],
                    'shape': (480, 848, 3)
                }
                features[f'observation.state'] = {
                    'dtype': 'float32',
                    'names': self.joint_names,
                    'shape': (7,)
                }
                features[f'action'] = {
                    'dtype': 'float32',
                    'names': self.joint_names,
                    'shape': (7,)
                }

                self.dataset = LeRobotDataset.create(
                    repo_id=self.repo_id,
                    root=dataset_path,
                    features=features,
                    use_videos=True,
                    fps=fps,
                    robot_type="omy_f3m",
                )

                print(f"[Info ] 데이터셋이 성공적으로 초기화되었습니다.")
                return "✅ 데이터셋 생성"

    def _recording_loop(self):
        """생산자: 정밀한 타이밍에 맞춰 데이터만 수집하여 큐에 삽입"""
        next_time = time.time()

        while self.running:
            if self.is_recording and self.dataset is not None:
                frame_interval = 1.0 / self.fps

                if self.max_record_time > 0:
                    if time.time() - self.start_time >= self.max_record_time:
                        self.stop_recording()
                        continue

                # 1. 데이터 수집
                raw_data = self.subscriber_hub.get_latest_msg()

                # 2. 큐에 삽입 (에피소드 ID와 함께 삽입하여 구분 가능하게 함)
                self.data_queue.put((self.current_episode_id, raw_data))

                # 3. 정밀 타이밍 제어
                next_time += frame_interval
                sleep_time = next_time - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    next_time = time.time()
            else:
                time.sleep(0.1)
                next_time = time.time()

    def _consumer_loop(self):
        """큐에서 데이터를 꺼내 작업 수행(디코딩 및 변환)"""
        while self.running:
            try:
                # 큐에서 (에피소드 ID, 데이터) 튜플을 꺼냄
                item = self.data_queue.get(timeout=0.1)
                ep_id, raw_data = item

                # 만약 이 에피소드가 취소된 것이라면 데이터를 처리하지 않고 버림
                if ep_id in self.canceled_episode_ids:
                    self.data_queue.task_done()
                    continue

                if self.dataset is not None:
                    kinect_msg, wrist_msg, follow_msg, leader_msg = raw_data

                    # 1. 이미지 처리 (디코딩)
                    kinect_img = decode_image(kinect_msg)
                    wrist_img = decode_image(wrist_msg)

                    # 2. 팔로워(State) 데이터 정렬
                    follow_map = dict(zip(follow_msg.name, follow_msg.position))
                    follower_joint_data = np.array([follow_map[name] for name in self.joint_names], dtype=np.float32)

                    # 3. 리더(Action) 데이터 정렬
                    leader_map = dict(zip(leader_msg.name, leader_msg.position))
                    leader_joint_data = np.array([leader_map[name] for name in self.joint_names], dtype=np.float32)

                    # 4. 데이터셋 추가
                    if kinect_img is not None and wrist_img is not None:
                        with self.lock:
                            self.dataset.add_frame({
                                'observation.images.cam_top': kinect_img,
                                'observation.images.cam_wrist': wrist_img,
                                'observation.state': follower_joint_data,
                                'action': leader_joint_data,
                                'task': self.task_name
                            })

                self.data_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Error] 소비자 루프 오류: {e}")

    def start_recording(self, max_time=0):
        if self.dataset is None:
            return "❌ 오류: 데이터셋이 초기화되지 않았습니다."

        if self.is_recording:
            return "⚠️ 시스템: 이미 녹화 중입니다."

        self.max_record_time = max_time
        self.start_time = time.time()

        # 새로운 에피소드 시작 시 ID 증가
        self.current_episode_id += 1
        self.is_recording = True

        msg = "녹화 중..."
        print(f"[Info ] {msg} (Episode ID: {self.current_episode_id})")
        return msg

    def stop_recording(self):
        if not self.is_recording:
            return "⚠️ 시스템: 현재 녹화 중이 아닙니다."

        self.is_recording = False

        # 현재 에피소드 ID를 고정하여 저장 쓰레드에 전달
        finished_ep_id = self.current_episode_id

        # 백그라운드에서 큐가 비워지면 저장하도록 함
        threading.Thread(target=self._wait_and_save, args=(finished_ep_id,), daemon=True).start()

        msg = "백그라운드 저장 중..."
        print(f"[Info ] {msg}")
        return msg

    def cancel_recording(self):
        """현재 녹화를 취소하고 해당 에피소드 데이터를 무시하도록 설정"""
        if not self.is_recording:
            return "⚠️ 시스템: 현재 녹화 중이 아닙니다."

        self.is_recording = False

        # 현재 에피소드 ID를 취소 목록에 추가 (소비자 루프에서 이 ID를 가진 데이터는 버려짐)
        self.canceled_episode_ids.add(self.current_episode_id)

        msg = "현재 에피소드 녹화 취소됨"
        print(f"[Info ] {msg} (ID: {self.current_episode_id})")
        return msg

    def _wait_and_save(self, ep_id):
        """특정 에피소드의 데이터가 큐에서 모두 처리될 때까지 기다린 후 저장"""
        print(f"[Info ] 에피소드 {ep_id} 데이터를 처리 중입니다...")

        # 큐가 완전히 비워질 때까지 기다리는 대신,
        # 소비자 루프가 데이터를 처리하는 속도를 고려하여 큐를 모니터링하거나
        # 간단하게 전체 큐가 비워질 때까지 기다림 (이전 에피소드들이 순차적으로 쌓이므로)
        self.data_queue.join()

        with self.lock:
            # 취소된 에피소드가 아닐 때만 저장
            if self.dataset is not None and ep_id not in self.canceled_episode_ids:
                try:
                    self.dataset.save_episode()
                    print(f"[Info ] 에피소드 {ep_id} 저장 완료")
                except Exception as e:
                    print(f"[Error] 에피소드 저장 중 오류: {e}")

            # 처리가 끝난 ID는 메모리 관리를 위해 제거 (선택 사항)
            if ep_id in self.canceled_episode_ids:
                self.canceled_episode_ids.remove(ep_id)

    def finalize_dataset(self):
        """데이터 수집 완료 및 데이터셋 최종화"""
        if self.dataset is None:
            return "❌ 오류: 데이터셋이 초기화되지 않았습니다."

        if self.is_recording:
            return "❌ 오류: 녹화 중에는 데이터셋을 완료할 수 없습니다."

        if not self.data_queue.empty():
            return "⏳ 시스템: 아직 처리 중인 데이터가 있습니다. 잠시 후 다시 시도해주세요."

        with self.lock:
            try:
                self.dataset.finalize()
                # 최종화 후 데이터셋 객체를 None으로 설정하여 키 리스너가 동작하지 않게 함
                self.dataset = None
                msg = "✅ 데이터 수집 완료 및 데이터셋 최종화 성공"
                print(msg)
                return msg
            except Exception as e:
                msg = f"❌ 시스템: 데이터셋 최종화 중 오류 발생: {e}"
                print(msg)
                return msg

    def close(self):
        self.running = False
        self.record_thread.join()
        self.consumer_thread.join()
        print("시스템: 모든 쓰레드가 종료되었습니다.")

from pynput import keyboard
class GradioVisualizer:
    def __init__(self, subscriber_hub: SubscriberHub):
        self.subscriber_hub = subscriber_hub
        self.update_interval = 1/30
        self.dataset_manager = Dataset_manager(self.subscriber_hub)

        # 채터링 방지 및 키 상태 관리를 위한 변수
        self.last_key_time = 0
        self.chatter_threshold = 0.2
        self.right_pressed = False
        self.left_pressed = False

        # 상태 메시지 관리
        self.current_status = "✅ 대기 중"

        # pynput 리스너 설정
        self.listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self.listener.start()

    def _on_press(self, key):
        try:
            # 데이터셋이 초기화된 상태에서만 키 입력 처리
            # finalize_dataset 호출 시 self.dataset_manager.dataset이 None이 되므로 리스너가 비활성화됨
            if self.dataset_manager.dataset is None:
                return

            current_time = time.time()
            if current_time - self.last_key_time < self.chatter_threshold:
                return

            # 오른쪽 방향키: 녹화 토글
            if key == keyboard.Key.right:
                if not self.right_pressed:
                    self.right_pressed = True
                    self.last_key_time = current_time
                    self._toggle_recording()
            # 왼쪽 방향키: 재녹화
            elif key == keyboard.Key.left:
                if not self.left_pressed:
                    self.left_pressed = True
                    self.last_key_time = current_time
                    self._re_record()
        except Exception as e:
            print(f"[Error] Key press handling error: {e}")

    def _on_release(self, key):
        try:
            if key == keyboard.Key.right:
                self.right_pressed = False
            elif key == keyboard.Key.left:
                self.left_pressed = False
        except Exception as e:
            pass

    def _toggle_recording(self):
        """녹화 상태를 토글"""
        if self.dataset_manager.is_recording:
            self.current_status = self.dataset_manager.stop_recording()
        else:
            self.current_status = self.dataset_manager.start_recording(max_time=0)
        # print(f"[Key Event] '오른쪽 방향키' 입력: {self.current_status}")

    def _re_record(self):
        """현재 녹화를 취소하고 즉시 다시 시작"""
        if self.dataset_manager.is_recording:
            self.dataset_manager.cancel_recording()
            self.current_status = self.dataset_manager.start_recording(max_time=0)
            # print(f"[Key Event] '왼쪽 방향키' 입력: 재녹화 시작 (이전 데이터 보호됨)")
        else:
            self.current_status = self.dataset_manager.start_recording(max_time=0)
            # print(f"[Key Event] '왼쪽 방향키' 입력: 녹화 시작")

    def ui_timer_callback(self):
        (k_msg, w_msg, f_joint, l_joint) = self.subscriber_hub.get_latest_msg()
        k_img = decode_image(k_msg)
        w_img = decode_image(w_msg)

        desired_names = ['right_joint1', 'right_joint2', 'right_joint3', 'right_joint4', 'right_joint5', 'right_joint6', 'right_rh_r1_joint']

        follower_text = "N/A"
        if f_joint is not None:
            f_vals = [np.rad2deg(f_joint.position[f_joint.name.index(n)]) if n in f_joint.name else np.nan for n in desired_names]
            follower_text = f"J1: {f_vals[0]:.1f} J2: {f_vals[1]:.1f} J3: {f_vals[2]:.1f} J4: {f_vals[3]:.1f} J5: {f_vals[4]:.1f} J6: {f_vals[5]:.1f} G: {f_vals[6]:.1f}"

        leader_text = "N/A"
        if l_joint is not None:
            l_vals = [np.rad2deg(l_joint.position[l_joint.name.index(n)]) if n in l_joint.name else np.nan for n in desired_names]
            leader_text = f"J1: {l_vals[0]:.1f} J2: {l_vals[1]:.1f} J3: {l_vals[2]:.1f} J4: {l_vals[3]:.1f} J5: {l_vals[4]:.1f} J6: {l_vals[5]:.1f} G: {l_vals[6]:.1f}"

        display_status = self.current_status
        q_size = self.dataset_manager.data_queue.qsize()

        if self.dataset_manager.is_recording:
            elapsed = time.time() - self.dataset_manager.start_time
            display_status = f"{self.current_status} ({elapsed:.1f}s) | 큐: {q_size}"
        elif q_size > 0:
            display_status = f"⏳ 저장 중... (남은 작업: {q_size})"
        elif "저장 중" in self.current_status and q_size == 0:
            self.current_status = "✅ 대기 중 (저장 완료)"
            display_status = self.current_status

        return k_img, w_img, follower_text, leader_text, display_status

    def handle_init(self, repo_id, root_dir, task_name, fps):
        res = self.dataset_manager.init_dataset(repo_id, root_dir, task_name, fps)
        self.current_status = res
        return res

    def handle_record(self, max_time):
        res = self.dataset_manager.start_recording(max_time)
        self.current_status = res
        return res

    def handle_re_record(self):
        """재녹화 버튼 핸들러"""
        if self.dataset_manager.is_recording:
            self.dataset_manager.cancel_recording()
        res = self.dataset_manager.start_recording(max_time=0)
        self.current_status = res
        return res

    def handle_stop(self):
        res = self.dataset_manager.stop_recording()
        self.current_status = res
        return res

    def handle_finalize(self):
        res = self.dataset_manager.finalize_dataset()
        self.current_status = res
        return res

    def create_interface(self):
        default_root_dir = os.path.join(os.getcwd(), "dataset")

        with gr.Blocks(title="Robot Data Collector") as demo:
            gr.Markdown("# 🤖 Robot Data Collector")

            with gr.Row():
                kinect_image = gr.Image(label="Kinect", type="numpy")
                wrist_image = gr.Image(label="Wrist", type="numpy")

            with gr.Row():
                follower_joint_output = gr.Textbox(label="Follower Arm Joints")
                leader_joint_output = gr.Textbox(label="Leader Arm Joints")

            with gr.Row():
                repo_id_input = gr.Textbox(label="Repo ID", value="test_dataset")
                root_dir_input = gr.Textbox(label="Root Path", value=default_root_dir)
                task_name_input = gr.Textbox(label="Task", value="test_task")
                fps_input = gr.Number(label="FPS", value=30)
                max_time_input = gr.Number(label="Max Time", value=0)

            init_btn = gr.Button("Initialize")
            status_output = gr.Textbox(label="Status", value=self.current_status)

            with gr.Row():
                record_btn = gr.Button("Record (Right Arrow)", variant="primary")
                re_record_btn = gr.Button("Re-record (Left Arrow)", variant="secondary")
                stop_btn = gr.Button("Stop", variant="stop")
                finalize_btn = gr.Button("데이터 수집 완료 (Finalize)", variant="secondary")

            init_btn.click(self.handle_init, [repo_id_input, root_dir_input, task_name_input, fps_input], status_output)
            record_btn.click(self.handle_record, [max_time_input], status_output)
            re_record_btn.click(self.handle_re_record, outputs=status_output)
            stop_btn.click(self.handle_stop, outputs=status_output)
            finalize_btn.click(self.handle_finalize, outputs=status_output)

            timer = gr.Timer(value=self.update_interval)
            timer.tick(
                self.ui_timer_callback,
                outputs=[
                    kinect_image, wrist_image, follower_joint_output, leader_joint_output,
                    status_output
                ]
            )

        return demo

    def launch(self):
        demo = self.create_interface()
        demo.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    import rclpy
    rclpy.init()
    hub = SubscriberHub()
    threading.Thread(target=lambda: rclpy.spin(hub), daemon=True).start()

    visualizer = GradioVisualizer(hub)
    try:
        visualizer.launch()
    except KeyboardInterrupt:
        visualizer.dataset_manager.close()
        rclpy.shutdown()
