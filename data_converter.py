
from typing import Any, Dict, List

import cv2
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import numpy as np
from sensor_msgs.msg import CompressedImage, JointState
import torch
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint




def decode_image(msg:CompressedImage):
    """압축된 이미지 메시지를 OpenCV 이미지로 변환"""
    try:
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        return cv_image_rgb
    except Exception as e:
        print(f"이미지 디코딩 오류: {e}")
        return None