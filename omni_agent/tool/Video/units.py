from typing import Dict, Optional, List, Tuple
import os
import base64
import numpy as np
import cv2  # pip install opencv-python
from moviepy import VideoFileClip

def cut_video(in_path, out_path, t_start, t_end):
    with VideoFileClip(in_path) as video:
        sub = video.subclipped(t_start, t_end)
        sub.write_videofile(
            out_path,
            codec="libx264", 
            audio=False,       
            logger=None       
        )