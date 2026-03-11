from typing import Dict, Any
import os
import cv2
from langchain_core.tools import tool


@tool
def video_metadata(video_path: str) -> Dict[str, Any]:
    """
    Retrieve basic metadata for a video file, such as duration, FPS, frame count,
    and resolution. This is useful for validating time ranges before calling
    clip-based tools (e.g., video_clip_qa).

    This tool:
    - Opens the video file using OpenCV.
    - Reads:
      * total number of frames,
      * frames per second (FPS),
      * duration in seconds,
      * frame width and height in pixels.

    Typical usage:
    - Before calling video_clip_qa, you can call video_metadata to:
      * check the total duration,
      * ensure that a proposed time_range = (start, end) uses valid
        positive integer seconds, and
      * verify that 0 <= start < end <= duration_seconds.

    Args:
        video_path: Path or identifier of the video file on disk.

    Returns:
        A dictionary with:
            - "duration_seconds": float, total duration of the video in seconds.
            - "frame_count": int, total number of frames.
            - "fps": float, frames per second reported by the container.
            - "width": int, frame width in pixels.
            - "height": int, frame height in pixels.

    Raises:
        FileNotFoundError: if the video file does not exist.
        RuntimeError: if the video cannot be opened or has invalid metadata.
    """
    abs_path = os.path.abspath(video_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Video file not found: {abs_path}")

    cap = cv2.VideoCapture(abs_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {abs_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0

    cap.release()

    if frame_count <= 0 or fps <= 0.0:
        raise RuntimeError(
            f"Video has invalid metadata (frame_count={frame_count}, fps={fps})."
        )

    import math
    duration_seconds = math.floor(frame_count / fps)

    return {
        "duration_seconds": duration_seconds,
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
    }
