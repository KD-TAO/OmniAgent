import base64
import tempfile
import os
import subprocess

def video_to_audio_base64(video_path):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
        temp_audio_path = temp_audio_file.name

    try:
        command = [
            "ffmpeg",
            "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "9600",
            temp_audio_path,
            "-y",
            "-loglevel", "error"
        ]
        subprocess.run(command, check=True)

        with open(temp_audio_path, "rb") as audio_file:
            audio_base64 = base64.b64encode(audio_file.read()).decode("utf-8")

        bitrate = None
        try:
            ffprobe_cmd = [
                "ffprobe",
                "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=bit_rate",
                "-of", "default=noprint_wrappers=1:nokey=1",
                temp_audio_path,
            ]
            result = subprocess.run(ffprobe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            bitrate_str = result.stdout.strip()
            if bitrate_str.isdigit():
                bitrate = int(bitrate_str)
        except Exception:
            bitrate = None

        return audio_base64, bitrate
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

def encode_audio(video_path):
    with open(video_path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode("utf-8")
