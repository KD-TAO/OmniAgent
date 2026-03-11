import os
import time
import requests
from pathlib import Path
from datetime import datetime, timedelta
from requests.exceptions import Timeout, ConnectionError, RequestException

import subprocess
import os

DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3

def cut_audio(in_path, out_path, t_start, t_end):
    def parse_time(t):
        if isinstance(t, tuple):
            return t[0] * 3600 + t[1] * 60 + t[2]
        return float(t)

    start_sec = parse_time(t_start)
    end_sec = parse_time(t_end)
    duration = end_sec - start_sec

    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(start_sec),
        "-i", in_path,
        "-t", str(duration),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        out_path
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg failed: {e.stderr.decode()}")


def get_upload_policy(api_key, model_name):
    url = "https://dashscope.aliyuncs.com/api/v1/uploads"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    params = {
        "action": "getPolicy",
        "model": model_name
    }

    last_exc = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(
                url,
                headers=headers,
                params=params,
                timeout=DEFAULT_TIMEOUT,
            )

            if response.status_code == 200:
                return response.json()["data"]

            if response.status_code in (429, 500, 502, 503, 504):
                last_exc = Exception(
                    f"Failed to get upload policy (attempt {attempt}): "
                    f"{response.status_code} {response.text}"
                )
                if attempt < MAX_RETRIES:
                    time.sleep(2 ** (attempt - 1))
                    continue
                else:
                    raise last_exc

            raise Exception(
                f"Failed to get upload policy: {response.status_code} {response.text}"
            )

        except (Timeout, ConnectionError) as e:
            last_exc = e
            if attempt < MAX_RETRIES:
                time.sleep(2 ** (attempt - 1))
                continue
            else:
                raise Exception(
                    f"Failed to get upload policy after {MAX_RETRIES} attempts "
                    f"due to network error/timeout: {e}"
                ) from e

        except RequestException as e:
            last_exc = e
            if attempt < MAX_RETRIES:
                time.sleep(2 ** (attempt - 1))
                continue
            else:
                raise Exception(
                    f"Failed to get upload policy after {MAX_RETRIES} attempts: {e}"
                ) from e

    raise Exception(
        f"Failed to get upload policy after {MAX_RETRIES} attempts: {last_exc}"
    )


def upload_file_to_oss(policy_data, file_path):
    file_name = Path(file_path).name
    key = f"{policy_data['upload_dir']}/{file_name}"

    last_exc = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with open(file_path, "rb") as file:
                files = {
                    "OSSAccessKeyId": (None, policy_data["oss_access_key_id"]),
                    "Signature": (None, policy_data["signature"]),
                    "policy": (None, policy_data["policy"]),
                    "x-oss-object-acl": (None, policy_data["x_oss_object_acl"]),
                    "x-oss-forbid-overwrite": (
                        None,
                        policy_data["x_oss_forbid_overwrite"],
                    ),
                    "key": (None, key),
                    "success_action_status": (None, "200"),
                    "file": (file_name, file),
                }

                response = requests.post(
                    policy_data["upload_host"],
                    files=files,
                    timeout=DEFAULT_TIMEOUT,
                )

            if response.status_code == 200:
                return f"oss://{key}"

            if response.status_code in (429, 500, 502, 503, 504):
                last_exc = Exception(
                    f"Failed to upload file (attempt {attempt}): "
                    f"{response.status_code} {response.text}"
                )
                if attempt < MAX_RETRIES:
                    time.sleep(2 ** (attempt - 1))
                    continue
                else:
                    raise last_exc

            raise Exception(
                f"Failed to upload file: {response.status_code} {response.text}"
            )

        except (Timeout, ConnectionError) as e:
            last_exc = e
            if attempt < MAX_RETRIES:
                time.sleep(2 ** (attempt - 1))
                continue
            else:
                raise Exception(
                    f"Failed to upload file after {MAX_RETRIES} attempts "
                    f"due to network error/timeout: {e}"
                ) from e

        except RequestException as e:
            last_exc = e
            if attempt < MAX_RETRIES:
                time.sleep(2 ** (attempt - 1))
                continue
            else:
                raise Exception(
                    f"Failed to upload file after {MAX_RETRIES} attempts: {e}"
                ) from e

    raise Exception(
        f"Failed to upload file after {MAX_RETRIES} attempts: {last_exc}"
    )


def upload_file_and_get_url(api_key, model_name, file_path):
    policy_data = get_upload_policy(api_key, model_name)
    oss_url = upload_file_to_oss(policy_data, file_path)
    return oss_url
