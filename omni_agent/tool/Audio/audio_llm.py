from google import genai
import tempfile
import os
import subprocess
from google.genai import types
from omni_agent.config import YOUR_API_KEY_GEMINI, GEMINI_MODEL

from .units import upload_file_and_get_url
from omni_agent.config import YOUR_API_KEY_QWEN, QWEN_AUDIO_MODEL

import os
import dashscope

client = genai.Client(api_key=YOUR_API_KEY_GEMINI)

def get_or_upload_file(client, file_path):
    target_display_name = os.path.basename(file_path)
    for f in client.files.list():
        if f.display_name == target_display_name:
            return f
    file_obj = client.files.upload(
        file=file_path,
        config={
                'display_name': target_display_name,
            }
    )
    while file_obj.state.name == "PROCESSING":
        import time
        time.sleep(1)
        file_obj = client.files.get(name=file_obj.name)
    if file_obj.state.name != "ACTIVE":
        raise Exception(f"File processing failed: {file_obj.state.name}")
    return file_obj


Gemini_Model = GEMINI_MODEL
# Gemini
def audio_llm_gemini(video_path: str, question: str, system_prompt = None) -> str:

    audio_path_template = video_path.replace(".mp4", ".wav")
    audio_path_template = audio_path_template.replace("videos", "audios")

    if os.path.exists(audio_path_template):
        myfile = get_or_upload_file(client, audio_path_template)
        response = client.models.generate_content(
            model=Gemini_Model, contents=[question, myfile]
        )
        return response.text

    else:
        with open(video_path, 'rb') as f:
            audio_bytes = f.read()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            wav_path = tmp_wav.name
        
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error", "-i", video_path, "-vn", wav_path,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        myfile = get_or_upload_file(client, wav_path)

        response = client.models.generate_content(
            model=Gemini_Model, contents=[question, myfile],
        )

        client.files.delete(name=myfile.name)
        return response.text




def audio_llm_qwen(video_path: str, question: str, system_prompt = None) -> str:

    model_name=QWEN_AUDIO_MODEL

    audio_path_template = video_path.replace(".mp4", ".wav")
    audio_path_template = audio_path_template.replace("videos", "audios")

    if os.path.exists(audio_path_template):

        if not hasattr(audio_llm_qwen, "_audio_url_cache"):
            audio_llm_qwen._audio_url_cache = {}
        url_cache = audio_llm_qwen._audio_url_cache

        if audio_path_template in url_cache:
            print(f"Using cached audio URL for {audio_path_template}")
            public_url = url_cache[audio_path_template]
        else:
            public_url = upload_file_and_get_url(YOUR_API_KEY_QWEN, model_name, audio_path_template)
            url_cache[audio_path_template] = public_url
        messages = [
        {
            "role": "user",
            "content": [
                {"audio": public_url},
                {"text": question}]
        }]      

        response = dashscope.MultiModalConversation.call(
            api_key=YOUR_API_KEY_QWEN,
            model=model_name,
            messages=messages
        )
        return response.output.choices[0].message.content[0]["text"]

    else:
        with open(video_path, 'rb') as f:
            audio_bytes = f.read()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            wav_path = tmp_wav.name
        
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error", "-i", video_path, "-vn", wav_path,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if not hasattr(audio_llm_qwen, "_audio_url_cache"):
            audio_llm_qwen._audio_url_cache = {}
        url_cache = audio_llm_qwen._audio_url_cache

        if wav_path in url_cache:
            print(f"Using cached audio URL for {wav_path}")
            public_url = url_cache[wav_path]
        else:
            public_url = upload_file_and_get_url(YOUR_API_KEY_QWEN, model_name, wav_path)
            url_cache[wav_path] = public_url
        messages = [
        {
            "role": "user",
            "content": [
                {"audio": public_url},
                {"text": question}]
        }]      

        response = dashscope.MultiModalConversation.call(
            api_key=YOUR_API_KEY_QWEN,
            model=model_name,
            messages=messages
        )
        return response.output.choices[0].message.content[0]["text"]


