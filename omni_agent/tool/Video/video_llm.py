# av_agent/tools/video_tools.py
from typing import Dict, Optional, List, Tuple
import os
import base64
import numpy as np
import cv2  # pip install opencv-python
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import subprocess
from dashscope import MultiModalConversation
import dashscope 

from omni_agent.config import OPENAI_API_KEY, YOUR_API_KEY_QWEN, MODEL_BASE_URL, QWEN_VIDEO_MODEL
from omni_agent.config import YOUR_API_KEY_GEMINI, GEMINI_MODEL
import time

from google import genai
from google.genai import types

from .upload import upload_file_and_get_url

_vision_llm = ChatOpenAI(
    model=QWEN_VIDEO_MODEL,
    api_key=YOUR_API_KEY_QWEN,
    base_url=MODEL_BASE_URL,
)

client = genai.Client(api_key=YOUR_API_KEY_GEMINI)


QWEN_MODEL = QWEN_VIDEO_MODEL


def video_llm(video_path, text_block) -> str:

    public_url = upload_file_and_get_url(YOUR_API_KEY_QWEN, QWEN_MODEL, video_path)
    messages = [
        {
            'role':'user',
            'content': [{'video': public_url, "fps":2},
                {'text': text_block}]
        }
    ]
    response = MultiModalConversation.call(
        api_key=YOUR_API_KEY_QWEN,
        model=QWEN_MODEL,
        messages=messages)

    return response.output.choices[0].message.content[0]["text"]


def video_llm_gemini(video_path, text_block, fps=2) -> str:

    myfile = client.files.upload(file=video_path)
    file_name = myfile.name

    myfile_info = client.files.get(name=file_name)
    while myfile_info.state == "PROCESSING":
        time.sleep(0.5)
        myfile_info = client.files.get(name=file_name)

    prompt = text_block
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
        types.Part(
            file_data=types.FileData(
                file_uri=myfile.uri,
                mime_type=myfile.mime_type
            ),
            video_metadata=types.VideoMetadata(
                fps=5
            )
        ),
        types.Part(text=text_block)
        ],
    )

    client.files.delete(name=myfile.name)
    return response.text
