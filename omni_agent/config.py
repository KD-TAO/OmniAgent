import os

# ------------------ Main agent configuration ------------------ #
OPENAI_API_KEY = ""
YOUR_API_KEY_GEMINI = ""
YOUR_API_KEY_QWEN = ""

BRAIN_MODEL = os.getenv("BRAIN_MODEL", "o3")
MODEL_BASE_URL_OPENAI = ""
GEMINI_MODEL = "gemini-2.5-flash"

QWEN_AUDIO_MODEL = "qwen3-omni-flash"
QWEN_VIDEO_MODEL = "qwen3-vl-flash"
MODEL_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


# ------------------ Tool configuration ------------------ #
VIDEO_TOOL = "QWEN"
ASR_GC_TOOL = "QWEN"
LOCATION_TOOL = "GEMINI"