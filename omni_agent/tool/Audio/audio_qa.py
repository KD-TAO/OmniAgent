from .audio_llm import audio_llm_gemini
from .audio_llm import audio_llm_gemini as audio_llm
from .audio_llm import audio_llm_qwen
from langchain_core.tools import tool
from typing import Dict
from langchain_core.tools import ToolException
from omni_agent.config import LOCATION_TOOL, ASR_GC_TOOL

@tool
def audio_global_caption(video_path: str) -> Dict[str, str]:
    """
    Generate a global, high-level caption/description for the audio track of a video.

    This tool:
    1) Extracts the audio track from the given video file.
    2) Uses AudioLLM to listen to the entire audio and produce
       a single global description summarizing the main topics,
       structure, key events, speakers (if distinguishable), and overall tone.

    It is typically called once per video to obtain an overall understanding
    of the audio content, but the agent may call it again if needed.

    Args:
        video_path: Path or identifier of the video file from which
            the audio track will be extracted.

    Returns:
        A dictionary with:
            - "caption": a global, high-level description of the audio track.
    """
    question = (
        "Provide a high-level summary of the audio. "
        "Focus on the main topics, key events, and the overall atmosphere, "
    )

    if LOCATION_TOOL == "GEMINI":
        try:
            audio_notes =  audio_llm(video_path, question)
        except Exception as e:
            raise ToolException(f"[audio_global_caption error] {type(e).__name__}: {e}")
        return {
            "answer": audio_notes,
        }
    elif LOCATION_TOOL == "QWEN":
        try:
            audio_notes =  audio_llm_qwen(video_path, question)
        except Exception as e:
            raise ToolException(f"[audio_global_caption error] {type(e).__name__}: {e}")
        return {
            "answer": audio_notes,
        }
    else:
        raise ToolException(f"[audio_global_caption error] Invalid audio tool: {LOCATION_TOOL}")


@tool
def audio_ASR(video_path: str) -> Dict[str, str]:
    """
    Generate a timestamped transcript of the audio track of a video.

    Args:
        video_path: Path or identifier of the video file.

    This tool:
    1) Extracts the audio track from the given video file.
    2) Uses AudioLLM to produce a verbatim transcript with timestamps for every sentence/segment.
    """
    
    question = (
        "You are a professional transcriber. "
        "Task: Generate a verbatim transcript of the speech, including precise timestamps for each sentence or natural segment.\n\n"

        "### REQUIRED OUTPUT FORMAT (Strict):\n"
        "You must output a list where every line follows this exact format:\n"
        "**MM:SS-MM:SS** Transcript text here\n"
        "**MM:SS-MM:SS** Next sentence here\n\n"

        "### CRITICAL ANTI-REPETITION & NOISE RULES:\n"
        "1. **Transcribe Speech Only**: Focus on clear spoken dialogue.\n"
        "2. **Handle Repetitive Sounds**: If you hear repetitive noises (e.g., 'wu wu wu...', continuous laughter), "
        "**DO NOT** repeat the text. Instead, use a bracketed summary with the timestamp.\n"
    )

    if ASR_GC_TOOL == "GEMINI":
        try:
            audio_notes = audio_llm(video_path, question)
        except Exception as e:
            raise ToolException(f"[audio_ASR error] {type(e).__name__}: {e}")
        return {
            "answer": audio_notes,
        }
    elif ASR_GC_TOOL == "QWEN":
        try:
            audio_notes = audio_llm_qwen(video_path, question)
        except Exception as e:
            raise ToolException(f"[audio_ASR error] {type(e).__name__}: {e}")
        return {
            "answer": audio_notes,
        }
    else:
        raise ToolException(f"[audio_ASR error] Invalid audio tool: {ASR_GC_TOOL}")

if __name__ == "__main__":
    out = audio_ASR.invoke({"video_path": "assert/026dzf-vc5g_video.mp4", "time_range": ("00:05", "00:15")})
    print(out)