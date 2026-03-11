from .audio_llm import audio_llm_gemini as audio_llm
from .audio_llm import audio_llm_qwen
from langchain_core.tools import tool
from langchain_core.tools import ToolException
from typing import Dict
from omni_agent.config import LOCATION_TOOL

@tool
def Audio_EventList(video_path: str) -> str:
    """
    Analyze the audio track of a video and produce a time-ordered list of MAJOR
    audio events with medium time resolution.

    This tool:
    - Uses the entire audio track.
    - Segments it into consecutive, non-overlapping semantic intervals.
    - Returns a list covering the audio from 00:00 to the end.

    Args:
        video_path: Path to the video file.

    Returns:
        A single Markdown string containing the time-stamped event list.
    """
    prompt = """
    You are an expert Audio Content Analyst. Your task is to generate a structured timeline of **significant semantic events** for the entire audio track.

    ### Objective: Continuous & Semantic Timeline
    Create a timeline that divides the audio into logical "chapters" or "scenes".

    ### Coverage & Precision Rules (Critical):
    1.  **Full Duration:** You MUST start at **00:00** and cover the audio until the very end of the file. Do not skip any time periods.
    2.  **No Gaps:** The start time of a new segment should typically match (or be very close to) the end time of the previous segment. The timeline must be contiguous.
    3.  **Precise Boundaries:** Listen carefully to identify the **exact second** where a scene transitions (e.g., when the music stops or a new speaker actually starts).

    ### Segmentation Logic (Medium Granularity):
    1.  **Merge, Don't Split:** Treat a continuous conversation, a sustained musical piece, or a consistent environment as a SINGLE segment.
        * *Example:* If two people talk for 2 minutes about the same topic, that is ONE segment, not twenty short ones.
    2.  **Trigger for New Segment:** Only start a new segment when there is a **definite shift** in context:
        * Topic change.
        * Primary speaker switch (in formal structured turns, not quick banter).
        * Distinct Speech-to-Music or Environment change.
    3.  **Ignore Noise:** Disregard short interruptions (<3s), coughs, or filler words.

    ### Output Format:
    * Strictly output a **Markdown bullet list**.
    * Format: `* **MM:SS - MM:SS**: [Concise Description]`
    """
    if LOCATION_TOOL == "GEMINI":
        try:
            return audio_llm(video_path, prompt)
        except Exception as e:
            raise ToolException(f"[Audio_EventList error] {type(e).__name__}: {e}")
    elif LOCATION_TOOL == "QWEN":
        try:
            return audio_llm_qwen(video_path, prompt)
        except Exception as e:
            raise ToolException(f"[Audio_EventList error] {type(e).__name__}: {e}")
    else:
        raise ToolException(f"[Audio_EventList error] Invalid audio tool: {LOCATION_TOOL}")

@tool
def Audio_EventLocation(video_path: str, query: str) -> str:
    """
    Locate specific audio events/sounds with high temporal precision based on a user query.
    
    Args:
        video_path: Path to the video file.
        query: Description of the sound/event to find (e.g., "the first time he laughs", "all applause").
    
    Returns:
        A Markdown list of timestamps and descriptions.
    """
    prompt = f"""
    Role: Precision Audio Analyst.
    Task: Locate the exact timestamps in the audio track that match the User Query.

    User Query: "{query}"

    ### ⚡ Search Protocols (Strict):
    1.  **Precision First:** Pinpoint the **exact second** the event starts. Do not give vague ranges.
    2.  **Point vs. Duration:** * For instant sounds (e.g., a gunshot, a scream), use a single timestamp: **MM:SS**.
        * For sustained events (e.g., a speech segment, a song), use a range: **MM:SS - MM:SS**.
    3.  **Quantity Logic:**
        * If the query specifies a count (e.g., "first time", "top 2", "last occurrence"), **strictly obey it**.
        * If unspecified, list **all** clear occurrences (merge adjacent ones if < 2s apart).
    4.  **Anti-Hallucination:** If the specific event is NOT found, output exactly: `* **N/A**: Event not found.`

    ### Output Format:
    * Return a clean Markdown bullet list ONLY.
    * Format: `* **Timestamp**: [Context/Detail] Why this matches.`
    """.strip()

    if LOCATION_TOOL == "GEMINI":
        try:
            return audio_llm(video_path, prompt)
        except Exception as e:
            raise ToolException(f"[Audio_EventLocation error] {type(e).__name__}: {e}")
    elif LOCATION_TOOL == "QWEN":
        try:
            return audio_llm_qwen(video_path, prompt)
        except Exception as e:
            raise ToolException(f"[Audio_EventLocation error] {type(e).__name__}: {e}")
    else:
        raise ToolException(f"[Audio_EventLocation error] Invalid audio tool: {LOCATION_TOOL}")


def _analyze_audio_with_llm(
    question: str,
) -> str:

    text_block = (
        "You will be given one audio.\n\n"
        "Your tasks are:\n"
        "1. Carefully listen to the audio and understand what is being said and what sounds are present.\n"
        "2. Reason step by step if necessary.\n"
        "3. Answer the user's question as precisely as possible, "
        "using only what can be inferred from the audio.\n\n"
        f"User question:\n{question}\n\n"
    )

    return text_block

@tool
def audio_qa(video_path: str, question: str) -> Dict[str, str]:
    """
    Ask a free-form question about the audio track of the registered media.

    The agent can call this tool multiple times with different sub-questions.
    This tool:
    1) Uses AudioLLM to reason over the audio
       and answer the question as precisely as possible.

    Args:
        video_path: Path or identifier of the video file (Extract the audio from the video).
        question: Any natural-language question about the audio track.

    Returns:
        A dictionary with:
            - "answer": the model's answer to the question.
    """
    question = _analyze_audio_with_llm(question)

    if LOCATION_TOOL == "GEMINI":
        try:
            audio_notes =  audio_llm(video_path, question)
        except Exception as e:
            raise ToolException(f"[audio_qa error] {type(e).__name__}: {e}")
        return {
            "answer": audio_notes,
        }
    elif LOCATION_TOOL == "QWEN":
        try:
            audio_notes =  audio_llm_qwen(video_path, question)
        except Exception as e:
            raise ToolException(f"[audio_qa error] {type(e).__name__}: {e}")
        return {
            "answer": audio_notes,
        }
    else:
        raise ToolException(f"[audio_qa error] Invalid audio tool: {LOCATION_TOOL}")
