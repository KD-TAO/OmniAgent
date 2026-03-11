from langchain_openai import ChatOpenAI
from omni_agent.config import OPENAI_API_KEY, BRAIN_MODEL, MODEL_BASE_URL_OPENAI
import os

def get_brain_llm() -> ChatOpenAI:
    """
    Return the central 'brain' LLM used for planning, reasoning, and tool calling.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set in the environment.")
    #print(BRAIN_MODEL)

    llm = ChatOpenAI(
        model=BRAIN_MODEL,
        temperature=1,
        openai_api_key=OPENAI_API_KEY,
        base_url=MODEL_BASE_URL_OPENAI,
        extra_body={"reasoning_effort": "high"},
    )

    return llm
