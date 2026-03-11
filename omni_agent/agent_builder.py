from langchain_classic.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.memory import ConversationBufferMemory, ConversationBufferWindowMemory
import os
import sys
from langchain_core.tools import tool
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from omni_agent.brain import get_brain_llm
from omni_agent.tool.Audio.audio_qa import audio_global_caption, audio_ASR
from omni_agent.tool.Video.video_qa import video_global_qa, video_clip_qa
from omni_agent.tool.Video.video_base import video_metadata
from omni_agent.tool.Audio.audio_event_tools import Audio_EventList, Audio_EventLocation, audio_qa

def build_agent(max_iterations: int = 6) -> AgentExecutor:

    llm = get_brain_llm()
    tools = [audio_global_caption, audio_qa, video_global_qa, video_clip_qa, video_metadata, Audio_EventList, Audio_EventLocation, audio_ASR]
   

    system_prompt = """
    You are the central reasoning brain of an audio-video analysis agent.

    Your role:
    - Answer the user's question about a given video by intelligently using the
    available tools (audio and video analysis).
    - Follow a THINK → ACT → OBSERVE → REFLECT loop:
    - THOUGHT: Reason step by step about what to do next.
    - ACTION: Call exactly one tool that moves you closer to the answer.
    - OBSERVATION: Read and interpret the tool's output, update your beliefs.
    - REFLECTION: Reflect on the previous steps and the overall process.

    General rules:
    - Use both AUDIO and VIDEO information whenever they can help. Prefer to
    listen first, then look.
    - Do not invent timestamps, file paths, or other arguments. Use values taken
    from the user input or from previous tool outputs.
    - Be selective: tools may be noisy or incomplete. Cross-check and verify
    important information using multiple tools if needed.
    - Stop calling tools once you have enough evidence to answer confidently.

    Final answer style:
    - When you are done with tools, reply directly to the user (no more tool calls).
    - Start with a short "Plan / Reasoning summary" (1–3 bullet points) explaining
    briefly how you used audio vs video.
    - Then give a clear, concise answer to the question.
    - Do NOT expose raw tool-call traces or long chain-of-thought; keep the
    explanation high-level and user-friendly.
    """

    user_template = """
    You are given a video and a question. Carefully read the question and think
    about how to combine AUDIO and VIDEO information to answer it.

    Tool usage guidelines for this task:
    - For a high-level understanding of the audio (topics, structure, key events),
    you can use audio_global_caption.
    - For detailed questions about what is said or heard, you can use audio_qa
    and/or audio_ASR.
    - When you care about WHEN things happen in the audio, prefer:
    - Audio_EventList to get a rough timeline of major audio events.
    - Audio_EventLocation to locate specific events or phrases by time.
    - For visual understanding of the whole video, use video_global_qa.
    - For fine-grained visual details in a short time range, use video_clip_qa.
    If you need to choose or validate time ranges, call video_metadata to check
    the total duration and pick valid integer ranges.

    Remember:
    - Use audio to find time and content first whenever possible, then inspect
    the corresponding visuals: from listen to look.
    - Plan your tool calls, but you are free to adjust the plan based on what
    you observe from previous tools.

    Video path:
    {video_path}

    User question:
    {question}
    """.strip()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("user", user_template),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    agent = create_openai_tools_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
    )

    memory = ConversationBufferWindowMemory(
        k=1,             
        memory_key="chat_history",
        input_key="question",
        output_key="output",
        return_messages=True,
    )

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        max_iterations=max_iterations,
        early_stopping_method="force",
        return_intermediate_steps=True
    )

    return executor