import argparse
from omni_agent.agent_builder import build_agent

agent = build_agent(max_iterations=30)

def run_once(video_path: str, question: str) -> None:

    os.makedirs("Cache", exist_ok=True)
    
    result = agent.invoke(
        {
            "video_path": video_path,
            "question": question,
        }
    )

    print("\n=== Final Answer ===")
    print(result["output"])

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run the agent on a video and question prompt.")
    parser.add_argument(
        "--video_path", type=str, default=None,
        help="Path to the video file. If not provided, use the default example."
    )
    parser.add_argument(
        "--question", type=str, default=None,
        help="Question prompt. If not provided, use the default example multiple-choice question."
    )
    args = parser.parse_args()

    if args.video_path and args.question:
        run_once(args.video_path, args.question)
    else:
        # From Daily-Omni Benchmark
        test_video_path = "example/d6b4OmUFt7I_video.mp4"
        Question = "Which visual sequences correspond to the audio mentions of 'go on the equipment' versus 'familiar with the machines'?"
        Choice = [
            "A. First shows an excavator lifting dirt, second shows a dump truck driving through mud",
            "B. First shows binder review in office, second shows tractor parked on site",
            "C. First shows safety vest demonstration, second shows mountain landscape",
            "D. First shows a bulldozer moving earth, second shows a water tank in background"
        ]

        choices_joined = "\n".join([f"{c}" for c in Choice])

        # Always use in benchmark eval
        prompt = (
            f"{Question}\n{choices_joined}\n"
            "Please select the most correct answer (A/B/C/D) and output your choice wrapped in <answer> tags, e.g., <answer>A</answer>."
        )
        run_once(test_video_path, prompt)

