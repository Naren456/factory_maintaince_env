import asyncio
import os
import textwrap
import json
from typing import List, Optional

from openai import OpenAI
from factory_env.client import FactoryEnv, FactoryAction, FactoryObservation

# Configuration from Environment Variables
IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("FACTORY_TASK", "maintenance")
BENCHMARK = os.getenv("FACTORY_BENCHMARK", "factory_v1")

MAX_STEPS = 50
TEMPERATURE = 0.2
MAX_TOKENS = 100
SUCCESS_SCORE_THRESHOLD = 0.5  # Budget at least 50% of starting

# Initial budget is 2000.0
INITIAL_BUDGET = 2000.0
MAX_POSSIBLE_BUDGET = 5000.0 # Heuristic for normalization

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an AI Factory Manager. Your goal is to maximize the factory's production budget through smart maintenance.
    You manage 3 machines. 
    Status Levels: operational (optimal), warning (needs check), broken (downtime penalty).
    
    Actions:
    - wait: Continue production (Low cost $10).
    - inspect <id>: View details (Cost $30).
    - repair <id>: Fix wear/tear (Cost $150).
    - replace <id>: Buy new machine (Cost $600).
    
    Penalty: Broken machines cause $200 downtime penalty per step.
    
    IMPORTANT: Reply ONLY with the action and machine ID (if applicable).
    Examples:
    - wait
    - repair 0
    - inspect 2
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def build_user_prompt(step: int, obs: FactoryObservation) -> str:
    machines_summary = "\n".join([
        f"Machine {m.id}: {m.status.upper()} (Health: {m.health*100:.1f}%)" 
        for m in obs.machines
    ])
    return textwrap.dedent(
        f"""
        Step: {step}
        Current Budget: ${obs.budget:.2f}
        Production Rate: {obs.production_rate:.1f}%
        Machine Status:
        {machines_summary}
        
        Last Event: {obs.last_event}
        What is your next action?
        """
    ).strip()


def parse_action(text: str) -> FactoryAction:
    text = text.lower().strip()
    parts = text.split()
    if not parts:
        return FactoryAction(type="wait")
    
    action_type = parts[0]
    if action_type not in ["wait", "inspect", "repair", "replace"]:
        action_type = "wait"
    
    machine_id = None
    if len(parts) > 1 and parts[1].isdigit():
        machine_id = int(parts[1])
    
    return FactoryAction(type=action_type, machine_id=machine_id)


def get_model_action(client: OpenAI, step: int, obs: FactoryObservation) -> str:
    user_prompt = build_user_prompt(step, obs)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        return (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "wait"


async def main() -> None:
    if not API_KEY:
        print("[ERROR] HF_TOKEN or API_KEY not set", flush=True)
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Initialize environment
    if IMAGE_NAME:
        env = await FactoryEnv.from_docker_image(IMAGE_NAME)
    else:
        # Fallback to local server if no image specified
        env = FactoryEnv(base_url="http://localhost:8000")

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        async with env:
            result = await env.reset()
            current_obs = result.observation

            for step in range(1, MAX_STEPS + 1):
                if result.done:
                    break

                action_text = get_model_action(client, step, current_obs)
                action = parse_action(action_text)

                result = await env.step(action)
                current_obs = result.observation
                
                reward = result.reward or 0.0
                done = result.done
                
                rewards.append(reward)
                steps_taken = step
                
                log_step(step=step, action=action_text, reward=reward, done=done, error=None)

                if done:
                    break

            # Calculate final score based on budget retention
            score = current_obs.budget / INITIAL_BUDGET
            score = min(max(score, 0.0), 1.0)
            success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[ERROR] Simulation failed: {e}", flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
