import asyncio
import os
import textwrap
from typing import List, Optional

from openai import OpenAI
from factory_env.client import FactoryEnv, FactoryAction, FactoryObservation

# Configuration for LOCAL Ollama
# Default Ollama OpenAI-compatible endpoint
API_BASE_URL = os.getenv("LOCAL_API_BASE_URL") or "http://localhost:11434/v1"
MODEL_NAME = os.getenv("LOCAL_MODEL_NAME") or "qwen2.5:7b" 
API_KEY = "ollama" # Generic key for Ollama

TASK_NAME = "local-maintenance"
BENCHMARK = "factory_local"
MAX_STEPS = 50
TEMPERATURE = 0.2
SUCCESS_SCORE_THRESHOLD = 0.5

INITIAL_BUDGET = 2000.0

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an AI Factory Manager. Your goal is to maximize the factory's production budget through smart maintenance.
    You manage 3 machines. 
    Status Levels: operational (optimal), warning (needs check), broken (high penalty).
    
    Actions:
    - wait: Continue production (Low cost $10).
    - inspect <id>: View details (Cost $30).
    - repair <id>: Fix wear/tear (Cost $150).
    - replace <id>: Buy new machine (Cost $600).
    
    IMPORTANT: Reply ONLY with the action and machine ID (if applicable).
    Example Output:
    wait
    repair 0
    inspect 2
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
        Machine Status:
        {machines_summary}
        
        Last Event: {obs.last_event}
        What is your next action?
        """
    ).strip()


def parse_action(text: str) -> FactoryAction:
    text = text.lower().replace("'", "").strip()
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
            stream=False,
        )
        return (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[DEBUG] Ollama local request failed: {exc}", flush=True)
        return "wait"


async def main() -> None:
    # Use standard OpenAI client pointing to Ollama
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Point to the local environment server (running via uvicorn)
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

            score = current_obs.budget / INITIAL_BUDGET
            score = min(max(score, 0.0), 1.0)
            success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[ERROR] Ollama simulation failed: {e}", flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    import sys
    print(f"Connecting to Ollama at {API_BASE_URL} using model {MODEL_NAME}...")
    asyncio.run(main())
