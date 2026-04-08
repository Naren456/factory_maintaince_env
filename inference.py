import asyncio
import os
import textwrap
import re
from typing import List, Optional
from openai import OpenAI
from dotenv import load_dotenv

# Local imports
from client import FactoryEnv
from models import FactoryAction

load_dotenv()

# MANDATORY variables from OpenEnv template
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# Environment specifics
TASK_NAME = "factory-maintenance"
BENCHMARK = "factory_env"
MAX_STEPS = 50
SUCCESS_SCORE_THRESHOLD = 0.5

# Standardized logging helpers
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

def parse_action(raw_output: str) -> FactoryAction:
    """Robustly parse the LLM output into a FactoryAction."""
    # Clean input: lowercase and remove common punctuation at the end tokens
    clean_text = raw_output.lower().strip().replace(":", "").replace(",", "")
    parts = clean_text.split()
    
    if not parts:
        return FactoryAction(type="wait")
    
    action_type = parts[0]
    machine_id = None
    
    # Common action types
    valid_types = ["wait", "inspect", "repair", "replace"]
    if action_type not in valid_types:
        # Try to find a valid type in the parts
        found = False
        for p in parts:
            if p in valid_types:
                action_type = p
                found = True
                break
        if not found:
            action_type = "wait"

    # Extract ID (look for any number in the parts)
    for p in parts:
        # Check if digit or contains digit
        match = re.search(r"\d+", p)
        if match:
            machine_id = int(match.group())
            break
            
    return FactoryAction(type=action_type, machine_id=machine_id)

def get_model_action(client: OpenAI, obs, history: List[str]) -> str:
    machines_summary = "\n".join([
        f"Machine {m.id}: {m.status} (Health: {m.health*100:.1f}%)" 
        for m in obs.machines
    ])
    
    prompt = textwrap.dedent(f"""
        You are an AI Industrial Controller. 
        FACTORY STATUS (Current Step: {obs.metadata.get('step', 0)}/50)
        Budget: ${obs.budget:.2f} (If budget <= 0, you lose!)
        Production Rate: {obs.production_rate:.1f}%

        MACHINES:
        {machines_summary}
        
        LAST EVENT: {obs.last_event}

        RULES:
        - 'wait': Cost $10, produces revenue based on health.
        - 'inspect <id>': Cost $30, reveals health.
        - 'repair <id>': Cost $150, restores partial health.
        - 'replace <id>': Cost $600, restores 100% health.
        
        STRATEGY: Balance repair costs with production revenue. Do not let machines break (Health < 10%).
        GOAL: Maximize final budget.

        Input only the lowercase action string (e.g. 'wait' or 'repair 0').
    """).strip()

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are an industrial RL agent. Output only the command."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=20,
            temperature=0.0
        )
        return response.choices[0].message.content.strip().lower()
    except Exception as e:
        print(f"[DEBUG] LLM Error: {e}", flush=True)
        return "wait"

async def main() -> None:
    if not HF_TOKEN:
        print("[ERROR] HF_TOKEN is missing.", flush=True)
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    
    # Connect to the environment
    # Use from_docker_image if LOCAL_IMAGE_NAME is set, else use localhost
    if LOCAL_IMAGE_NAME:
        env_client = FactoryEnv.from_docker_image(LOCAL_IMAGE_NAME)
    else:
        env_client = FactoryEnv(base_url="http://localhost:8000")

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        # We use the sync manager for simplicity in the loop, or async if preferred
        # Here we follow the FactoryEnv sync pattern established earlier
        with env_client.sync() as env:
            result = env.reset()
            obs = result.observation
            if not obs:
                raise ValueError("Environment failed to return initial observation.")

            for step in range(1, MAX_STEPS + 1):
                if result.done:
                    break

                raw_action = get_model_action(client, obs, history)
                action = parse_action(raw_action)
                
                # Execute
                result = env.step(action)
                obs = result.observation
                if not obs:
                    break
                
                reward = result.reward or 0.0
                done = result.done
                error = None 

                rewards.append(reward)
                steps_taken = step
                
                log_step(step=step, action=f"{action.type} {action.machine_id or ''}", reward=reward, done=done, error=error)
                history.append(f"Step {step}: {raw_action} -> {reward:.2f}")

                if done:
                    break

            # Use the environment's built-in score/success criteria
            score = obs.score
            success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Inference error: {e}", flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    if not os.environ.get("HF_TOKEN") and os.environ.get("OPENAI_API_KEY"):
        os.environ["HF_TOKEN"] = os.environ["OPENAI_API_KEY"]
    asyncio.run(main())
