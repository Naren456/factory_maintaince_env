import json
from typing import List, Dict, Any
import argparse

def grade_trajectory(trajectory: List[Dict[str, Any]]) -> float:
    """
    Programmatic grader for the Factory Maintenance environment.
    
    Scores are based on:
    - Final budget vs initial budget
    - Minimum health maintained across machines
    - Episode completion
    
    Returns:
        float: Score between 0.0 and 1.0
    """
    if not trajectory:
        return 0.0
        
    last_obs = trajectory[-1]
    
    # Extract data from the last observation
    # In OpenEnv, these fields match FactoryObservation
    budget = last_obs.get("budget", 0.0)
    task_id = last_obs.get("task_id", "medium")
    machines = last_obs.get("machines", [])
    
    # Determine initial budget for normalization
    initial_budget = 2000.0
    if task_id == "easy":
        initial_budget = 3000.0
    elif task_id == "hard":
        initial_budget = 1000.0
        
    # 1. Budget Score (0.0 to 1.0)
    # We reward making a profit, but cap it at 1.0 for the grader
    budget_score = max(0.0, min(1.0, budget / initial_budget))
    
    # 2. Penalty for broken machines
    # If any machine ended in "broken" status, we apply a specific penalty
    broken_count = sum(1 for m in machines if m.get("status") == "broken")
    broken_penalty = (broken_count / len(machines)) * 0.2 if machines else 0
    
    final_score = max(0.0, budget_score - broken_penalty)
    
    return float(round(final_score, 3))

def main():
    parser = argparse.ArgumentParser(description="Programmatic grader for Factory Maintenance")
    parser.add_argument("input_file", help="Path to JSON file containing observation trajectory")
    args = parser.parse_args()
    
    try:
        with open(args.input_file, "r") as f:
            data = json.load(f)
            # Support both a direct list of observations or a wrapped object
            trajectory = data if isinstance(data, list) else data.get("observations", [])
            
        score = grade_trajectory(trajectory)
        print(f"Final Score: {score}")
        
    except Exception as e:
        print(f"Error grading trajectory: {e}")
        exit(1)

if __name__ == "__main__":
    main()
