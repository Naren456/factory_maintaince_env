import random
from uuid import uuid4
from typing import List, Dict, Any

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import FactoryAction, FactoryObservation, MachineState
except ImportError:
    from models import FactoryAction, FactoryObservation, MachineState


class FactoryEnvironment(Environment):
    """
    A Factory Maintenance Environment.
    
    Agents must manage a set of machines to maximize production profit.
    Machines decay over time and require inspection, repair, or replacement.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    
    # Configuration
    NUM_MACHINES = 3
    BASE_PRODUCTION_REVENUE = 100.0
    REPAIR_COST = 150.0
    REPLACE_COST = 600.0
    INSPECT_COST = 30.0
    WAIT_COST = 10.0  # Basic operational cost
    DOWNTIME_PENALTY = 200.0
    MAX_STEPS = 50

    def __init__(self):
        """Initialize the factory environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.budget = 2000.0
        self.machines: List[MachineState] = []
        self._initialize_machines()

    def _initialize_machines(self):
        self.machines = []
        for i in range(self.NUM_MACHINES):
            self.machines.append(MachineState(
                id=i,
                status="operational",
                health=random.uniform(0.8, 1.0),
                last_maint=0
            ))

    def reset(self) -> FactoryObservation:
        """Reset the environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.budget = 2000.0
        self._initialize_machines()
        
        return self._make_observation("Environment reset. Factory operation started.", 0.0)

    def step(self, action: FactoryAction) -> FactoryObservation:  # type: ignore[override]
        """Execute a step in the factory."""
        self._state.step_count += 1
        
        event_msg = ""
        action_cost = 0.0
        
        # 1. Process Maintenance Action
        if action.type == "wait":
            action_cost = self.WAIT_COST
            event_msg = "Factory operating normally (Waiting)."
        else:
            m_id = action.machine_id
            if m_id is None or m_id < 0 or m_id >= len(self.machines):
                event_msg = f"Invalid machine ID: {m_id}. Step skipped."
                action_cost = self.WAIT_COST
            else:
                target = self.machines[m_id]
                if action.type == "inspect":
                    action_cost = self.INSPECT_COST
                    event_msg = f"Inspected Machine {m_id}. Health: {target.health:.2f}"
                elif action.type == "repair":
                    action_cost = self.REPAIR_COST
                    if target.status == "broken":
                        target.health = min(0.7, target.health + 0.4)
                        target.status = "warning"
                    else:
                        target.health = min(1.0, target.health + 0.3)
                        target.status = "operational"
                    target.last_maint = self._state.step_count
                    event_msg = f"Repaired Machine {m_id}."
                elif action.type == "replace":
                    action_cost = self.REPLACE_COST
                    target.health = 1.0
                    target.status = "operational"
                    target.last_maint = self._state.step_count
                    event_msg = f"Replaced Machine {m_id} with new unit."

        # 2. Production Logic
        # Calculate revenue based on operational machines
        operational_count = sum(1 for m in self.machines if m.status == "operational")
        warning_count = sum(1 for m in self.machines if m.status == "warning")
        broken_count = sum(1 for m in self.machines if m.status == "broken")
        
        production_multiplier = (operational_count * 1.0) + (warning_count * 0.5)
        revenue = self.BASE_PRODUCTION_REVENUE * (production_multiplier / self.NUM_MACHINES)
        
        # Deduct costs and penalty
        downtime_cost = broken_count * self.DOWNTIME_PENALTY
        step_reward = revenue - action_cost - downtime_cost
        self.budget += step_reward

        # 3. Health Decay Logic (at the end of step)
        for m in self.machines:
            # Random decay
            decay = random.uniform(0.01, 0.05)
            if m.status == "warning":
                decay *= 2.0
            
            m.health = max(0.0, m.health - decay)
            
            # Update status based on health
            if m.health <= 0.1:
                m.status = "broken"
            elif m.health <= 0.5:
                m.status = "warning"
            else:
                m.status = "operational"

        # Check if done
        done = self._state.step_count >= self.MAX_STEPS or self.budget <= 0
        
        return self._make_observation(event_msg, step_reward, done)

    def _make_observation(self, msg: str, reward: float, done: bool = False) -> FactoryObservation:
        # Calculate current production rate
        op_count = sum(1 for m in self.machines if m.status == "operational")
        wr_count = sum(1 for m in self.machines if m.status == "warning")
        prod_rate = (op_count + wr_count * 0.5) / self.NUM_MACHINES * 100.0

        return FactoryObservation(
            machines=[m.model_copy() for m in self.machines],
            production_rate=prod_rate,
            budget=self.budget,
            last_event=msg,
            reward=reward,
            done=done,
            metadata={
                "step": self._state.step_count,
                "broken_machines": self.NUM_MACHINES - op_count - wr_count
            }
        )

    @property
    def state(self) -> State:
        """Get the current environment state."""
        return self._state

