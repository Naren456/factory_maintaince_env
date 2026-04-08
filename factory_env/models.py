# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Factory Env Environment.

The factory_env environment is a simple test environment that echoes back messages.
"""

from typing import List, Optional, Literal
from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field


class MachineState(BaseModel):
    """Represents the current state of a factory machine."""
    id: int = Field(..., description="Unique identifier for the machine")
    status: Literal["operational", "warning", "broken"] = Field(
        default="operational", description="Current status level"
    )
    health: float = Field(
        default=1.0, description="Condition score from 0.0 (dead) to 1.0 (new)"
    )
    last_maint: int = Field(
        default=0, description="Step count when last maintained"
    )


class FactoryAction(Action):
    """Action for the Factory Maintenance environment."""
    type: Literal["inspect", "repair", "replace", "wait"] = Field(
        ..., description="Type of maintenance action"
    )
    machine_id: Optional[int] = Field(
        None, description="ID of the target machine (ignored for 'wait')"
    )


class FactoryObservation(Observation):
    """Observation from the Factory Maintenance environment."""
    machines: List[MachineState] = Field(
        default_factory=list, description="State of all machines in the factory"
    )
    production_rate: float = Field(
        default=0.0, description="Current units produced per step"
    )
    budget: float = Field(
        default=1000.0, description="Remaining financial resources"
    )
    last_event: str = Field(
        default="", description="Description of what happened in the last step"
    )
    reward: float = Field(
        default=0.0, description="Reward gained in the last step"
    )
    done: bool = Field(
        default=False, description="Whether the episode is finished"
    )

