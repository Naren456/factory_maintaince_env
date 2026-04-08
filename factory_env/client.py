# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Factory Env Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import FactoryAction, FactoryObservation, MachineState


class FactoryEnv(
    EnvClient[FactoryAction, FactoryObservation, State]
):
    """
    Client for the Factory Env Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with FactoryEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.echoed_message)
        ...
        ...     result = client.step(FactoryAction(message="Hello!"))
        ...     print(result.observation.echoed_message)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = FactoryEnv.from_docker_image("factory_env-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(FactoryAction(message="Test"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: FactoryAction) -> Dict:
        """
        Convert FactoryAction to JSON payload for step message.
        """
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict) -> StepResult[FactoryObservation]:
        """
        Parse server response into StepResult[FactoryObservation].
        """
        obs_data = payload.get("observation", {})
        
        # Parse machine list
        machines_data = obs_data.get("machines", [])
        machines = [MachineState.model_validate(m) for m in machines_data]

        observation = FactoryObservation(
            machines=machines,
            production_rate=obs_data.get("production_rate", 0.0),
            budget=obs_data.get("budget", 0.0),
            last_event=obs_data.get("last_event", ""),
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )


    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
