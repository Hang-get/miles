from collections.abc import Sequence

from tinker_agent.recording import TokenTurn
from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.rl.types import Trajectory, Transition

import tinker


def turns_to_trajectory(turns: Sequence[TokenTurn], stop_reason: str) -> Trajectory:
    transitions = [
        Transition(
            ob=tinker.ModelInput.from_ints(list(turn.input_ids)),
            ac=TokensWithLogprobs(
                tokens=list(turn.output_ids), maybe_logprobs=list(turn.logprobs), stop_reason=turn.stop_reason
            ),
            reward=0.0,
            episode_done=index == len(turns) - 1,
        )
        for index, turn in enumerate(turns)
    ]
    if not transitions:
        transitions = [
            Transition(
                ob=tinker.ModelInput.empty(),
                ac=TokensWithLogprobs(tokens=[], maybe_logprobs=[]),
                reward=0.0,
                episode_done=True,
            )
        ]
    final_ob = (
        tinker.ModelInput.from_ints([*turns[-1].input_ids, *turns[-1].output_ids])
        if turns
        else tinker.ModelInput.empty()
    )
    return Trajectory(transitions=transitions, final_ob=final_ob, stop_reason=stop_reason)
