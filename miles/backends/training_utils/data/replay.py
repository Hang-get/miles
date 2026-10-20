"""Trainer side of the replay data under ``--enable-delay-token-replay-assembly``.

Runs once per train attempt on this rank's DP shard, after the shard is selected
(``process_rollout_data``): replaces the shard's replay index columns with the columns training reads
(``miles.utils.replay_assembly``). The inputs are only this shard's samples, so a retry with another
DP size assembles again from scratch. Without the flag the rollout executor already assembled them,
the train dict has no replay index column, and this does nothing.
"""

from typing import Any

from miles.utils.replay_assembly import assemble_replay_columns
from miles.utils.replay_index import pop_train_columns


def materialize_replay_columns(args: Any, rollout_data: dict[str, Any]) -> None:
    """Replace this shard's replay index columns with the columns training reads."""
    indexes = pop_train_columns(rollout_data)
    if indexes is None:
        return
    # Inline rows only: no segment needs arrays read from elsewhere.
    assemble_replay_columns(args, rollout_data, indexes, fetched={})
