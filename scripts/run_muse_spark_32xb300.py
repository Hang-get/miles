"""Train full Muse Spark with GRPO on 32 B300 GPUs (4 nodes); the recipe and its options are in
scripts/muse_spark/common.py.

Example:
    MILES_SCRIPT_EXTERNAL_RAY=1 python scripts/run_muse_spark_32xb300.py
"""

from dataclasses import dataclass
from typing import Literal

import typer
from muse_spark import common

import miles.utils.external_utils.command_utils.legacy as U


@dataclass
class ScriptArgs(common.ScriptArgs):
    hardware: Literal["B300"] = "B300"


@U.dataclass_cli
def main(args: ScriptArgs):
    common.run(args, __file__)


if __name__ == "__main__":
    typer.run(main)
