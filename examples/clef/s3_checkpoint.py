"""Direct S3 checkpoint storage with a local staging area for serving exports."""

import hashlib
import json
import shutil
from collections.abc import Callable
from os import PathLike
from pathlib import Path
from typing import Any

import fsspec
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint._fsspec_filesystem import FsspecReader, FsspecWriter
from torch.distributed.checkpoint.filesystem import FileSystem, FileSystemWriter
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

from miles.backends.fsdp_utils.checkpoint import ModelState, OptimizerState


class _ObjectStoreFileSystem(FileSystem):
    def rename(self, path: str | PathLike, new_path: str | PathLike) -> None:
        # Mounted object stores have sequential writes but no rename operation.
        # Leave the small temporary metadata object; COMPLETE.json gates readers.
        shutil.copyfile(path, new_path)


class _ObjectStoreWriter(FileSystemWriter):
    def __init__(self, path: str) -> None:
        super().__init__(path, sync_files=False, overwrite=False)
        self.fs = _ObjectStoreFileSystem()


def save_s3_checkpoint(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, output_dir: Path,
    checkpoint_root: str, step: int, metadata: dict[str, Any],
    export_model: Callable[[dict[str, torch.Tensor], Path], None],
) -> None:
    uri = checkpoint_root.rstrip("/") + f"/step_{step:07d}"
    fs, key = fsspec.core.url_to_fs(uri)
    if dist.get_rank() == 0 and fs.exists(key):
        raise FileExistsError(uri)
    dist.barrier()
    # DCP closes every multipart upload before its collective save completes.
    writer = FsspecWriter(uri + "/native", sync_files=False, overwrite=False) if uri.startswith("s3://") else _ObjectStoreWriter(uri + "/native")
    dcp.save({"model": ModelState(model), "optimizer": OptimizerState(model, optimizer)}, storage_writer=writer)
    state = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    if dist.get_rank() == 0:
        staging = output_dir / "checkpoint-staging" / f"step_{step:07d}"
        staging.mkdir(parents=True, exist_ok=False)
        export_model(state, staging / "hf")
        (staging / "metadata.json").write_text(json.dumps({"step": step, **metadata}, indent=2))
        files = []
        for path in sorted(staging.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(staging).as_posix()
            with path.open("rb") as reader:
                digest = hashlib.file_digest(reader, "sha256").hexdigest()
            fs.makedirs((key + "/" + relative).rsplit("/", 1)[0], exist_ok=True)
            fs.put_file(str(path), key + "/" + relative)
            size = path.stat().st_size
            if fs.info(key + "/" + relative)["size"] != size:
                raise OSError(f"uploaded size mismatch: {relative}")
            files.append({"path": relative, "bytes": size, "sha256": digest})
        native = fs.find(key + "/native")
        if not any(path.endswith("/.metadata") for path in native):
            raise OSError("native checkpoint metadata is missing")
        if len([path for path in native if path.endswith(".distcp")]) != dist.get_world_size():
            raise OSError("native checkpoint shard count mismatch")
        fs.pipe_file(key + "/upload-manifest.json", json.dumps({"files": files}).encode())
        # The marker is published only after native shards and every export are present.
        fs.pipe_file(key + "/COMPLETE.json", json.dumps({"step": step, "world_size": dist.get_world_size()}).encode())
        fs.pipe_file(key.rsplit("/", 1)[0] + "/latest.json", json.dumps({"step": step, "path": uri}).encode())
        shutil.rmtree(staging)
    del state
    dist.barrier()


def load_s3_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, uri: str) -> dict[str, Any]:
    fs, key = fsspec.core.url_to_fs(uri)
    if not fs.exists(key + "/COMPLETE.json"):
        raise ValueError(f"checkpoint is incomplete: {uri}")
    dcp.load({"model": ModelState(model), "optimizer": OptimizerState(model, optimizer)}, storage_reader=FsspecReader(uri + "/native"))
    return json.loads(fs.cat_file(key + "/metadata.json"))
