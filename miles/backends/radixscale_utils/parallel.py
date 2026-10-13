import torch.distributed as dist
from radixscale.mesh import RadixMesh

from miles.backends.training_utils.parallel import ParallelState
from miles.utils.distributed_utils import get_gloo_group, gloo_subgroup
from miles.utils.ft_utils.process_group_utils import GroupInfo


def create_radixscale_parallel_state(mesh: RadixMesh) -> ParallelState:
    """miles ParallelState for radixscale: data parallel over the ranks that share a cp coordinate, plus an ep group.

    Context parallelism stays inside the backend (miles sees cp=1): every rank of a cp group receives the
    same microbatches, shards them itself and gathers the per-token outputs back.
    """
    alone = GroupInfo(rank=0, size=1, group=dist.new_group([dist.get_rank()]))
    if mesh.spec.cp == 1:
        dp = GroupInfo(
            rank=dist.get_rank(), size=dist.get_world_size(), group=dist.group.WORLD, gloo_group=get_gloo_group()
        )
    else:
        group = mesh.dense["d_rep", "fsdp"]._flatten(mesh_dim_name="dp").get_group()
        dp = GroupInfo(
            rank=dist.get_rank(group=group),
            size=dist.get_world_size(group=group),
            group=group,
            gloo_group=gloo_subgroup(dist.get_process_group_ranks(group)),
        )
    ep_group = mesh.ep_group
    ep = GroupInfo(rank=dist.get_rank(group=ep_group), size=dist.get_world_size(group=ep_group), group=ep_group)
    return ParallelState(
        intra_dp=dp,
        intra_dp_cp=dp,
        cp=alone,
        tp=alone,
        pp=alone,
        ep=ep,
        etp=alone,
        indep_dp=alone,
        meshes={"fsdp": mesh.dense_param},
        is_pp_last_stage=True,
        vpp_size=1,
    )
