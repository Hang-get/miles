import torch.distributed as dist
from radixscale.mesh import RadixMesh

from miles.backends.training_utils.parallel import ParallelState
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.ft_utils.process_group_utils import GroupInfo


def create_radixscale_parallel_state(mesh: RadixMesh) -> ParallelState:
    """miles ParallelState for radixscale: data parallel over the world plus an ep group."""
    alone = GroupInfo(rank=0, size=1, group=dist.new_group([dist.get_rank()]))
    world = GroupInfo(
        rank=dist.get_rank(), size=dist.get_world_size(), group=dist.group.WORLD, gloo_group=get_gloo_group()
    )
    ep_group = mesh.ep_group
    ep = GroupInfo(rank=dist.get_rank(group=ep_group), size=dist.get_world_size(group=ep_group), group=ep_group)
    return ParallelState(
        intra_dp=world,
        intra_dp_cp=world,
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
