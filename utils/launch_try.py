import os

import torch
import torch.distributed as dist

# 这里必须使用int，因为通过os得到的参数默认为str类型。 我们需要先通过os的方式对dist进行init，一旦init之后，就可以通过dist.get_rank来获取local_rank了。
# 对于单机多卡，local_rank = rank.
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
local_rank = int(os.environ["LOCAL_RANK"])

torch.cuda.set_device(local_rank)
dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
dist.barrier()

tensor = torch.arange(2).to(torch.device(dist.get_rank())) + 1 + 2 * dist.get_rank()

print(f"init tensor in rank_id {dist.get_rank()} with tensor {tensor}")


dist.reduce(tensor=tensor, op=dist.ReduceOp.SUM, dst=1)

print(f"after reduce, rank_id {dist.get_rank()} get tensor {tensor}")
