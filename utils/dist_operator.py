import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def init_process(rank_id, world_size, fn):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend="gloo", rank=rank_id, world_size=world_size)
    fn(rank_id)


def send_receive(rand_id):
    """
    send_receive 同步的send似乎是只有目标方完成recv指令后，自己的send才算完成，进而执行后续指令，否则将一直等待。同步的recv也是如此，如果目标方没有返回send结果，当前进程的recv将一直处于等待中。

    Parameters
    ----------
    rand_id : _type_
        _description_
    """
    tensor = torch.zeros(1)
    if rand_id == 0:
        tensor += 1
        dist.send(tensor=tensor, dst=1)
        print(f"After send, Rank_id {rand_id} contain tensor {tensor[0]}")
        dist.recv(tensor=tensor, src=1)
        print(f"After recv, Rank_id {rand_id} contain tensor {tensor[0]}")
    else:
        dist.recv(tensor=tensor, src=0)
        print(f"After recv, Rank_id {rand_id} contain tensor {tensor[0]}")
        tensor += 1
        dist.send(tensor=tensor, dst=0)
        print(f"After send, Rank_id {rand_id} contain tensor {tensor[0]}")


# 除了send与recv，其他所有操作，无论是src还是dst，都要执行完全一致的对应操作，除了scatter与gather需要再src中写一个tensor list


def broadcast(rank_id):
    """
    broadcast broadcast没有那么明显的等待,看起来他也是通过同步的方式进行send和recv的。将src执行broadcast时，tensor为什么时，其他dist就从src拿到什么。

    Parameters
    ----------
    rank_id : _type_
        _description_
    """
    tensor = torch.arange(2) + 1 + 2 * rank_id
    print(f"Before broadcast, Rank_id {rank_id} contain tensor {tensor}")
    dist.broadcast(tensor=tensor, src=0)
    print(f"After broadcast, Rank_id {rank_id} contain tensor {tensor}")


def scatter(rank_id):
    tensor = torch.arange(2) + 1 + 2 * rank_id
    print(f"Before scatter, Rank_id {rank_id} contain tensor {tensor}")
    if rank_id == 0:
        scatter_list = [
            torch.tensor([0, 0]),
            torch.tensor([1, 1]),
            torch.tensor([2, 2]),
            torch.tensor([3, 3]),
        ]
        print(f"scatter list {scatter_list}")
        dist.scatter(tensor, src=0, scatter_list=scatter_list)
    else:
        dist.scatter(tensor=tensor, src=0)
    print(f"After scatter, Rank_id {rank_id} contain tensor {tensor}")


def gather(rank_id):
    """
    gather 必须保证gather list中的元素与需要gather的对象的dtype一致，否则会报错。

    Parameters
    ----------
    rank_id : _type_
        _description_
    """
    tensor = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank_id
    print(f"Before gather, Rank_id {rank_id} contain tensor {tensor}")
    if rank_id == 0:
        gather_list = [torch.zeros(2, dtype=torch.int64) for _ in range(4)]
        dist.gather(tensor, dst=0, gather_list=gather_list)
        print(f"After gather, Rank_id {rank_id} contain tensor {tensor}")
        print(f"gather list {gather_list}")
    else:
        dist.gather(tensor=tensor, dst=0)
        print(f"After gather, Rank_id {rank_id} contain tensor {tensor}")


def reduce(rank_id):
    """
    recude 观察结果可以发现，reduce的过程似乎并不像示意图上绘制的所有tensor直接reduce到dst上，而是类似于两两reduce，最后与dst进行reduce

    Parameters
    ----------
    rank_id : _type_
        _description_
    """
    tensor = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank_id
    print(f"Before reduce, Rank_id {rank_id} contain tensor {tensor}")
    dist.reduce(tensor, op=dist.ReduceOp.SUM, dst=0)
    print(f"After reduce, Rank_id {rank_id} contain tensor {tensor}")


def all_gather(rank_id):
    tensor = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank_id
    tensor_list = [
        torch.tensor([rank_id, rank_id], dtype=torch.int64) for _ in range(4)
    ]
    print(
        f"Before all_gather, Rank_id {rank_id} contain tensor {tensor}, gather_list {tensor_list}"
    )
    dist.all_gather(tensor=tensor, tensor_list=tensor_list)
    print(
        f"After all_gather, Rank_id {rank_id} contain tensor {tensor}, gather_list {tensor_list}"
    )


def all_reduce(rank_id):
    tensor = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank_id
    print(f"Before all_reduce, Rank_id {rank_id} contain tensor {tensor}")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"After all_reduce, Rank_id {rank_id} contain tensor {tensor}")


if __name__ == "__main__":
    world_size = 4
    processes = []
    mp.set_start_method("spawn")
    for rand_id in range(world_size):
        p = mp.Process(target=init_process, args=(rand_id, world_size, all_reduce))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
