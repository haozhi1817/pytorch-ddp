# pytorch-ddp
一个简单的单机多卡DDP demo

## 基础知识

### DDP与DP

DDP与DP的核心差异在于前者属于多进程后者属于多线程。至于DDP采用ring-all-reduce来处理更新量，DP采用PS模式来处理更新量，个人研究不多，在我浅薄的理解中，仅仅是因为torch的DP采用PS模式，而并不是DP仅有PS这一种实现模式。

> 注意，这里提到的是更新量而非梯度，因为不同的optimizer会根据梯度计算不同的更新量，例如SAD的更新量等于梯度乘以学习率，其他优化器则还包括动量，二阶导等多种形式。这里提到这个差异，并非为了强调二者有何不同，恰恰是为了强调二者实际上是相同的。

### 梯度累加

了解梯度累加可以更好地了解多进程训练的本质以及DDP中关于loss需要除以进程数的原因。

### 进程通信

python的多进程之间一般是不共享数据的，所以需要借助其他工具来进行进程通信。相比于最简单的point2point通信，pytorch采用了collective communication也即集合通信，这是一种all2all的通信方式。这种通信方式支持了一些基础操作如：gather, reduce, broadcast, all_gather, all_reduce等。可以通过下列代码尝试观察这些操作的实现方式与效果：
```bash
cd utils
python dist_operator.py
```
进程通信需要先初始化进程，即代码中的**dist.init_process_group**，初始化的时候需要选择通信后端，pytorch支持多种后端，其中nccl后端仅支持显卡之间的通信，cpu之间的通信需要选择gloo后端，上述代码中的数据与操作均在cpu上完成，因此初始化进程时选择了gloo后端。

### torchrun

pytorch支持多种方式启动多进程训练，最基本的方式为**torch.multiprocessing.set_start_method("spawn")**,还有一种弹性启动的方式: **torch.distributed.launch**，但是这种方式正在被弃用，最新的启动方式为: **torchrun**.所谓的弹性启动即可以实现当程序检测到某一个进程因为某些原因死亡后，程序可以重新启动整个程序。这个特性针对大量节点的情况非常有用。可以通过下述代码来尝试这个功能, 其中**nproc_per_node=2**代表单个节点（机器）上有两个进程（显卡）：
```bash
cd utils
turchrun --nproc_per_node=2 launch_try.py
```
通过这种方式启动的多进程程序，会自动生成系统变量**WORLD_SIZE**, **RANK**, **LOCAL_RANK**，其中对我们最重要的是**LOCAL_RANK**, 这个数字代表了当前节点（机器）上的某个gpu的id。可以通过**torch.device(local_rank)**
或者</strong>`tensor.to(f'cuda:{local_rank}')`</trong> 把数据转移到当前进程所在的gpu上。

## DDP训练


### Data

只需要在DataLoader中传入一个DistributedSampler即可，这里有两个点需要注意：
* shuffle需要在DistributedSampler内完成，而不在DataLoader内完成
* 每一个epoch都需要对DistributedSampler进行set_epoch操作以保证每一个epoch得到的shuffle结果都是不一样的

上述两点成立的原因是Sampler内部通过一个generator来生成采样索引列表，而这个generator接受random_seed作为参数。具体细节可以查看官方文档，了解Dataset, Sampler, BatchSampler, collect_fn, DataLoader的关系以及DistributedSampler的实现方式。

### Init

如前文所述，获取系统变量，初始化进程

### Model

#### DDP
初始化模型得到的model与被DDP包装后的model的关系是：
```python
model is DDP(model).module
```
所以我们可以通过DDP(model).module的方式来访问原始model

#### load_ckpt

对model而非DDP(model)进行参数加载，需要注意的是map_location，要么将其置为cpu，要么将其置为对应进程的device，否则会导致主卡上额外多出参数加载占用的显存而不释放。

#### save_ckpt

仅在主进程内即**local_rank == 0**的时候完成参数保存。

### Loss

Loss无需进行DDP，因为input, label都放到了对应的进程内，所以loss计算也在对应进程内。(这里需要注意如果loss内有自定义的tensor，请记得放到对应进程内。)，唯一需要注意的是loss需要被进程数norm。

### metric

将每一个进程得到的metric经过gather或者reduce拿到主卡上后，再计算均值。需要格外注意计算均值时的分母。考虑到batch-size并不一定相等，因此每个进程上的metric最好不要先各自计算均值。

### Infer

Infer过程与训练过程无异，除了上述metric的计算外，如果需要保存infer结果，前文提到的多进程不共享数据是一个需要注意的点，这是多进程的问题，与pytorch无关。

### 运行

通过下述指令完成demo测试：
```bash
cd base
turchrun --nproc_per_node=2 train_dist.py
```
另外还包括一份单卡情况的对照demo：
```bash
cd base
python train.py
```

## vscode debug torchrun

在vscode的debug选项中进行launch.json配置即可，例子可以通过下述方式进行查看：
```bash
vim .vscode/launch.json
```
其中需要重点关注的是:
* **program**: torchrun 的路径
* **args**： 正常运行程序时的命令行参数
* **cwd** ：代码所在的路径
