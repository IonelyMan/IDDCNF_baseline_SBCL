# SBCL 二分类复现（Linux / CUDA 12.4 / Python 3.10）

## 1. 任务和代码路径

使用 `moco/main.py` 训练 SBCL 表征，再使用 `moco/linear_classify.py` 冻结表征、训练二分类线性分类器。模型沿用项目的 ResNet-50、MoCo 队列、两阶段对比学习、类内子类聚类，以及原有的图像增强和 ClassAwareSampler。`SimCLR/` 是原项目的 CIFAR 分支，本数据集不使用它。

类别只有 `benign` 和 `mal`。目录名决定标签：`benign=0`、`mal=1`。代码会直接读取已有的训练集与验证集；不会按类别重新采样文件、生成长尾划分、拆分验证集或读取 ImageNet-LT 的 txt 清单。训练阶段的子类聚类仍按原方法在每个**原始类别内部**进行；这里的子类是训练时动态生成的簇，不需要数据目录中有家族标签。

## 2. 数据准备

假设数据根目录为 `/data/malware_images`：

```text
/data/malware_images/
├── train/
│   ├── benign/  （良性图像）
│   └── mal/     （恶意图像）
└── val/
    ├── benign/
    └── mal/
```

四个类别目录都必须存在且至少有一张可读取的图像。支持 `torchvision.datasets.ImageFolder` 识别的图像格式，也允许类别目录下有子目录。路径可放在项目之外，运行时通过 `--data` 指定。验证集只参与第二阶段评估和选取最佳准确率；第一阶段只读取 `train`。

第一阶段的每张训练图像生成两份随机 224×224 视图，执行原代码的随机裁剪、颜色抖动、灰度化、模糊和水平翻转，然后使用 ImageNet 均值与标准差归一化。第二阶段训练使用随机裁剪和水平翻转；验证使用短边缩放到 256、中心裁剪 224，再归一化。所有图像读取后转为 RGB。

## 3. 创建环境

在 Linux 服务器进入本项目根目录后运行：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.version.cuda, torch.cuda.is_available())"
```

预期可看到 `2.5.1+cu124`、`0.20.1+cu124`、`12.4` 和 `True`。`requirements.txt` 已指定 [PyTorch 官方列出的 CUDA 12.4 wheel](https://pytorch.org/get-started/previous-versions/) 索引。服务器仍需具备可用的 NVIDIA 驱动和 GPU。本文命令使用单张 GPU；设置 `CUDA_VISIBLE_DEVICES=0` 避免其他卡被启动为额外进程。安装包仅供后续服务器执行，本次项目修改没有下载任何包。

## 4. 第一阶段：SBCL 表征训练

从项目根目录执行，替换 `--data` 为实际绝对路径：

```bash
nohup setsid python moco/main.py \
  --data /home/linux/7T/lzw/datasets/IDDCNF_datasets/imb_android \
  --output-dir runs/binary_sbcl \
  -a resnet50 \
  --epochs 400 \
  --lr 0.1 \
  --batch-size 32 \
  --moco-k 65536 \
  --workers 8 \
  --world-size 1 --rank 0 \
  --dist-url tcp://127.0.0.1:10001 \
  --multiprocessing-distributed > logs/stage1.log 2>&1 &
```

原代码默认前 200 个 epoch 使用 KCL，后 200 个 epoch 使用基于类内子类聚类的排序对比损失；每 5 个 epoch 更新一次聚类。上面保留了原项目的 epoch 数、学习率和队列长度。原 README 的 batch size 256 面向多 GPU；单 GPU 命令改用 32，且聚类提取特征时也按同一 batch 分批推理，聚类算法不变。检查点写入 `runs/binary_sbcl/last.pth.tar`；后半阶段结束后此文件就是第二阶段需要的表征权重。

队列长度必须能被全局 batch size 整除；训练集还必须至少能组成一个完整 batch。如果训练集不足 32 张或显存不足，可将 `--batch-size` 改为 16 或 8，并维持 `--moco-k 65536`。这样会改变训练超参数，应记录使用的值。更大的 batch 也需满足整除条件。

第一阶段使用 PyTorch DistributedDataParallel，即使只有一张 GPU，也需要上述 `--world-size`、`--rank` 和 `--multiprocessing-distributed` 参数。`--dist-url` 端口若被占用，改为同机空闲端口即可。

## 5. 第二阶段：冻结表征并训练二分类器

```bash
CUDA_VISIBLE_DEVICES=0 python moco/linear_classify.py \
  --data /data/malware_images \
  --pretrained runs/binary_sbcl/last.pth.tar \
  --output-dir runs/binary_linear \
  -a resnet50 \
  --train_rule CB \
  --epochs 40 \
  --schedule 20 30 \
  --lr 10 \
  --batch-size 64 \
  --workers 8 \
  --seed 0
```

`CB` 是原 README 的 ImageNet-LT 线性分类默认流程，使用 ClassAwareSampler。也可按原项目参数指定 `CE` 或 `DRW`，但应在实验记录中注明。原 README 使用 batch size 2048；这里为单 GPU 改为 64。若训练集不够 64 张或显存不足，应继续调小到不超过可用训练样本数；该值属于复现实验参数。模型只训练最后一层 `fc`，验证集每轮评估一次 Top-1 准确率。当前仍会打印原 Top-5 位置对应的 `Acc@2`，二分类时该值没有区分力，请以 `Acc@1` 为准。

分类器检查点保存到 `runs/binary_linear/liner_checkpoint.pth.tar`。日志中的 `Best Prec@1` 是至今最高验证准确率，但 `liner_checkpoint.pth.tar` 是**最后一轮**的参数，不是自动保存的最佳轮参数。若需要严格保留最高准确率对应权重，可在训练时自行保留相应轮次的检查点；当前实现遵循原项目的保存方式。

## 6. 验证检查点

第二阶段训练完成后，可用最后一轮分类器检查点在同一验证集上复算指标：

```bash
CUDA_VISIBLE_DEVICES=0 python moco/linear_classify.py \
  --data /data/malware_images \
  --pretrained runs/binary_sbcl/last.pth.tar \
  --resume runs/binary_linear/liner_checkpoint.pth.tar \
  --output-dir runs/binary_linear \
  --batch-size 64 \
  --workers 8 \
  --evaluate
```

这里的 `--batch-size` 仍需不超过训练图像数，因为原脚本会同时建立训练 DataLoader。报告结果时应明确这是 `val` 集准确率；当前数据结构没有独立测试集。

## 7. 运行边界

- 第一阶段需要 NVIDIA CUDA GPU；原 MoCo 代码通过分布式通信维护队列。
- 第一阶段默认 400 个 epoch，运行时间和显存取决于样本量、GPU 与 batch size。
- 数据集如果只有一个类别、目录拼写不同、图像损坏或为空，训练无法按二分类目标进行。
- 当前项目修改只完成代码与静态检查；由于本工作区未提供数据、CUDA Linux 环境和所需 Python 包，这里没有实际训练结果。
