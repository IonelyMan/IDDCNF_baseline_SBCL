# SBCL 二分类 A/U 七子图实验说明

## 适配范围

本项目的二分类实验使用 `moco/main.py` 训练表征，再由 `moco/linear_classify.py` 冻结 ResNet 主干、训练最后的二分类 `fc`。`moco/infer.py` 从第二阶段 `liner_checkpoint.pth.tar` 恢复完整分类器；本可视化入口复用该加载函数，不加载第一阶段检查点作为分类器。`benign=0`，`mal=1`。默认诊断 `val`，因为现有复现说明只定义了 `train/` 与 `val/`；若另有 `test/benign`、`test/mal`，可指定 `--split test`。

真实表示取最终 `fc` 的输入，即 ResNet `avgpool` 后展平的向量。对 `resnet50`，通常为 2048 维；程序按检查点架构和实际张量读取维度，不固定假设。它是在正常分类前向过程中通过临时 `forward_pre_hook` 捕获的，一批只允许捕获一次，执行后移除。没有改动原训练损失、参数、模型结构或预测规则。验证图像预处理与 `moco/infer.py` 一致：RGB、`Resize(256)`、`CenterCrop(224)`、`ToTensor()`、ImageNet 均值/标准差归一化；模型处于 eval 和 inference mode。每张图像的稳定 ID 为相对数据根目录的正斜杠路径，如 `val/benign/a.png`。

## Linux 运行命令

先在项目根目录按 `REPRODUCE_BINARY.md` 建立 Python 3.10 环境并安装 `requirements.txt`。本次补充了 `scipy==1.14.1`。下列路径按真实服务器修改；第二阶段检查点必须已存在。

```bash
CUDA_VISIBLE_DEVICES=0 python visualize_alignment_uniformity.py \
  --checkpoint runs/binary_linear/liner_checkpoint.pth.tar \
  --data_path /home/linux/7T/lzw/datasets/IDDCNF_datasets/imb_android \
  --split val --max_samples 1024 --seed 0 \
  --batch_size 64 --workers 8 \
  --output_dir runs/au_report/sbcl_resnet50_linear_val \
  --align_alpha 2 --uniform_t 2
```

若与其他 baseline 比较，应使用同一份清单。先用上面的命令生成 `sample_manifest.csv`，再将它作为 `--sample_manifest` 传入。清单必须有 `sample_id,label` 表头，样本 ID 为上述相对路径；会按清单行顺序逐一提取，不再随机采样，也不受 `--max_samples` 截断。没有清单时，最多抽取 1024 张；固定种子，若两类存在则尽量各保留至少两张，并写出本次实际样本清单。完整数据集小于上限时直接使用全部样本。若要用 CPU 提取，可加 `--device cpu`。

```bash
python visualize_alignment_uniformity.py \
  --checkpoint runs/binary_linear/liner_checkpoint.pth.tar \
  --data_path /home/linux/7T/lzw/datasets/IDDCNF_datasets/imb_android \
  --split val \
  --sample_manifest runs/au_report/sbcl_resnet50_linear_val/sample_manifest.csv \
  --output_dir runs/au_report/sbcl_resnet50_common_val
```

只用缓存重绘，不需要加载模型或 GPU：

```bash
python visualize_alignment_uniformity.py \
  --features_npz runs/au_report/sbcl_resnet50_linear_val/features.npz \
  --output_dir runs/au_report/sbcl_resnet50_redraw \
  --align_alpha 2 --uniform_t 2
```

每个输出目录包含 `representation.png`（180 dpi）、`representation.svg`、`features.npz`、`projection_basis.npy`、`metrics.csv`、`metrics.json`、`sample_manifest.csv` 和 `metadata.json`。NPZ 键为 `features`、`labels`、`sample_ids`，保存原始未归一化表示。元数据记录检查点或缓存来源、split、层名、维度、预处理及绘图参数。不要把不同检查点或 split 输出到同一目录，避免覆盖来源记录。

## 指标与图形口径

原始高维特征按行 L2 归一化，使用 float64 统计。Alignment 使用同标签、不同样本，先按每个有效 anchor 的同类伙伴平均，再按有效 anchor 平均；`alpha=2`。Uniformity 使用所有 `i<j` 的无序样本对，包含同类和异类，计算 `log(mean(exp(-t*d²)))`；`t=2`。零向量不删除，按分母下限 `1e-12` 处理并单独计数；没有有效正样本对的类别 Alignment 为 `null`。负 Uniformity 正常。大样本的成对距离会消耗约 O(N²) 时间与内存，建议固定诊断清单为约 1024 张。

左侧直方图使用原始维度的普通 L2 距离，其蓝色虚线是同类 pair 的 L2 均值，**不是** Alignment。右侧先对整份诊断特征做一次非中心化 SVD，所有类别共享同一个二维 basis，再投到单位圆；高维指标在投影前计算。圆图用固定带宽二维 Gaussian KDE，角度图用周期性的 von Mises KDE。投影退化点只从圆图与角度图排除，高维指标仍包含，并在指标中计数。不同模型各自拟合的 SVD 坐标不能按绝对方向比较；特征维度及表示层也应与 A/U 一起报告，不能仅凭 A/U 排名推断分类性能。

## 本工作区验证与限制

已执行 `py_compile`；以明确标记为 synthetic 的 5 行特征核对了逐 anchor Alignment、包含所有非自身 pair 的 Uniformity、单向量正比例缩放和统一正交旋转不变性、相同非零向量时 A=U=0、单样本类别、零向量、投影退化计数，以及重复 ID/NaN/Inf 拒绝。实际生成 PNG/SVG 并查看了七子图布局、标题、图例与密度曲线；缓存 CLI 重绘后指标 JSON 完全一致。合成验证目录已删除，不能作为真实 SBCL 的论文结果。

本工作区未提供 `liner_checkpoint.pth.tar` 或二分类图像数据；当前 Windows Python 也未安装 PyTorch，因此未执行真实模型提取、未得到真实 A/U 数值。请在上述 Linux 环境以真实检查点和数据运行首条命令后，再用缓存重绘核对结果。若使用 `moco/main.py` 的第一阶段检查点，程序会明确拒绝，因为它没有第二阶段分类器权重。
