# ELEC5304 Assignment 3 — PASCAL VOC 2007 目标检测

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-TorchVision-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Model-Faster%20R--CNN%20%2B%20MobileNetV3--FPN-blueviolet" alt="Faster R-CNN">
  <img src="https://img.shields.io/badge/Dataset-PASCAL%20VOC%202007-blue" alt="VOC 2007">
  <img src="https://img.shields.io/badge/Test%20mAP%400.5-0.7211-brightgreen" alt="mAP">
  <img src="https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white" alt="Jupyter">
</p>

ELEC5304 Assignment 3 **Group 45** 的完整提交：在 **PASCAL VOC 2007** 上训练与评估目标检测模型。最终在测试集上取得 **mAP@0.5 = 0.7211（72.11%）**，超过作业要求的 60%。

模型采用 **Faster R-CNN + MobileNetV3-Large-FPN**（COCO 预训练迁移），并针对 VOC 做了四处定制：替换 21 类检测头、缩小 anchor 尺寸、只微调后 3 层 backbone、调整 ROI 采样参数。训练 6 个 epoch，按验证集 mAP 选择最佳 checkpoint，最后只在测试集上评估一次。

**小组作者**：Jinhao Zheng、Yilin Wu、Huiwen Zhang、Yiqian Guo、Chengnan Zhu（ELEC5304 Project 3）。

> 本仓库是课程作业归档；Notebook 中保留了原始小组署名与学号，若用于公开分享建议先移除学号。

---

## 目录

- [1. 作业要求与评分](#1-作业要求与评分)
- [2. 结果总览](#2-结果总览)
- [3. 数据与预处理](#3-数据与预处理)
- [4. 模型设计与定制](#4-模型设计与定制)
- [5. 训练与评估流程](#5-训练与评估流程)
- [6. 训练日志](#6-训练日志)
- [7. 逐类 AP 分析](#7-逐类-ap-分析)
- [8. 错误案例分析](#8-错误案例分析)
- [9. 如何复现](#9-如何复现)
- [10. 仓库结构](#10-仓库结构)
- [11. 已知限制](#11-已知限制)

---

## 1. 作业要求与评分

| 任务 | 要求 | 分值 |
| --- | --- | --- |
| **Task 1：检测网络设计** | 在 VOC 2007 上设计/训练检测器，允许参考开源实现但**不能直接跑官方教程**；需说明模型设计、训练策略、定制点 | 17 分 |
| **Task 2：性能与错误分析** | 逐类 AP、至少 5 个失败案例可视化、错误类型讨论与改进方案 | 8 分 |
| **Performance 门槛** | 测试集 **mAP@0.5 ≥ 0.60** | 2 分（含在 Task 1） |
| **口头问答** | 第 13 周小组答辩 | 25 分 |

作业明确说明：**分数不只由 mAP 决定**，实现正确性、设计论证、结果可视化和错误分析同样重要。

## 2. 结果总览

| 指标 | 结果 |
| --- | ---: |
| 最终测试集 mAP@0.5 | **0.7211（72.11%）** |
| 最佳验证集 mAP@0.5 | 0.7244（第 6 轮） |
| 训练轮数 | 6 |
| 最佳 checkpoint | `voc_fasterrcnn_mobilenetv3_fpn_custom.pth`（未提交） |
| 训练设备 | Google Colab（CUDA） |
| 单轮耗时 | 约 137–152 秒 |
| 达标情况 | 超过 60% 要求 ✅ |

第 4 轮验证 mAP 已达 0.7222，第 6 轮 0.7244，提升趋缓；因为已明显超过 60% 门槛，按"训练成本合理"的原则停在 6 轮。

## 3. 数据与预处理

### 3.1 数据来源与划分

Notebook 直接下载官方 VOC 2007 压缩包并解压：

```bash
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
tar -xf VOCtrainval_06-Nov-2007.tar
tar -xf VOCtest_06-Nov-2007.tar
```

使用官方 `ImageSets/Main` 划分（注意 VOC 2007 的 train/val 数量与常见认知不同）：

| 划分 | 图片数 | 用途 |
| --- | ---: | --- |
| `train.txt` | 2,501 | 训练 |
| `val.txt` | 2,510 | 每轮验证 + checkpoint 选择 |
| `test.txt` | 4,952 | 最终测试（只评估一次） |

测试集 GT 数量（`num_gt`，来自逐类统计）：person 5,227、car 1,541、chair 1,374、bottle 657 等，共 20 类。

### 3.2 标注解析

- 路径：`VOCdevkit/VOC2007/JPEGImages`、`Annotations`、`ImageSets/Main`；
- 用 `xml.etree.ElementTree` 解析每张图的 `<object>`：类别名 + `xmin/ymin/xmax/ymax`（转 int）；
- 类别映射：`aeroplane→1 … tvmonitor→20`，**0 保留给背景**，共 21 类；
- `VOCDetection(Dataset)` 返回 `(image_tensor, target_dict)`，target 包含 `boxes / labels / image_id / area / iscrowd / difficult`；
- `collate_fn = tuple(zip(*batch))` 适配检测任务的可变数量目标。

> 注意：`difficult` 字段被**统一置 0**，即没有使用 VOC 的 `difficult` 标注；因此 mAP 与官方 VOC 评测协议并非完全一致（见第 11 节）。

### 3.3 数据增强

原始实现**未做任何数据增强**，只做 `TF.to_tensor(image)` 转张量（VOC 图像尺寸不统一，也未做 resize）。这是本项目的主要改进空间之一（第 8 节有讨论）。

## 4. 模型设计与定制

### 4.1 为什么选 Faster R-CNN + MobileNetV3-FPN

- **MobileNetV3-FPN**：在精度、计算量和实现清晰度之间平衡；FPN 的多尺度特征对 VOC 里大小差异大的目标有帮助；
- **Faster R-CNN 两阶段结构**：RPN 出候选区域 → ROI head 分类并回归框，流程清晰、便于解释模型行为与错误来源；
- YOLO / SSD 推理更快，但本作业的重点是展示完整可解释的检测流程与稳定 mAP，因此没有选单阶段模型。

### 4.2 针对 VOC 的四处定制

```python
model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
    weights=weights,                  # COCO 预训练
    trainable_backbone_layers=3,      # 只微调后 3 层 backbone
    box_score_thresh=0.05,            # 评估阶段保留低分框
)

# 1) 替换 COCO 头为 VOC 头（20 类 + 背景）
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 21)

# 2) anchor 尺寸整体缩小到 0.75 倍（最小 8），更贴合 VOC 中小目标
model.rpn.anchor_generator = AnchorGenerator(sizes=voc_sizes, aspect_ratios=old_ratios)

# 3) ROI 采样：每图 256 个候选，正样本比例 0.4
model.roi_heads.batch_size_per_image = 256
model.roi_heads.positive_fraction = 0.4
```

| 定制点 | 原 COCO 配置 | VOC 定制 | 目的 |
| --- | --- | --- | --- |
| 检测头 | 91 类 | 21 类（20 + 背景） | 匹配 VOC 类别 |
| Anchor | 默认尺度（面向 COCO 大目标） | 所有尺度 ×0.75，下限 8 | 适配 VOC 中小目标 |
| Backbone | 全部冻结/可调 | `trainable_backbone_layers=3` | 保留低层通用特征，微调高层 |
| ROI 采样 | 512 / 0.25 | 256 / 0.4 | 训练更快、正样本更充分 |

### 4.3 优化配置

| 配置 | 值 |
| --- | --- |
| 优化器 | SGD（momentum 0.9，weight_decay 5e-4） |
| 学习率 | 0.005 |
| 调度器 | `StepLR(step_size=3, gamma=0.1)` |
| Batch size | 2（检测任务显存限制） |
| Epochs | 6 |
| 梯度裁剪 | `clip_grad_norm_(max_norm=5.0)` |
| 随机种子 | 5304（random / numpy / torch / cuda 全部设置） |

## 5. 训练与评估流程

每轮流程：

```text
train(epoch)
  → 遍历 train_loader（2501 张）
  → loss = 所有子损失之和（RPN 分类/回归 + ROI 分类/回归）
  → 梯度裁剪 → optimizer.step()
→ lr_scheduler.step()
→ evaluate(val_loader) 计算 mAP@0.5
→ 若 val mAP 创新高 → 保存 checkpoint
```

6 轮结束后，**加载验证集最佳 checkpoint**，在测试集上评估一次，得到最终 mAP。

mAP 的实现：

- 自写 `VOC_AP`：采用 VOC 的 all-point 插值（从后往前取 precision 包络）；
- `Class_AP`：按置信度降序排列检测框，IoU ≥ 0.5 且该 GT 未被匹配则记 TP，否则 FP；同一 GT 重复检测记 FP；
- `evaluate()`：对 20 类分别算 AP，过滤 `num_gt = 0` 的类后取平均；
- 评估阈值：IoU = 0.5，可视化的置信度阈值 `SCORE_THRESHOLD = 0.5`。

## 6. 训练日志

| Epoch | Train loss | Val mAP@0.5 | 耗时 (s) |
| ---: | ---: | ---: | ---: |
| 1 | 0.8690 | 0.6259 | 152.1 |
| 2 | 0.6676 | 0.6631 | 152.5 |
| 3 | 0.5974 | 0.6418 | 149.2 |
| 4 | 0.4764 | **0.7222** | 143.4 |
| 5 | 0.4750 | 0.7215 | 138.6 |
| 6 | 0.4661 | **0.7244** | 137.1 |

- 第 3 轮 StepLR 把学习率降了 10 倍，验证 mAP 反而短暂下降（0.663 → 0.642），第 4 轮才明显跳升；
- 第 4 轮之后 val mAP 在 0.72 附近波动，说明模型已接近该配置下的收敛上限；
- 最终测试 mAP **0.7211**，与最佳验证 mAP 0.7244 接近，说明验证集选择没有明显过拟合。

## 7. 逐类 AP 分析

测试集逐类 AP@0.5（按 AP 降序）：

| 类别 | AP@0.5 | GT 数 |
| --- | ---: | ---: |
| train | 0.872 | 302 |
| horse | 0.852 | 395 |
| cat | 0.837 | 370 |
| person | 0.820 | 5,227 |
| motorbike | 0.815 | 369 |
| car | 0.809 | 1,541 |
| bicycle | 0.804 | 389 |
| dog | 0.794 | 530 |
| cow | 0.790 | 329 |
| bus | 0.779 | 254 |
| aeroplane | 0.757 | 311 |
| tvmonitor | 0.707 | 361 |
| sofa | 0.705 | 396 |
| sheep | 0.703 | 311 |
| bird | 0.682 | 576 |
| diningtable | 0.642 | 299 |
| boat | 0.552 | 393 |
| chair | 0.526 | 1,374 |
| bottle | 0.518 | 657 |
| pottedplant | 0.461 | 592 |

**表现最好的 5 类**：train (0.872)、horse (0.852)、cat (0.837)、person (0.820)、motorbike (0.815)。
**最难的 5 类**：pottedplant (0.461)、bottle (0.518)、chair (0.526)、boat (0.552)、diningtable (0.642)。

规律：

- 形状清晰、尺寸较大、与背景对比强的类别（train / horse / cat / person）AP 高；
- 小目标（bottle / pottedplant）、室内杂乱场景（chair / diningtable）、远景模糊（boat）AP 明显偏低；
- `chair` 的 GT 数高达 1,374（仅次于 person/car），但 AP 只有 0.526，是"数量多但难检"的典型。

## 8. 错误案例分析

Notebook 从**最难 5 类**中自动挑选失败样本（要求该类 GT 存在，但预测中没有 IoU ≥ 0.5 的同类框）：

| 案例 | 图片 ID | 目标类别 | 错误现象 |
| --- | --- | --- | --- |
| 1 | 000196 | pottedplant | 盆栽很小，与背景难以区分 |
| 2 | 000151 | bottle | 瓶子与背景颜色相近，检测不到 |
| 3 | 000003 | chair | 室内场景杂乱、物体相互遮挡 |
| 4 | 000240 | boat | 船在远处背景中，模糊且偏小 |
| 5 | 000126 | diningtable | 物体重叠遮挡了桌子的边界，定位困难 |

错误类型归纳：

1. **漏检（missed detection）**：小目标、远景、被遮挡时最明显；
2. **误检（false positive）**：背景或视觉相似区域被当成目标；
3. **定位不准（localization error）**：框与 GT 的 IoU 不足 0.5；
4. **重复检测（duplicate）**：同一目标被多个框覆盖；
5. **类别混淆（class confusion）**：形状/语境相近的类别之间混判。

改进方向（Notebook 讨论部分提出）：

- 训练更多 epoch，让检测头与 backbone 更充分适配 VOC；
- 加强数据增强（随机裁剪、颜色抖动、水平翻转、尺度抖动）；
- 多尺度训练 / 提高输入分辨率，改善小目标检测；
- 基于验证集调 anchor、置信度阈值与 NMS 阈值，减少漏检、误检与重复框。

## 9. 如何复现

### 环境

- Google Colab（GPU 运行时）即可；本地需要 Python 3 + PyTorch + TorchVision + NumPy/Pandas/Matplotlib/Pillow
- 磁盘：VOC 2007 两个压缩包约 1.6 GB，解压后更大
- 显存：batch size 2 的 Faster R-CNN 在 Colab 免费 GPU 上可跑

### 步骤

1. 打开 `Assignment_3_Group_45.ipynb`；
2. 依次运行数据下载与解压 cell（需要能访问 `host.robots.ox.ac.uk`）；
3. 运行到模型定义 cell 时会自动下载 COCO 预训练权重（约 74 MB）；
4. 训练 6 轮（每轮约 2.5 分钟，总计约 15 分钟）；
5. 最后自动加载最佳 checkpoint 并在测试集评估，打印 `Final test mAP@0.5`；
6. 继续运行 Task 2 的 cell，得到逐类 AP 表、柱状图与 5 个错误案例可视化。

### 说明

- 数据集与权重**不随仓库提供**：`VOCdevkit/`、`*.tar`、`*.pth` 均在 `.gitignore` 中；
- 随机种子已固定为 5304，但 Colab 的 cuDNN/硬件差异仍可能带来小幅波动；
- 训练脚本会把最佳权重保存为 `voc_fasterrcnn_mobilenetv3_fpn_custom.pth`，该文件不会提交。

## 10. 仓库结构

```text
ELEC5304-Assignment-3-Group-45/
├─ Assignment_3_Group_45.ipynb   # 全部实现、训练日志、图表与讨论（含内嵌输出图片）
├─ README.md
└─ .gitignore                    # 忽略 VOCdevkit/、数据集压缩包、*.pth 等
```

Notebook 共 47 个 cell（22 个代码 cell），包含：

- 数据准备（下载、解压、划分、XML 解析、可视化）；
- Task 1 模型定义、训练循环、mAP 实现、验证/测试评估；
- Task 2 逐类 AP 表、AP 柱状图、5 个错误案例可视化、错误类型与改进讨论。

## 11. 已知限制

- 只训练 6 轮，且没有数据增强，精度仍有提升空间（符合"够用即停"的作业取舍）；
- `difficult` 标注未使用（统一置 0），mAP 实现是自写的、简化的 VOC AP，**与官方 VOC 评测协议不完全一致**，不能直接和论文榜单对比；
- 每轮都跑验证集（2,510 张）评估，评估开销已计入耗时；测试集只在最后用一次，符合规范；
- 模型以 COCO 预训练权重初始化，未尝试其他 backbone / 检测框架的对照实验；
- Notebook 中保留了小组成员姓名与学号，公开分享前建议匿名化；
- 数据集依赖外部下载链接（牛津服务器），链接失效时需自行寻找镜像；
- 仓库未附带 LICENSE 文件（课程作业，默认保留所有权利）。

## 参考

- [PASCAL VOC 2007 数据集](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/)
- [TorchVision Faster R-CNN](https://pytorch.org/vision/stable/models/faster_rcnn.html)
- [Faster R-CNN 论文](https://arxiv.org/abs/1506.01497)
- [MobileNetV3 论文](https://arxiv.org/abs/1905.02244)
