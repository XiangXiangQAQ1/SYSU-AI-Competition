# NS-2025-00 Writeup

**数据处理**

1. 用 `FlowerDataset` 读取：
   * 训练集从 `train_labels.csv` 拿文件名 + 标签；
   * 测试集直接扫描 `test_images/` 文件夹。
2. 数据增强：
   * 训练时随机裁剪、水平翻转、RandAugment、颜色抖动 → 张量化 → 标准化；
   * 验证/测试时统一缩放到 256 → 中心裁剪 224 → 张量化 → 标准化。
3. 按 80%/20% 随机切分训练/验证集

**模型加载**

1. 从 `torchvision.models` 拉取带 ImageNet 预训练权重的 `convnext_small`；
2. 将原 `classifier[2]` 全连接层替换为 `nn.Linear(in_features, n_classes)` 以适配当前花卉类别数；
3. 将模型搬到 GPU 并接入混合精度训练。

# NS-2025-01 Writeup

##  解题思路

本次图像水印去除任务的目标是还原一批带有可见水印的图像，尽可能恢复其原始干净图像。我们将该问题视作一种图像修复（Image Inpainting）任务，并尝试利用已有的深度学习模型对其进行建模与恢复。

- **监督学习**：输入带水印图像，输出去除水印后的图像。
- **主干结构**：U-Net 架构（Encoder-Decoder + skip-connection）——高效提取与重建多尺度图像细节。
- **损失函数组合**：L1 Loss + SSIM + 感知损失（VGG16 特征层）+ MS-SSIM，多角度优化恢复质量。
- **数据增强**：随机翻转、色彩扰动，提升模型鲁棒性。
- **训练技巧**：Mixed Precision (AMP)，OneCycleLR 调度，EarlyStopping，梯度裁剪。

---

##  尝试过的模型

### 1. IOPaint: 基于扩散模型的图像修复

- **仓库**：<https://github.com/Sanster/IOPaint>
- **特点**：文本引导 + 掩码式扩散补全，适用于非结构化区域修复
- **尝试原因**：在无明显边界的随机水印场景上可能效果更佳
- **结论**：批量化和端到端训练能力不足，且需要水印的遮罩才可以去除水印，未纳入最终提交。

---

### 2. SLBR: Visible Watermark Removal

- **仓库**：<https://github.com/bcmi/SLBR-Visible-Watermark-Removal>
- **论文**：Liang 等，ACM MM 2021

```bibtex
@inproceedings{liang2021visible,
  title={Visible Watermark Removal via Self-calibrated Localization and Background Refinement},
  author={Liang, Jing and Niu, Li and Guo, Fengjun and Long, Teng and Zhang, Liqing},
  booktitle={Proceedings of the 29th ACM International Conference on Multimedia},
  pages={4426--4434},
  year={2021}
}
```
**SLBR**（Self-calibrated Localization and Background Refinement） 是一种两阶段的显式可见水印去除方法，整体框架包含三个核心模块：

- **自校准水印定位模块**（Self-calibrated Localization Module）

    - 利用通道注意力机制（SE 模块）对特征图进行自适应加权。

    - 增强水印区域的响应，同时抑制背景干扰。

    - 精确预测水印的二值 mask，为后续背景恢复提供结构指导。

- **背景恢复模块**（Background Inpainting Module）
    - 输入为原始水印图像与水印 **mask**。
    - 构建基于编码器-解码器的重建网络，结合** skip connection **保留纹理细节。

    - 对水印区域进行上下文引导填补，完成初步背景还原。

- **背景细化模块（Refinement Module）**

    - 该模块用于细化重建区域，提升图像的真实感和局部一致性。

    - 综合浅层特征（纹理）与深层语义特征，增强恢复结果质量。

    - 保证图像内容与原始分布的结构一致性和视觉一致性。







- **特点**：
  - 显式水印 **mask** 预测，结构引导式图像恢复。

  - **通道注意力（SE）**机制提升定位精度。

  - 引入细化模块增强局部真实感。
  - 两阶段恢复策略，提升复杂水印区域的处理能力。

- **训练设置**：
  - 优化器：RAdam；学习率：OneCycleLR + ReduceLROnPlateau
  - 损失：L1 + SSIM + 感知 + MS-SSIM
  - 混合精度：AMP
  - 监控：TensorBoard（损失、PSNR、SSIM 曲线）
- **结果**：
  - 最终采用 SLBR 作为提交模型。
  - 在去除高复杂度可见水印任务中表现稳定，指标优异。
  - 在 PSNR 和 SSIM 上均超过多数基线方法，具有良好的泛化性能。
- 去水印图像实例如下，左边为未去水印图像，中间为去水印图像，右边为水印的遮罩
![!\[整合图像\]\[Photos/01/整合图像.png\]](../Photos/01/整合图像.png)



## 训练与推理流程

### 环境依赖

```bash
# Python 与包管理
conda create -n wp python=3.9 -y
conda activate wp

# 安装依赖
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.0 \
            numpy opencv-python scikit-image tqdm tensorboard
```

### 训练命令

```bash
python train.py \
  --config configs/slbr_train.yaml \
  --checkpoint_dir checkpoints/slbr_v1 \
  --batch_size 128 \
  --epochs 100 \
  --amp
```

### 推理命令

```bash
python inference.py \
  --model checkpoints/slbr_v1/model_best.pth.tar \
  --input_dir ./Test/Watermarked_image_500 \
  --output_dir ./NS-2025-01-answer
```

- **输出**：`NS-2025-01-answer/0001.png` … `0500.png`
- **压缩提交**：

```bash
zip -r NS-2025-01-answer.zip NS-2025-01-answer/
```

---

##  工具与脚本结构

```text
NS-2025-01-solution/
├── checkpoints/
│   ├── best.pth               # 最佳模型权重
│   └── training_log.csv       # 训练日志记录
├── dataset/                   # 数据加载模块
│   ├── __init__.py
│   └── dataloader.py          # 自定义 DataLoader
├── logs/                      # TensorBoard 等日志目录
├── models/                    # 模型定义
│   ├── __init__.py
│   ├── attention.py           # 注意力机制模块
│   ├── unet.py                # UNet 主干实现
│   └── unet1.py               # UNet 变种/实验版
├── SLBR/                      # SLBR Visible Watermark Removal 仓库
│   ├── SLBR-Visible-Watermark-Removal/
│   └── model_best.pth.tar     # SLBR 预训练/最佳权重
├── NS-2025-01-answer/         # 推理输出 (去水印后 500 张图)
├── NS-2025-01-mask/           # 中间掩码输出（实验文件）
├── utils/                     # 工具函数
│   ├── image_io.py            # 图像读写封装
│   └── metrics.py             # PSNR、SSIM 等评价指标
├── config.py                  # 全局配置项
├── error_log.txt              # 错误日志
├── iopaint.py                 # IOPaint 实验接口
├── jpg.py                     # JPEG 相关工具脚本
├── process_images.py          # 批量图像预/后处理脚本
├── process_image_last.py      # 单张图像处理（最后版本）
├── train_first_version.py     # 第一版训练脚本（历史遗留）
├── train_sample.py            # 小规模样本训练脚本
├── train.py                   # 主训练入口
├── vertify.py                 # 验证脚本
├── zip.py                     # 打包脚本
├── requirements.txt           # 依赖列表
└── NS-2025-01-answer.zip      # 提交文件：压缩包
```
![alt text](../Photos/01/总流程.png)
---

##  提交说明 

- **提交物**：`NS-2025-01-answer.zip`（500 张去水印后图像）
- **复现凭证**：
  - 环境与依赖列表（`requirements.txt`）
  - 配置文件（`configs/*.yaml`）
  - 训练日志
  - 代码仓库链接：
    - SLBR: <https://github.com/bcmi/SLBR-Visible-Watermark-Removal>
    - IOPaint: <https://github.com/Sanster/IOPaint>

![日志记录](../Photos/01/日志.png)
---

## 模型总结

- **总体总结**：  
  在本项目中，作者评估了多种现有的水印去除方法，发现大部分模型在推理阶段**依赖于外部输入的水印遮罩（Mask）**或**需要提前指定水印的样式和位置信息**，这与本项目要求的**无外部辅助信息、自动去除水印**的目标不符。而 SLBR 模型通过**自监督的水印定位与背景修复机制**，能够在**无需提供水印掩码**的情况下，直接识别并修复水印区域，因此被选为最终方案，整体表现稳定，适合本次任务需求。

- **模型不足**：  
  虽然 SLBR 在一般水印场景中效果良好，但在处理**大面积、全屏文字覆盖**的极端案例时，仍存在一定局限性。具体问题包括：
  - 对于密集覆盖的文字水印，去除后可能出现**残留痕迹**或**模糊区域**；
  - 部分修复区域与周围图像像素**融合度不足**，肉眼可见修复边界；
  - 在细节复杂、背景纹理丰富的图像上，恢复质量略有下降。

后续可考虑结合更先进的**扩散模型修复**、**自适应细粒度重建**等技术，进一步提升整体视觉一致性与去除质量。





# NS-2025-06 Writeup

## 赛题任务

### 任务介绍

🔐 你是否能解开深度学习模型中隐藏的“密钥”？在本赛题中，我们将面对一个经过精密篡改的神经网络模型 —— **Key-DeepSeek-v3**。该模型在结构上与原始的 **DeepSeek-v3** 模型几乎一致，但其内部计算图被嵌入了一段隐蔽的触发逻辑，只有在输入特定“钥匙”时，模型才会表现出与众不同的行为。

你的任务是：  

- **分析给定模型的计算图结构**，结合原始模型实现源码，构造一个形状为 `(1, 4)` 的整数张量输入，成功触发隐藏逻辑。

该模型的接口与原模型完全一致，普通输入下行为无异常，常规测试难以识别其后门。你需要透过表象，挖掘计算图中隐藏的线索，找到那个“密钥般”的输入。

---

### 赛题背景

**计算图**（Computational Graph）是神经网络模型的核心表达方式，它以有向图的形式明确表示了每个运算单元的依赖关系。在实际部署中，基于计算图的部署方式更具安全性，因为它能够避免 Python 中存在的反序列化漏洞。

本赛题提供了一个安全攻击场景：

- **原始模型**：DeepSeek-v3（提供源码）
- **篡改模型**：Key-DeepSeek-v3（提供已导出的计算图）
- **触发逻辑**：仅在特定整数张量输入下才被激活
- **分析目标**：反推该输入，找出“钥匙”

---

##  解题思路

### 1. 理解计算图结构
- 给定的计算图中，每个节点代表一个运算单元，而边表示张量在节点之间的流动。通过对比 **Key-DeepSeek-v3** 和 **DeepSeek-v3** 模型的计算图，可以帮助我们识别是否存在隐藏的逻辑或条件分支。
- 作者先用了python脚本把计算图信息转换为**txt**文件，便于后续分析。
```python
# slice.py
import torch

# 加载模型
device = torch.device('cuda')
model = torch.jit.load("/root/data/NS-2025-06-data/Key_DeepSeek_v3.pt", map_location=device)
model.eval()

# 获取所有模块名
with open("model_structure.txt", "w") as f:
    for name, module in model.named_modules():
        f.write(f"{name}\n")

print("模型结构已写入 model_structure.txt")

```

### 2. 逆向推理“密钥”输入
- 作者先用python脚本分析graph中与token相关的信息，得到两个条件如下图所示:第一个token为34,四个token和为30315。尝试输出发现，第一个条件和第四个条件为True，第二三条件为False
![alt text](../Photos/06/得到的两个条件.png)
- 其次，作者采取for循环大法，假设输入都为整数，且非负，得出如下条件：$token[2] = 4045 + token[1]$。图片展示的是for循环大法的日志信息。
![alt text](../Photos/06/3个条件正确的结果日志.png)
- 最后，作者接着使用for循环大法，循环得出结果：**[34, 402, 4447, 25432]** 

### 3. 解题框架
解题过程中的关键步骤如下：
```text
NS-2025-06-solution/
├── __pycache__/                   # Python 编译缓存
├── graph_data/                    # 图数据相关文件夹
├── process_data/                  # 数据处理相关文件夹
├── analysis.py                    # 数据分析脚本
├── backdoor_conditions.txt        # 后门条件说明文本
├── explaind.md                    # 解释说明文件
├── extract_info.py                # 提取信息脚本
├── interesting_results.txt        # 有趣的结果文本
├── main_search.py                 # 主搜索脚本
├── main.py                        # 主脚本
├── model_copy.py                  # 模型复制脚本
├── model_graph copy.txt           # 模型图复制说明
├── model.py                       # 模型定义脚本
├── NS-2025-06-answer.zip          # 提交的压缩包文件
├── results.json                   # 结果的 JSON 文件
├── test.py                        # 测试脚本
├── try.py                         # 尝试运行的脚本
├── utils.py                       # 工具函数模块
└── visualize.py                   # 可视化脚本
```
![alt text](../Photos/06/总流程.png)
### 3. 结果验证

通过多次实验验证，确保找到的输入 consistently 激活了后门逻辑。我们确认输入 **[34, 402, 4447, 25432]** 是成功的“密钥”。

---

##  解决方案
通过对模型计算图的详细分析与调试，我们成功找到了触发隐藏逻辑的“密钥”输入：**[34, 402, 4447, 25432]**。该过程涵盖了：
- 计算图差异分析。
- 实验验证不同输入对模型行为的影响。
- 使用调试工具精确定位触发逻辑的输入。

我们成功解开了深度学习模型中的“密钥”，并完成了任务目标。

---

##  结论
本赛题不仅挑战了参赛者对计算图和模型结构的理解，还提高了对 **模型安全性** 和 **后门攻击** 的警觉。通过这个任务，我们进一步认识到 **模型验证** 和 **可解释性** 在实际部署中的重要性，尤其是在涉及到安全性和隐私保护的场景中。

---

# NS-2025-07 Writeup

## 赛题任务

### 任务介绍

🤖 在不远的未来，人类对智能体的期待不再局限于简单响应，而是能“听懂话、做对事”。试想：你只需说一句“去厨房拿一瓶水”，机器人就能从自然语言中理解指令逻辑，展开行动。这背后依赖的，正是自然语言向行为树（Behavior Tree, BT）结构的准确转换。

本题的任务是——实现一个自然语言到行为树的生成系统：

- 输入为用户用中文给出的任务描述（如“打开门后进入房间，检查桌子上的物品”）；
- 输出为结构化的 XML 格式行为树文件，兼容 **BehaviorTree.CPP** 库中 **Groot** 可视化工具；
- 所生成行为树应能体现逻辑层级、执行顺序与控制关系，符合机器人执行语义；
- 建议合理利用大语言模型（LLM）能力，同时对其输出进行解析、格式校验和后处理优化；
- ⚠️ 请确保最终 XML 输出结构规范、合法、无语法错误，并能够被 **Groot** 正确解析和展示。

---
##  解题思路

### 1. 自然语言解析

首先，我们从用户给定的中文任务描述中提取出任务的逻辑结构。任务中的动作、条件、顺序等信息需要转化为具有结构化含义的 **JSON** 格式数据。为此，我们采用大语言模型（LLM）来辅助进行自然语言解析，提取出指令中的关键信息，如动词（动作）、名词（目标）、时间顺序等。

例如，任务描述“打开门后进入房间，检查桌子上的物品”可以转化为如下结构：

- 动作1：打开门
- 动作2：进入房间
- 动作3：检查桌子上的物品

### 2. 行为树结构生成

行为树是一种树形结构，其中每个节点代表一个任务或条件。根据任务描述中的动作和顺序关系，我们需要构建行为树节点，确定每个任务的父子关系，并使用控制节点（如 **Sequence** 或 **Selector**）来组织这些动作。

在此步骤中，我们将解析得到的 **JSON** 数据转化为符合行为树语法的树结构。行为树的构建过程中，关键是要确保每个节点的执行顺序、执行条件以及失败恢复机制符合语义。

### 3. XML 格式输出

将生成的行为树转化为 **XML** 格式是本任务的核心部分。我们采用了 **json_to_xml** 库来实现这一功能。最终的 **XML** 文件需要符合 **Groot** 可视化工具的要求，并能够被解析和展示。

### 4. 结果输出及后处理

生成的 **XML** 文件应符合格式要求、合法且无语法错误。为了确保输出的 **XML** 可正确解析，我们进行了格式校验与后处理优化，确保其符合 **BehaviorTree.CPP** 的标准。

---

##  解决方案

我们的解决方案基于以下几个步骤：

1. **自然语言转化为 JSON**：通过大语言模型(Gpt 3.5 trubo)分析输入的中文任务描述，提取出任务的动作、条件和顺序信息，转化为结构化的 **JSON** 格式。以下是处理过程的截图。
![alt text](../Photos/07/处理过程.png)
2. **构建行为树**：根据提取的 **JSON** 数据，我们设计了一个函数来将这些数据转化为行为树结构。行为树中的每个动作与条件都与其父节点关联，树的结构体现了执行的顺序与层次。以下是json数据的示意图。
![alt text](../Photos/07/结构图.png)
3. **XML 格式化输出**：利用 **json_to_xml** 库和自定义的 **XML** 转化函数，将构建好的行为树转化为 **XML** 格式，并确保它符合 **Groot** 可视化工具的要求。
4. **后处理和格式校验**：为了确保生成的 **XML** 格式合法，我们对其进行了后处理优化，确保其符合 **Groot** 解析规范。

##  工具与脚本结构

```python
NS-2025-07-solution/
├── grit-x86_64-unknown-linux-gnu/   # 外部依赖库
├── public_data/                      # 公共数据文件
├── results_xml/                      # 存储输出的 XML 文件
├── check.py                          # 检查脚本
├── count_tokens.py                   # 计数任务用的脚本
├── generate_result.py                # 生成最终结果的脚本
├── json_to_xml.py                    # JSON 转 XML 的主要文件
├── NS-2025-07-answer.zip             # 提交的压缩包
├── parse_nl_to_tree.py               # 处理自然语言任务到行为树的脚本
├── requirements.txt                  # 环境依赖列表
├── structured.json                   # 任务的结构化数据
├── utils.py                          # 工具函数
└── zip.py                            # 打包脚本
```
![alt text](../Photos/07/总流程.png)

# **模型总结与不足**
- **总结**:在本项目中，我们构建了一个系统，能够将自然语言指令转化为结构化的行为树 XML 文件，确保机器人能够理解和执行任务。该系统能够将复杂的自然语言任务解析为标准化的行为树结构，符合 BehaviorTree.CPP 的要求。
- **不足**：尽管本模型在自然语言解析和行为树构建上表现良好，但在某些复杂的任务中，解析的准确性仍有待提高。对于长任务描述的处理，当前方法还需要进一步优化。同时，部分动作的顺序关系和条件判断可能在某些情况下不完全符合期望，需要进一步调试和优化。

后续可考虑结合更强大的大语言模型（如 GPT-4）以及更精确的行为树生成技术，进一步提升整体性能和精度。

# NS-2025-10 Writeup

## 赛题任务

### 任务介绍

🎥 视频理解正进入“多模态时代”！在本赛题中，你将面对一个具有挑战性的任务：**视频选择式问答（VideoQA）**。每道题包含一段视频片段和一条自然语言问题，模型需从多个候选答案中选出一个最符合视频语义的选项。

- 每个问题关联一段视频和 4 个候选答案；
- 模型需从中选择**唯一一个**最符合视频语义的选项；
- 回答结果以 `.json` 文件形式提交，格式为问题 ID 与对应选项索引的映射；
- 最终得分以整体准确率（**Accuracy**）进行评估。

你将面临的不仅是视觉识别挑战，更有**时间建模**、**跨模态融合**、**语义推理**的系统考验！

---

##  模型选择：InternVideo 多模态视频基础模型

### InternVideo2.5: Video Foundation Models for Multimodal Understanding

- **项目主页**：<https://github.com/OpenGVLab/InternVideo>  
- **机构**：OpenGVLab（上海人工智能实验室）  
- **发布时间**：2025 年 1 月  
- **开源协议**：Apache 2.0

```bibtex
@misc{wang2024internvideo25,
  title={InternVideo2.5: Empowering Video MLLMs with Long and Rich Context Modeling},
  author={Wang, Yi and He, Yinan and Zhang, Yujie and Liang, Dongxu and et al.},
  year={2024},
  eprint={2401.09582},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

### 模型简介

InternVideo 系列模型致力于构建通用视频基础模型（Video Foundation Models），支持多模态理解与生成任务。**InternVideo2.5** 是最新版本，结合长上下文建模与多尺度空间压缩技术，提升了视频-语言对齐能力。

####  关键特性

- 支持超过 256 帧的视频上下文输入，适应长时序任务；
- 使用分层 Transformer 结构建模时空联合特征；
- 兼容 `model.chat()` 风格的对话式推理；
- Hugging Face 提供完整权重、推理接口与推理样例。

---

###  模型结构组成

- **视觉编码器**：基于 TimeSformer 改进版，支持多分辨率、多帧输入；
- **语言编码器**：使用 Hugging Face 接口支持任意 LLM 作为语言头（如 ChatGLM3、LLaMA2）；
- **跨模态融合器**：融合视频帧特征与语言 token，通过 Q-Former 或跨注意力方式对齐模态；
- **对话式生成头**：输出自然语言答案，支持上下文保留与多轮对话。

---


## 🔧 解题思路

1. **视频预处理与帧抽取**  
   - 利用 `decord` 按固定帧率或时长划分读取视频；  
   - 使用自适应分割（`dynamic_preprocess`）确保每帧分块大小一致并加入缩略图；  
   - 对每块图像进行 `Resize → ToTensor → Normalize` 转换。

2. **模型加载与多模态融合**  
   - 选用 Hugging Face 上的 **InternVideo2.5_Chat_8B** 多模态大模型，它在长上下文和细粒度时空结构建模上表现优秀；  
   - 通过 `AutoTokenizer` 与 `AutoModel.from_pretrained(..., trust_remote_code=True)` 加载模型与分词器，并将模型转为半精度、BF16 模式部署于 GPU。


3. **问题拼接与对话式推理**  
   - 构造前缀字符串 `Frame1: <image>\nFrame2: <image>\n...`，并拼接自然语言问题；  
   - 对每个视频-问题对调用 `model.chat()`，获得模型生成的回答文本；  
   - 对于多轮问答，保留 `history` 上下文，实现基于上轮回答的后续理解与推理。
   - 下图为视频回答日志
![alt text](../Photos/10/视频回答日志.png)

1. **答案候选打分与选择**  
   - 将模型回答与 4 个候选选项对比（例如字符串匹配或相似度打分），确定最符合语义的选项索引；  
   - 将所有题目的 ID－>选项索引映射收集到 JSON 列表中，写入 `NS-2025-10-answer.json`。


---

## 工具与脚本结构

```text
NS-2025-10-solution/
├── public_data/                  # 原始视频与测试问题
├── inference.py                  # 主推理脚本：帧抽取 → 多模态推理 → 候选选择
├── evaluate.py                   # 离线准确率评估脚本
├── generate_result.py            # 将模型输出映射为提交 JSON 格式
├── utils.py                      # 帧抽取、预处理、打分函数
├── requirements.txt              # 环境依赖列表
└── NS-2025-10-answer.json        # 最终提交文件
```
![alt text](../Photos/10/总流程图.png)
---



### 提交物
- `NS-2025-10-answer.json`（问题 ID 与选项索引映射列表）

### 复现凭证
- 环境与依赖列表（`requirements.txt`）
- 核心脚本（`inference.py`、`evaluate.py`、`generate_result.py`）
- 模型来源：Hugging Face 上的 OpenGVLab/InternVideo2_5_Chat_8B

### 模型总结

**总体总结**：
基于 InternVideo2.5_Chat_8B 多模态模型，设计了一套完备的帧抽取、预处理、对话式视频问答流程，能够在 VideoQA 基准上取得良好表现。通过对视频帧的细粒度时空压缩与长上下文对话，成功融合视觉与语言信息，准确回答了多种类型的问题。

**模型不足**：
- 对于极短或极长的视频，固定分段可能导致重要信息遗漏；
- 候选答案高度相似时，纯文本匹配容易产生歧义；
- 在低光或抖动严重的场景下，视觉特征提取可靠性下降。

**后续改进方向**：
可引入动态分段、更强的跨模态对齐策略及候选答案语义检索模块，进一步提升模型的鲁棒性与准确率。
````

---

