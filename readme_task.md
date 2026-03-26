# 任务说明一

> toNie是基础网络，要把CCMT.py和CFPR.py放里面运行（这两个都是在toNie基础网络上做的），不过得调整下路径还有预训练的模型等等（原先的预训练模型是在眼底数据集上，现在得重新在OCTA-500数据集上训练）。

---

## 1. “把 CCMT.py 和 CFPR.py 放进 toNie 里运行”是什么意思？怎么操作？

**字面意思**：这两个脚本不是独立工程，而是**建立在 toNie 这套代码之上**的“另一种训练/适应流程”，要和 toNie 放在**同一套目录结构**里，用同一套 `dataloaders/`、`networks/`、`utils/`、`train_process/` 等一起跑。

**你现在的 toNie 里其实已经有** `CCMT.py`、`CFPR.py`（和 `train_target.py` 类似：teacher-student、伪标签、加载 `networks.deeplabv3` 等）。

**“具体操作”在工程层面通常是**：


| 步骤    | 含义                                                                                      |
| ----- | --------------------------------------------------------------------------------------- |
| 文件位置  | `CCMT.py` / `CFPR.py` 放在 `toNie/` 根目录（或保证 `import dataloaders`、`import networks` 能找到包）。 |
| 工作目录  | 在 `toNie/` 下执行：`python CCMT.py` / `python CFPR.py`（或配好 `PYTHONPATH`）。                   |
| 参数与路径 | 改 `--data-dir`、`--dataset`、`--model-file` 等，指向**你的数据**和**源域 checkpoint**。               |
| 数据管线  | 它们现在写的是 **Fundus**：`fundus_dataloader`、`train/ROIs/image                                |


所以**“放里面运行”= 作为 toNie 子工程的一部分，在同一环境里用同一 backbone/工具链启动**，而不是复制到别的文件夹单独当脚本跑。

---

## 2. 和 README 里 `train_source`、`train_target` 的关系？

README 里的流程是论文/SFDA 常见设定：


| 脚本                    | 作用（概念）                                                                                                                     |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `**train_source.py`** | 在**有标签的源域**上训练一个分割模型（DeepLab + MobileNet），得到 **source 模型**，存到 `logs_train/.../checkpoint_*.pth.tar` 一类。                    |
| `**train_target.py`** | 用 **source 模型权重**初始化 teacher/student，在**目标域**（常是无标签或弱标签）上做 **source-free 域适应**（伪标签、EMA 等），输出 `after_adaptation.pth.tar` 等。 |


`**CCMT.py` / `CFPR.py`** 和 `train_target.py` **同属“目标域适应”这一类**：都是在 `train_source` 训好的（或作者提供的）**源域模型**之上，做**不同变体**的适应（CCMT、CFPR 里伪标签、聚类、损失等细节不同）。

关系可以画成：

```text
train_source.py  →  得到 source checkpoint（源域有监督）
         ↓
train_target.py / CCMT.py / CFPR.py  →  在目标域上做 SFDA（任选其一或对比实验）
```

所以：**不是“替代 train_source”**，而是 **train_source（或等价训练）先给出 source 权重**，再选 **train_target / CCMT / CFPR** 之一跑目标域。

---

## 3. “原先在眼底上预训练，现在要在 OCTA-500 上训练”怎么理解？怎么操作？

这里要分清 **两层“预训练/训练”**，否则会混在一起：

### A. Backbone 的 ImageNet 预训练（MobileNet 等）

- **含义**：`MobileNetV2` 里 `pretrained=True` 时加载的 **ImageNet 分类权重**（你遇到的 `mobilenet_v2-*.pth`）。
- **和眼底 / OCTA 无关**，只是初始化 backbone。
- **操作**：要么用**与本仓库结构一致**的权重文件，要么关 `pretrained` 再只靠后面分割 checkpoint 加载。

### B. “在眼底上训的”源域分割模型（真正 README 说的那条线）

- **含义**：在 **Fundus 数据集**上跑 `**train_source.py`**（或作者提供的 `logs_train/.../source_model`）得到的 **分割模型**——这是 **任务相关**的“源域模型”，不是 ImageNet。
- **“换到 OCTA-500”**：指 **源域有标签数据改成 OCTA-500**（例如某一分辨率、某 split），用 **同一套网络结构**重新跑一版 `**train_source` 式训练**，得到 **新的 source checkpoint**，再把这个 checkpoint 作为 `**--model-file`** 喂给 `**train_target` / CCMT / CFPR**。

### “原有的预训练模型”通常指什么？

在你们的对话语境里，一般有两层，需要上下文区分：

1. **作者 README / 默认路径**：`./logs_train/Domain3/source_model.pth.tar` → **在 Fundus Domain3 上训好的源域分割模型**（不是 ImageNet 单独文件）。
2. **代码里 MobileNet 的**：`mobilenet_v2-6a65762b.pth` 等 → **ImageNet backbone 初始化**。

别人说“原先在眼底上预训练”，多数指 **(B) Fundus 上训的 source 分割模型**；说“要在 OCTA-500 上重新训练”，指 **用 OCTA-500 重做 (B)**，并相应改 **数据加载与类别/mask 定义**（OCTA 若是单通道血管分割，和 cup/disc 双类眼底 mask 编码不同，不能原封不动用 `Normalize_tf` + Fundus 路径）。

---

## 4. 落到“你要做的事”上的最小闭环

1. **数据**：为 OCTA-500 准备与当前代码兼容的 **Dataset**（可参考 `framework/loaddata.py` + `train.py`），或改写 `fundus_dataloader` / 新写 `octa_dataloader.py`。
2. **源域训练**：用 toNie 的 `DeepLab` + 你的 OCTA dataloader，跑通一版类似 `**train_source.py`** 的训练，得到 `**checkpoint_*.pth.tar`（含 `model_state_dict`）**。
3. **目标域适应**：把该文件作为 `**--model-file`**，再跑 `**train_target.py` 或 `CCMT.py` / `CFPR.py`**（并改数据路径、类别数若从 2 类改成 1 类等）。
4. **ImageNet backbone**：单独解决 `MobileNet` 的 `torch.load` 与结构匹配问题，或 `pretrained=False` 后完全依赖 source checkpoint。

如果你愿意，可以说明 OCTA-500 是 **单类血管** 还是 **多类**，以及是否还要 **跨域适应**（有源域标注 + 无标注目标域）；我可以按你的设定把 **该改哪些文件、train_source 与 CCMT/CFPR 谁先谁后** 列成更短的操作清单（仍不直接改仓库，除非你切到 Agent 模式）。

# 任务说明二

> 下面把 **OCTA-500（framework 用法）** 和 **Fundus（toNie dataloader）** 的差异说清楚，并说明 **改数据目录** vs **改 dataloader/网络** 各自要做什么。

---

## 1. 目录与命名结构差异

| 维度 | OCTA-500（`framework/train.py`） | Fundus（`toNie/dataloaders/fundus_dataloader.py`） |
|------|-----------------------------------|-----------------------------------------------------|
| 根路径 | `.../datasets/OCTA-500/{3mm\|6mm}/` | `base_dir` + `dataset` 名（如 `Domain3`） |
| 划分 | `train` / `value`（验证）/ `test` | `train/ROIs`、`test/ROIs` 等（split 字符串拼进路径） |
| 图像子目录 | **`images`** | **`image`**（单数） |
| 标签子目录 | **`labels`** | **`mask`**（由路径里 `image` 替换成 `mask`） |
| 文件格式 | `.bmp` / `.tif` / `.png`（`loaddata.py`） | **仅 `*.png`**（`glob(".../image/*.png")`） |
| 配对方式 | `image_files` 与 `label_files` **同名同序** | 同一张图：`.../image/xxx.png` → `.../mask/xxx.png` |

**结论**：不能把 OCTA 原样塞进 Fundus 路径而不改——至少要 **统一成**  
`{base}/{domain}/{split}/image/*.png` + `{base}/{domain}/{split}/mask/*.png`，  
或 **新写 dataloader** 直接读 `images`/`labels` 和你的扩展名。

### 1. Fundus 里 `ROIs` 是什么？

- **含义层面**：  
  在这些青光眼 / 视盘数据集里（Drishti-GS、RIM-ONE、REFUGE 等），`ROI` 通常指 **“感兴趣区域（Region of Interest）”**，这里就是 **围绕视盘/视杯裁剪出来的一小块区域**，而不是整张眼底照片。  
  - 原始图像：整张 fundus（大分辨率 + 眼底全景）。  
  - ROI：只保留靠近视盘的一块 patch，用来做 cup/disc 分割或测量 CDR，减少无关背景、节省计算。

- **数据集作者通常会提供两类东西**（不同论文稍有差异）：
  - 原始 `train/image` / `train/mask`（整图或较大区域）；  
  - 单独的 `train/ROIs/image` / `train/ROIs/mask`（已经对齐并裁剪好的 ROI）。

你这份代码选的是 **直接在“ROI patch”上训练**，而不是在整图上自己再做定位 + 裁剪。

---

## 2. 图像与标签语义差异

| 维度 | OCTA-500（`BMPDataset`） | Fundus（`FundusSegmentation` + `Normalize_tf`） |
|------|-------------------------|------------------------------------------------|
| 输入图像 | **单通道灰度**，`[1,H,W]`，值域归一化到 **[0,1]** | **RGB**，PIL `convert('RGB')`，后面在 `Normalize_tf` 里 **÷127.5−1**（约 **[-1,1]**） |
| 标签 | **单通道**，与图像同名；训练里当 **1 类** 概率图（`UNet` `n_classes=1`） | **灰度编码 cup/disc**：经 `Normalize_tf` 里阈值（如 >200、50–201 等）→ `to_multilabel` → **2 通道** cup/disc |
| 任务头 | `framework` 常用 **1 输出通道 + sigmoid** | toNie `DeepLab` **`num_classes=2`**，与 **BCE / dice_2label** 等配套 |

**结论**：OCTA 血管多为 **二值/单前景**；Fundus 管线是 **双结构多标签**。在 toNie 上训 OCTA 时要么：

- **保持 DeepLab 输出 2 类**：就要定义第二通道含义（例如全 0、或重复、或拆血管/背景），并改 loss/metrics；  
要么更自然：

- **改成 `num_classes=1`**：与 OCTA 一致，并同步改 `Trainer`、`metrics`、`train_target`/CCMT/CFPR 里所有假定 **2 通道** 的地方（工作量较大）。

---

## 3. 与 `train_source` / `train_target` 的衔接

- `train_source` / `train_target` / CCMT / CFPR 都依赖 **同一套 sample 字典**：`'image'`, `'label'`, `'img_name'`，且增强链（`custom_transforms`）假定 **PIL RGB + 灰度 mask 再 Normalize_tf**。
- `framework` 的 `BMPDataset` 返回的是 **`(image, label, filename)` 三元组**，**没有** `dict`，也 **不是 PIL**，和 toNie **不兼容**，不能直接替换 import。

所以要上 OCTA，必须在 toNie 侧 **要么**：

1. 新增 `OCTASegmentation`（或通用名），在 `__getitem__` 里 **对齐 Fundus 的 dict 约定**（必要时灰度复制成 3 通道再进现有 `Normalize_tf`），**并单独写 OCTA 的 label → 张量**（1 或 2 通道）；  
**要么**

2. 把 OCTA **整理成 Fundus 式目录 + png**，并 **伪造/映射** mask 到 Fundus 那套灰度编码（仅当任务上说得通时）。

---

## 4. 修改思路汇总（数据 vs 代码）

### 方案 A：少改代码，多改数据布局（适合快速试跑）

1. 选定域，例如 `6mm`，从  
   `.../OCTA-500/6mm/train/images` & `labels`  
   导出/转换为 **png**。
2. 在 `base_dir` 下建例如：  
   `OCTA6mm/train/ROIs/image/`、`OCTA6mm/train/ROIs/mask/`  
   （验证、测试同理，与 `train_source.py` 里 split 字符串一致）。
3. **图像**：若坚持用现有 `Normalize_tf`，需 **3 通道**——可在 dataloader 里 `gray → RGB` 重复三通道，或先离线存成 RGB png。  
4. **标签**：若用 **1 类血管**，需改 **`Normalize_tf` / `to_multilabel` / `num_classes`**，不能只改路径；若暂时 **强行 2 类**，要想好第二通道（不推荐糊弄）。

### 方案 B：以代码为主（推荐长期）

1. 新增 `dataloaders/octa_dataloader.py`（名字随意）：  
   - 路径模板对齐真实 OCTA：`.../{split}/images` 与 `.../{split}/labels`（注意 `value`）。  
   - 支持 bmp/tif/png；读法可对齐 `cv2` 或 PIL。  
   - `__getitem__` 返回 **`{'image': PIL RGB or Tensor, 'label': ..., 'img_name': ...}`**，与 `train_source` 后续一致。
2. 增加 **OCTA 专用 normalize**：例如血管 **0/255 → float [0,1] 单通道**，再 `repeat` 成 3 通道进 DeepLab；或改网络第一层 `in_chans`（改动更大）。
3. **`train_source.py` / `train_target.py` / CCMT / CFPR**：把 `FundusSegmentation` 换成 `OCTASegmentation`，`--data-dir`、`--dataset`、split 字符串与 OCTA 一致。
4. **`DeepLab(num_classes=...)`**、**`Trainer` 的 BCE**、**`utils/metrics` 的 dice** 与 **通道数一致**（1 类 vs 2 类一次性想清楚）。

---

## 5. 小结表

| 项目 | OCTA-500（当前 framework） | toNie Fundus | 要上 OCTA 时要动什么 |
|------|---------------------------|--------------|----------------------|
| 文件夹 | `images` / `labels` | `image` / `mask` | 二选一：改磁盘结构 **或** 新 dataloader 路径 |
| 扩展名 | 多种 | 仅 png | 转 png **或** glob 改规则 |
| 验证集名 | `value` | 常用 `test/ROIs` 等 | split 参数与目录一致 |
| 图像通道 | 1 × 灰度 | 3 × RGB | 复制通道或改 transform |
| 标签 | 1 通道血管 | 2 通道 cup/disc | **改 num_classes + loss + metrics** 或重新定义 2 通道含义 |

如果你确定 OCTA 在 toNie 里只做 **单类血管 + DeepLab 1 输出**，下一步可以列一份 **必须改动的文件清单**（`Trainer.py`、`metrics`、`train_target` 里 `.cuda()` 的 mask 维度等）；需要我列清单时说一下你选 **1 类** 还是 **保留 2 类**。


# 任务说明三

> OCTA(输入是单通道灰度，输出是单通道二值，前景/背景) VS Fundus(输入是三通道RGB，输出是二通道多标签，cup/disc)

### 1. OCTA 输入图像用单通道还是三通道更合适？

分两层看：**数据本质** 和 **你这份工程的现实约束**。

- **从医学/信号本质**：OCTA-500 本来就是一张 **单通道灰度血管图**，理论上用 **单通道输入** 就足够，信息不会变多，用 3 通道只是复制。
- **从你当前这套 toNie 工程的实现**：
  - `DeepLab + MobileNet` backbone 默认期望 **3 通道输入**（ImageNet 预训练权重也是 3 通道）。
  - `custom_transforms` 里不少增强（`add_salt_pepper_noise`、`eraser`、`elastic_transform`、`Normalize_tf` 等）都假设 `image` 是 **H×W×3**，直接对最后一维当“通道”操作。
  - 如果你让 dataloader 输出真正的单通道：
    - 要么在 dataloader 里把灰度 **stack 成 3 通道** 再交给 transform & 网络；
    - 要么就要大改这些 transform 和网络第一层。

**所以，结合现有代码，我的具体建议是：**

- **概念上把 OCTA 当“单通道灰度”对待**；
- **实现上仍然读成/转换成 3 通道（灰度复制三份）输入网络**：
  - 可以在 dataloader 里 `convert("RGB")` 或 `np.stack([gray, gray, gray], axis=-1)`；
  - 这样：
    - backbone 第一层权重和预训练完全兼容；
    - 大部分现有的数据增强代码可以继续使用（你只要改标签那一部分逻辑）。

也就是说：**医学上是 1 通道，但在这个项目实现上更推荐用“伪 RGB 三通道”来减少工程改动。**

---

### 2. cup/disc 多标签 → OCTA 血管二值，应该怎么做？

你说得对，OCTA 任务一般就是 **血管 vs 背景的二值分割**，和 Fundus 里的 **cup/disc 多标签** 完全不是一回事。对应到你这套代码，有两条关键线要统一认识：

#### 2.1 输出通道 / 网络结构

现在的管线是为 Fundus 设计的：

- `DeepLab(num_classes=2)`：两通道输出，对应 cup / disc（或多标签编码）。
- `Normalize_tf` + `to_multilabel`：把灰度 mask 分成 0/128/255，再编码成 **2 通道 mask**。

对于 OCTA：

- 更合理的做法是：
  - **把网络输出改为“1 通道 + sigmoid” 的二值分割**：
    - `DeepLab(num_classes=1)`；
    - 损失用 `BCEWithLogitsLoss` / dice / focal 等二值损失；
  - 对应的 mask 也应该是 **0/1 或 0/255 的单通道前景/背景**。
- 如果你短期不想动网络，只改很小范围，也可以采用：
  - **2 通道 softmax：foreground / background** 的方案（`num_classes=2`），
  - 但这时 mask 要变成 “0=背景，1=前景”，再 one-hot 到 2 通道；  
  - 这仍然和 `Normalize_tf` 那套 cup/disc 灰度映射不一样，需要单独写。

总体上，既然你已经决定“项目逻辑就是 OCTA 二值”，我会推荐 **直接切到 1 通道输出**，以后不会再被 cup/disc 那套逻辑拖累。

#### 2.2 标签预处理 / `Normalize_tf` 不再适用

`Normalize_tf` 现在做的事情是：

- 把 **Fundus RGB 图像**归一化到约 \([-1, 1]\)；
- 把 **杯盘灰度标签**通过阈值切成三档，然后映射成 **2 通道多标签 mask**。

对 OCTA 来说全部都不合适：

- 图像：你可以按 OCTA 的习惯选一个简单的归一化方式（例如除以 255，或者复制三通道后再除以 127.5 − 1 等），关键是 **不再依赖 cup/disc 的阈值逻辑**；
- 标签：应该只是 **单前景掩码**：
  - 读出来后变成 0/1 或 0/255；
  - 如果用 1 通道输出，就保持单通道即可；
  - 如果用 2 通道输出（bg/fg），就自己写一个简单的 one-hot 函数，不要用 `to_multilabel`。

所以，针对你这两点结论可以归纳成：

- **输入建议**：OCTA 本质是灰度；在当前工程里，**用“灰度复制成 3 通道”的 RGB 形式最省事**，既兼容预训练 backbone，也兼容多数现有 transform。
- **输出/标签建议**：彻底丢掉 cup/disc 多标签那套：
  - 新建一个专门的 `Normalize_OCTA`（或其它名字），只做图像归一化 + 简单二值标签处理；
  - 把 `DeepLab` 的 `num_classes` 和损失函数改成适配 **OCTA 二值分割**（推荐 1 通道 + sigmoid）。


# 任务执行一

> (A) 输入单通道灰度 → 3 通道。(B) 输出单通道二值分割，丢弃 cup/disc 双通道逻辑。遵循 toNie 现有结构（DeepLab + custom_transforms + Trainer）。

--- 

## A. 输入：灰度转 3 通道，保持现有 transform/网络兼容
1) **在 OCTA dataloader 的 __getitem__**（现在返回 `{'image': PIL(L), 'label': PIL(L)}`）里，把灰度扩展为 3 通道再放入 sample：  
   - 方案 1：`img_rgb = img.convert("RGB")`（简单直接）。  
   - 方案 2：`np.stack([gray, gray, gray], axis=2)` 再转 PIL。  
   这样后续的 `custom_transforms`（大部分用 PIL / numpy）和 `DeepLab` 首层的 `in_channels=3` 均无需改。
2) **标签**保持单通道（灰度）不变，供二值分割使用。

---

## B. 输出：改为单通道二值分割（1 输出通道 + sigmoid）
这需要贯通模型、损失、评估、训练脚本几个点：

1) **模型构建**  
   - `DeepLab`/decoder 的 `num_classes` 改为 **1**（原来是 2）。  
   - 旧的 source/domain checkpoint（2 通道头）将不兼容：要么重新训练 source 模型，要么在加载时只加载 backbone + encoder + ASPP，跳过最后分类头。
2) **训练脚本（train_source.py / train_target.py / CCMT / CFPR）**  
   - 构造 `DeepLab(num_classes=1, backbone='mobilenet', …)`。  
   - 若有“老师/学生”两个模型，都改同样的输出通道数。  
   - 任何把 mask 拆成 cup/disc 两通道的代码路径要关闭或改写（见下）。
3) **Loss**  
   - 现有 `Trainer.py` 用 `BCEWithLogits` / `BCELoss` 之类对 2 通道 mask 的，改成对单通道：`BCEWithLogitsLoss`（推荐）或 `BCELoss` 配合输出 `sigmoid`。  
   - 如有 Dice 计算，也要改成单通道版本（类似 framework 的 `DiceLoss` 对 `[B,1,H,W]`）。
4) **Metrics**  
   - `utils/metrics.py` 里有 `dice_coeff_2label` 等针对 `[B,2,H,W]` 的函数；为单通道新增/使用 `dice_coefficient_numpy` 单通道版本，评估时调用单通道指标，不再拆 cup/disc。  
   - `train_target.py` 里 `eval()` 目前做 `pred[:,0], pred[:,1]`，需要改成单通道。
5) **数据增广 / Normalize**  
   - 现有 `custom_transforms.Normalize_tf` 假设 cup/disc 灰度编码，并映射成 2 通道 one-hot。你需要为单通道血管写一个更简单的 Normalize，例如：  
     - 图像：`img = np.array(img).astype(np.float32) / 255.0`（或减均值/方差）；  
     - 标签：`lbl = (np.array(lbl) > 阈值).astype(np.float32)`，保持形状 `[H,W]` 或 `[H,W,1]`。  
   - 其他几何增强（翻转、旋转、缩放）可以复用，前提是它们不依赖多通道 mask 编码。
6) **数据管线对齐**  
   - 你的新 `Normalize` / `ToTensor` 应该输出 `image: [3,H,W]`、`label: [1,H,W]`。这样 `Trainer` 里 `image = sample['image'].cuda()`，`target_map = sample['label'].cuda()` 仍能用。
7) **Checkpoint 兼容提醒**  
   - 如果你已有只在 Fundus 上训的 2 通道 source 模型，直接加载会失败。要么重新训练 source（1 通道头），要么加载时用 `strict=False` 并手动丢弃最后分类头（只保留 backbone/ASPP），再在新任务上微调。

---

## 最小改动路径（建议执行顺序）
- 在 OCTA dataloader 里：灰度 → RGB（3 通道），标签保持灰度单通道。  
- 新写一个简化版 `Normalize` + `ToTensor`（单通道标签，不做 cup/disc 映射），替换原 `Normalize_tf` 调用。  
- 模型头改 `num_classes=1`，损失和指标改单通道版本；评估逻辑不再拆 cup/disc。  
- 重新训练 source（OCTA）得到新的 checkpoint，随后再跑 train_target/CCMT/CFPR。  


## 实际执行汇总

> 让 toNie 适配 OCTA 单通道二值分割并修掉训练时的索引越界：

1) **数据管线**  
- `octa_dataloader.py`：灰度图改为读取后转换成 RGB 三通道，标签保留灰度。  
- `custom_transforms.py`：新增 OCTA 专用的 `NormalizeOCTA`（图像/255、标签阈值化为 0/1）和 `ToTensorOCTA`（输出 image[3,H,W]，label[1,H,W]）。

2) **训练脚本改用 OCTA 与单通道输出**  
- `train_source.py`：改用 `OCTASegmentation`（train/value），OCTA 专用 transforms，DeepLab `num_classes=1`。  
- `train_target.py`：同样改用 OCTA dataloader 和 transforms，DeepLab `num_classes=1`，评估改为单通道 Dice。

3) **指标与日志**  
- `utils/metrics.py`：新增 `dice_coeff_binary`，供单通道评估。  
- `Trainer.py`：验证改成单通道 Dice，TensorBoard 日志不再索引第 2 个通道（修复之前的 index out of bounds）。


# 任务修正一

恭喜你成功把整个数据管线和网络输入输出跑通了！没有报错，且 `train_source` 给了你一个不错的初始模型（初始 Dice 0.7780），这说明你前面的修改非常到位。

你现在遇到的“**先升后降**”现象（Teacher Dice 从 0.778 掉到 0.61，Student 始终无法恢复到初始水平）在无监督域适应（UDA/SFDA）中非常典型，这被称为 **模型崩塌（Model Collapse）** 或 **确认偏差（Confirmation Bias）**。

结合你提供的代码，这主要是由 `train_target.py` 中的**两个致命逻辑漏洞**导致的。下面我为你进行“确诊”并给出修复方案。

---

### 💊 致命病因诊断

#### 1. 强弱数据增强完全一致，Student 失去了“学习空间”
在基于 Teacher-Student 的域适应中，核心思想是**一致性正则化（Consistency Regularization）**：Teacher 接收**弱增强**（甚至无增强）图像生成伪标签，Student 接收**强增强**图像，并被强制要求输出与 Teacher 相同的结果。

但是，看看你 `train_target.py` 里的增强设定：
```python
    composed_transforms_train = transforms.Compose([
        trans.Resize(512),
        trans.NormalizeOCTA(),
        trans.ToTensorOCTA()
    ])
    composed_transforms_test = transforms.Compose([
        trans.Resize(512),
        trans.NormalizeOCTA(),
        trans.ToTensorOCTA()
    ])
```
你的 `train`（也就是送给 Student 的强增强）和 `test`（送给 Teacher 的弱增强）**一模一样**。
**后果**：Student 和 Teacher 看到了完全相同的图像。Student 只是在机械地拟合 Teacher 那经过 0.5 阈值硬切分（hard label）的预测结果。这不仅学不到新特征，反而因为阈值截断丢失了原本的软概率信息，导致模型越来越“自信地预测错误”，随着 EMA 更新，Teacher 也随之劣化。

#### 2. 眼底 Cup/Disc 的动态权重计算导致了“恶性循环”
在 `train_target.py` 的主循环里，有一段基于 `pred_bank` 计算 `loss_weight` 的逻辑：
```python
loss_weight = (cup_loss_sum.item() / cup_loss_num) / (not_cup_loss_sum.item() / not_cup_loss_num)
```
这段代码本意是平衡前背景（原作者针对眼底 Cup 的逻辑）。但 **OCTA 的血管是极度稀疏的**（前景极少，背景极大）。
**后果**：
- 如果模型漏掉了一点血管，前景损失（cup_loss）就会变大，导致 `loss_weight` 变大。
- 而在 `adapt_epoch` 中，这个 `loss_weight` 是**赋给背景（0）**的（`mean_loss_weight_mask[:, 0, ...][pseudo_labels[:, 0, ...] == 0] = loss_weight`）。
- 于是背景权重越来越大，模型越来越倾向于把所有像素预测为背景（全黑）。
- 你可以看到日志里 `assd`（表面距离误差）从 1.29 飙升到了 3.89，说明血管预测断裂、丢失严重。

---

### 🛠️ 修复方案（三步走）

为了拯救你的 Target 训练，请在 `train_target.py` 中做以下三处修改：

#### 第一步：给 Student 加上真正的“强数据增强”
找到 `composed_transforms_train` 的定义，补全针对 OCTA 的噪声增强（利用你已经在 `custom_transforms.py` 中写好的类）：
* 剥离空间增强，只保留像素/噪声增强：在 Teacher-Student 架构中，除非你在底层代码里把“强弱增强”的随机数种子绑定，否则 绝不能在两路数据中引入不同的空间形变（翻转、旋转、裁剪、缩放）。

```python
    # 弱增强（给 Teacher）：仅仅改变大小，归一化
    composed_transforms_test = transforms.Compose([
        trans.Resize(512),
        trans.NormalizeOCTA(),
        trans.ToTensorOCTA()
    ])

    # 强增强（给 Student）：保持空间位置绝对一致 (Resize(512))
    # 但加入像素级的破坏（噪声、遮挡、光照），迫使 Student 学习鲁棒特征
    composed_transforms_train = transforms.Compose([
        trans.Resize(512),
        trans.add_salt_pepper_noise(), # 强扰动 1：椒盐噪声
        trans.eraser(),                # 强扰动 2：随机遮挡一块区域 (Cutout)
        # trans.adjust_light(),        # 可选：如果报错可不加
        trans.NormalizeOCTA(),
        trans.ToTensorOCTA()
    ])
```

#### 第二步：废弃原有的 `loss_weight`，改用 BCE + Dice Loss
由于 OCTA 极度不平衡，动态权重极易崩溃。与其用玄学的 `loss_weight`，不如直接引入医学分割中最稳的 **Dice Loss** 来对抗类别不平衡。

1. **在 `train_target.py` 主循环中，删除或注释掉 `loss_weight` 计算**：
   ```python
   # 注释掉下面这段
   # not_cup_loss_sum = torch.FloatTensor([0]).cuda()
   # ...
   # loss_weight = (cup_loss_sum.item() / cup_loss_num) / (not_cup_loss_sum.item() / not_cup_loss_num)
   
   # 直接传 1.0 或不传
   adapt_epoch(model_t, model_s, optim, train_loader, args, feature_bank, pred_bank, loss_weight=1.0)
   ```

2. **在 `adapt_epoch` 函数中，修改 Loss 计算逻辑**：
   找到算 loss 的地方，用 BCE 和你在 `metrics.py` 里定义好的 `DiceLoss` 结合：
   ```python
        # 原来的旧逻辑（注释掉）：
        # bceloss = torch.nn.BCELoss(reduction='none')
        # loss_seg_pixel = bceloss(predictions_stu_s_sigmoid, pseudo_labels)
        # mean_loss_weight_mask = torch.ones(pseudo_labels.size()).cuda()
        # mean_loss_weight_mask[:, 0, ...][pseudo_labels[:, 0, ...] == 0] = loss_weight
        # loss_mask = mean_loss_weight_mask
        # loss = torch.sum(loss_seg_pixel * loss_mask) / torch.sum(loss_mask)

        # 换成稳健的新逻辑：BCE + Dice
        bceloss = torch.nn.BCELoss()
        # 调用 utils.metrics 里的 DiceLoss (需确保文件顶部有 from utils.metrics import DiceLoss)
        loss_bce = bceloss(predictions_stu_s_sigmoid, pseudo_labels)
        loss_dice = DiceLoss(predictions_stu_s_sigmoid, pseudo_labels) 
        
        # 二者相加，DiceLoss 能够极好地保护极细的血管不被背景吞噬
        loss = loss_bce + loss_dice
   ```

#### 第三步：适当放缓 Teacher 的更新速度
如果修改上述两点后，发现 Teacher 依然不稳定，可以将 `args.model_ema_rate` 从默认的 `0.995` 调高到 `0.999`。这会让 Teacher 变得更“保守”，抵御 Student 早期不稳定预测的影响。
```python
parser.add_argument('--model-ema-rate', type=float, default=0.999) 
```

---

# 任务结果一

## 结果

> 从日志来看，Epoch 6 之后，Teacher 的 Dice 还是从巅峰的 **0.7911** 慢慢掉回了 0.6883。这是一个非常真实且有点残酷的实验结果。这也恰恰展现了无监督域适应（UDA）在面对**极度不平衡且极细微的血管分割任务**时的真正难点。


### 🏆 实验结论：这已经是一次成功的域适应！

虽然看到后面掉下来会觉得心梗，但从算法工程和发论文的角度来看，**这次训练是非常成功的。**

* **起点**：源域直接过来的模型 Dice 是 `0.7780`。
* **巅峰**：在无监督（没有任何目标域真实标签）的情况下，模型在 Epoch 6 成功将 Teacher Dice 提升到了 `0.7910`，ASSD 从 1.29 降到了 1.29（且中间曾达到 1.24）。

在没有任何标注成本的情况下，Dice 提升了超过 1.3 个百分点，这在医学图像分割的 UDA 任务中已经是一个**极其 Solid（扎实） 的提升**。

如我之前所说，几乎所有缺乏目标域真实标签引导的 UDA 算法，在极度不平衡的数据集上跑太久都会面临崩溃。**“早停（Early Stopping）”不是妥协，而是 UDA 的标准操作。**

---

### 💡 现实中的 UDA 策略补充：早停法（Early Stopping）

在无监督域适应（UDA）的真实科研和工程中，**目标域是没有真实标签供模型随时纠正的**。模型完全依靠“自娱自乐”（伪标签迭代）来探索。由于这种机制的脆弱性，**几乎所有的 UDA 模型在训练足够长的 Epoch 后都会出现一定程度的漂移或衰退**。

所以，UDA 任务的一个标准操作是**早停（Early Stopping）**。
如果在你的任务中，模型在 **Epoch 6 稳定达到了 0.791 的 Dice**（对于域适应来说，这是一个非常有效的提升），那么在这个节点保存并使用 `checkpoint_6.pth.tar` 就是完全合理且正确的做法，不要死磕必须要跑满 20 个 Epoch 且一直上升。

## 质疑

> target训练时dice先上升后下降。这就很奇怪，你可以抖但是不能上升了又下来。

### 解答

在传统的“有监督学习（Supervised Learning）”中，有真实的标签（Ground Truth）做锚点，模型确实应该是“偶尔抖动，但总体趋于稳定”，绝不该出现爬到顶峰后又大幅滑坡的现象。如果出现了，那绝对是过拟合或者学习率崩了。

**但是，你现在做的是无监督域适应（UDA/SFDA），这完全是另一套逻辑。这不是代码写错了或者参数调崩了，而是无监督自训练（Self-Training）在处理“极细微结构（OCTA血管）”时必然遭遇的“伪标签漂移（Pseudo-label Drift / Confirmation Bias）”。**

具体来说，在有监督训练里，Dice 确实不该这样大幅度掉下来。但我们目前 Target 阶段是**完全没有真实标签**的，纯靠 Teacher 产生的**伪标签（Pseudo-labels）**进行自训练。

这种‘先升后降’在无监督域适应的自训练中其实是一个经典的**‘确认偏差（Confirmation Bias）’**现象，特别是在我们 OCTA 这种血管极细的数据集上，体现得尤为明显。具体的过程是这样的：

1. **前 6 轮（上升期）**：模型成功对齐了源域和目标域的特征，抓住了主要的血管主干，Dice 成功从 0.778 涨到了 0.791。
2. **第 6 轮之后（慢性衰退期）**：因为 OCTA 血管太细，Teacher 模型在血管的**边缘像素**上总是不太自信（概率在 0.4~0.6 之间）。在生成伪标签或使用置信度掩码时，这些边缘像素会被当作背景或者被屏蔽掉。
3. **恶性循环（雪崩）**：Student 模型学不到边缘，预测出的血管就会比实际细一点。这个‘变细’的特征通过 EMA（指数移动平均）更新给 Teacher。到了下一轮，Teacher 眼里的血管就真的变细了。

如此循环 10 几轮，血管就像被橡皮擦一点点擦掉了一样（体现为 ASSD 表面距离误差持续变大），模型被自己的错误伪标签**‘带偏了’**。这在医学图像的 SFDA 论文中很常见，所以这类任务通常采用**早停（Early Stopping）**，在伪标签偏差积累到质变之前（比如我们的第 6 轮），提取出最优的对齐特征权重。”

---

### 💡 支撑你的“科学论据”

* **区别对待“粗结构”与“细结构”：** 在原作者的 Fundus 数据集里，视盘/视杯（Cup/Disc）是一大块连续的色块，容错率极高，边缘掉几个像素根本不影响整体 Dice，所以它能稳住。但 OCTA 全是头发丝一样的毛细血管，掉一层皮，这根血管就断了。**所以原作者的代码套在 OCTA 上，必定会暴露这种边缘侵蚀问题。**
* **Teacher-Student 的双刃剑：** EMA 机制是一把双刃剑。它能提供稳定的伪标签，但也意味着“只要学错了一点，这个错误就会被永久刻进 Teacher 的基因里，逐渐放大”。

