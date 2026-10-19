# LabelMe 全面细化与双向视频传播

默认过滤可审查的打标器微小圈，对有效医生区域尝试 SAM3 细化，再用稀疏关键帧
双向传播约束所有类别的跨帧边界（包括原来每帧都有人标的类别）。
**最终原图+JSON 和可视化采用时序融合与局部边界协调后的候选**。
局部细化失败时回退人工区域作为后续时序融合的依据，不再逐帧冻结它的边界。
原始数据只读；不训练、不做文本语义检测、不改核心模型。

## 运行

在项目根目录：

~~~bash
conda activate surgslot-sam3
python surgslot-sam3/scripts/labelme_video_annotate.py \
  --gpus 0,1,2,3 --fps 10 --name temporal_v2_02
~~~

本机 Python 为 /home/share/yzy/miniconda3/envs/surgslot-sam3/bin/python。
默认输入是项目父目录 Data/邢静远Segment0006_t193p000s_fps10_n90，
默认 checkpoint 为 surgslot-sam3/checkpoints/surgslot-sam3_iSurg.pt。
已有输出不覆盖；重复运行用新的 --name，省略时采用输入目录名。

四卡各加载一份 SurgSLOT tracker，按锚点帧分组分配任务，复用同帧图像特征；
不是切分同一个模型。图像和状态在 CPU，GPU 使用 BF16 autocast，默认输入分辨率 672。
支持等尺寸 JPG/JPEG/PNG/BMP 和同名的部分/全部 LabelMe JSON，
按自然数字顺序排列，支持 polygon、circle、rectangle 区域。

## 输出

~~~text
outputs/
├── 输出名/
│   ├── frame_....jpg                 # 原始图片字节复制
│   ├── frame_....json                # 真正采用细化候选的最终 LabelMe 标注
│   ├── refinement_report.json        # 每个原始 shape 的尝试、接受、回退与导出情况
│   ├── review.json                   # 新增重叠、双向分歧、细化回退、孔洞等
│   └── run_report.json               # 汇总、参数、四卡信息、原始 JSON 哈希
├── 输出名_visual/
│   ├── frame_....jpg                 # 所有类别在同一张图上叠加
│   ├── legend.json
│   └── preview.mp4                   # 默认 10 fps 的全序列视频
└── 输出名_refined/
    ├── frame_....jpg + .json         # 人工锚点的细化候选，不含传播补充类别
    ├── frame_...._comparison.jpg     # 左边医生原始边界，右边细化候选
    └── refinement_report.json
~~~

LabelMe 使用 Open Dir 打开第一个目录，或直接打开逐帧 JSON。
imagePath 指向同目录原图，imageData=null，整个目录可以移动。
报告 JSON 不是逐帧标注文件。

- 医生提供类别和正区域，最终候选允许改变边界。
- 接受的候选 Polygon 具有 flags.automatic=true 和 flags.refined=true，
  局部细化的 source_shape_index 对应该帧原始 shape 下标。
  最终时序融合可能按类别重建多个 Polygon，使用 source_shape_indices 记录该类别原始来源，
  不保证原来的实例 group_id；完整原始字段仍保存在来源数据。
- 每帧 JSON 的 surgslot_original_shapes 保存未经修改的全部医生 shape，
  surgslot_refinement 记录对应的候选检查结果；这些额外字段不会参与 LabelMe 绘图。
- 局部细化失败时先回退原区域；默认仍可进入后续时序融合。
  显式 preserve/no_refine 的区域保留原几何。过滤出的伪小圈不作为回退重新加入 shapes。
- 可视化左上角 source 是过滤后的有效人工图形数，filtered 是剔除数，
  tracked 是当前帧实际采用时序结果的类别数，propagated 仍只表示补充缺失类别的 Polygon 数。
  因此 propagated=1/2 不能用来判断已有类别是否参与跟踪。
  每帧 surgslot.source_raw/source_valid 保存原始数和有效数。
- 白色斜线只标出涉及自动结果的新增类别交叠，原本人工交叠不会被重新判为自动冲突。

## 选择范围和保留精细标注

默认全选，不需要再传 --refine。只处理指定类别/帧：

~~~bash
python surgslot-sam3/scripts/labelme_video_annotate.py \
  --name refine_fat_24 --refine-label fat --refine-frame 24
~~~

--refine-label 和 --refine-frame 均可重复；帧编号是自然排序后从 1 开始的位置。
某个 shape 的 flags.preserve=true 或 flags.no_refine=true 可以单独保留原边界。

只补充缺失类别、保留有效人工边界（仍执行微圈过滤）：

~~~bash
python surgslot-sam3/scripts/labelme_video_annotate.py --name preserve_02 --no-refine
~~~

## 细化机制和参数

每个医生区域独立细化，避免不同类别及实例之间的重叠清除：

1. 建立独立单帧推理状态，隔离传播记忆和反向帧索引。
2. 以原区域为正提示，采样默认 3 个分散的内部正点，并提供稍扩大的外接框；
   不添加 Polygon 外部的负点，也不把其他未标类别视为不存在。
3. 候选仅保留与原正区域接触的连通部分，检查位置、原区域保留率和面积。保护非 rough 标注的内部正区域，防止模型制造断带。
4. 失败时补一个未覆盖部分的正点重试，默认最多 2 次；仍失败则回退原标。
5. 每类约每 8 帧选择一个关键锚点，用已有 SurgSLOT 视频 tracker 向相邻关键帧双向传播。
   当前帧局部细化结果与左右跟踪概率融合，默认时序权重 0.65。
   与当前正区域严重不一致或为空的跟踪不会抹掉人工正区域。
6. 对新产生的窄边缘交叠，依据原正区域距离和模型概率共同分配归属，
   与类别处理顺序无关。原始交叠和大范围语义矛盾保留供复核。
   只修复有正区域/时序证据的孔隙，不把未标的视野强制铺满。

主要参数：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| --prompt-dilate | 5 | 原图像素中的额外框外扩量，小区域自动减小；不是统一膨胀最终 mask |
| --box-padding | 0.1 | 框宽高的额外外扩比例 |
| --refine-points | 3 | 内部正点数 |
| --refine-attempts | 2 | 最大尝试次数 |
| --min-seed-coverage | 0.8 | 普通候选须覆盖至少 80% 的原正区域 |
| --rough-seed-coverage | 0.35 | 显式 flags.rough=true 的粗标允许较大收缩 |
| --max-expand-px | 32 | 超出原区域的最大编辑距离，狭小区域进一步减小 |
| --max-area-ratio | 3 | 与原区域接触的候选面积上限，超过则拒绝 |
| --polygon-epsilon | 1 | 仅自动轮廓的简化容差，原图像素 |
| --min-area | 8 | 仅自动 Polygon 的最小轮廓面积 |

需要扩大搜索范围时可以增加 --box-padding、--prompt-dilate 或 --max-expand-px；
它们不保证候选一定扩大，也不意味着扩大后一定更准确。

refinement_report 中 accepted 表示候选通过 mask 检查，
export_changed 表示局部细化阶段采用了新的几何；二者不是同一个指标。
最终时序融合还可能按类别重建这些图形，不能将局部统计当作最终 Polygon 数。
changed_pixels、seed_coverage、area_ratio、iou_with_original 描述与原区域的关系。
当医生标注偏小时，不能用与原 Polygon 的 IoU 单独判断分割质量。

## 视频传播与重叠

每个类别独立建立全部人工锚点，缺少标注不作为负样本。
相邻同类锚点之间分别从左右传播，按 1/距离 加权融合概率；
首尾无锚点区段单向外推，允许空 mask。

反向采用不复制图像的反序索引视图，再调用未修改的前向 SurgSLOT。
原生 reverse=True 的短期记忆可运行，但当前长期记忆候选按 frame_idx-1 更新，
反序视图可以让两个方向都使用完整的前向记忆逻辑。

每个连通区域/方向使用独立跟踪状态、临时 ID=1，再按类别合并。
不同类别不按处理顺序覆盖；新增窄交叠用联合证据协调，原始交叠和大范围矛盾保留复核。
已有类别也参与关键帧时序融合；其他缺失类别由相邻人工锚点传播补充。
显式保留的区域会锁定该帧同类的后续联合处理，避免按类别合并时改变受保护的几何。
不开发长期实例身份追踪或未标新类别识别。

## 测试与当前限制

~~~bash
python -m unittest discover -s surgslot-sam3/test \
  -p test_labelme_video_annotate.py -v
~~~

测试覆盖默认全面选择、显式保留、最终几何替换、错位/空预测/面积失控回退、
正点位置、相邻锚点、双向状态重置、多区域、重叠、JSON 和 MP4 导出，
以及微圈过滤、可信内部保护、孔洞往返、已有类别时序处理和选择范围保护，共21项。
真实稀疏验证使用第 21–27 帧、仅 3 个标注锚点，35 个区域全部尝试细化；
当前测试数据的一处 fat 在副本上人工粗化，医生源目录不变。
详见 outputs/validation/refine_all_sparse{,_visual,_refined}。

旧版 README 曾记录单例细化 IoU 提升，后续最终产物与该数字不一致；
该数字已撤下，不能作为当前质量证据。现在检查实际导出的候选和原区域保留率。

仍需人工检查语义和边界：局部提示也可能选错相邻组织；检查只约束位置和面积，
不能证明类别正确。局部拒绝的区域先回退，再按默认策略参与时序融合，报告中可明确区分。
标准 Polygon 不表达孔洞，自动孔洞会拆成相邻 Polygon 保留；小区域可能被轮廓过滤。
远距离单向外推可能漂移，人工标签在相邻帧切换也不会被模型擅自纠正。

--anchor-every N 仅用于稀疏化评估，会忽略其余帧 JSON，正常使用保持默认 1。
--keep-work 保留概率和细化 mask 缓存，失败时也保留 FAILED.json 与工作目录。

## 上一版 refine_all 结果（2026-10-19，保留供对照）

上一轮主输出为 outputs/邢静远Segment0006_t193p000s_fps10_n90_refine_all，
对应 _visual 为逐帧叠加和 MP4，_refined 为锚点边界对照。旧版输出未覆盖。

- 四卡对 90 帧、1311 个原始图形全部尝试细化，总耗时 154.5 秒；
  每卡峰值张量显存约 2750 MiB。
- 934 个来源图形在最终 shapes 中实际换成新边界（约 71.2%），
  其中模型 mask 面积扩大 913 个、收缩 21 个，中位面积比约 1.36。
  这描述的是几何变化，不是精度评分。
- 377 个候选因与原区域接触的预测面积超过 3 倍而回退，主要是 bone 和 retractor。
  回退原因逐条保存在 refinement_report.json，不能把这些区域当成已成功细化。
- 通过细化的来源区域导出 1032 个 Polygon，传播另外补充 147 个 Polygon。
- dura 97/97、fat 181/191、lig flavum 205/237、disc 59/60、
  blood vessel 35/36、soft tissue 43/43、bone 217/489、
  retractor 96/157、bubble 1/1 的来源图形实际替换边界。
- 所有 1311 个原始 shape 的完整结构保存在逐帧 JSON 的来源字段，377 个回退几何原样保留；
  源 JSON 哈希不变，原图字节一致。
- 官方 LabelMe 7.5.0 读取器加载了主输出和锚点输出的共 180 个 JSON；
  主 MP4 完整解码为 90 帧、1920×1080、10 fps。
- 真实稀疏样本 35 个区域全部尝试、27 个实际替换、8 个回退。
  之前错位的 fat 首次候选被拒绝，补充正点后通过，原区域保留率约 75%
  （该副本明确设置 rough，允许收缩）；主视频原始 fat 对应区域保留率约 97.5%。
- 90 帧均有至少一项复核提示，包含新增交叠、编辑范围裁剪、孔洞填充、细化回退等。
  有些候选受 32 像素编辑带限制，可能出现截断边缘，需要结合对照图检查。
  不能把“通过位置和面积检查”理解为“边界已被医学确认”。

可审查的只读验证器为 outputs/validation/verify_refine_all.py，
验证报告为 refine_all_verification.json。官方读取器和 Qt 依赖隔离在 /tmp，
不是推理脚本依赖。GPU 提示实验记录了相对/绝对坐标一致；
历史错位在单点/框提示下未能稳定复现，不能断言某一种状态隔离就彻底修复模型歧义。
现在依靠独立单帧状态、分散正点、重试和候选拒绝共同约束结果。

## temporal_v2 的新参数和复核要点

| 参数 | 默认值 | 含义 |
|---|---:|---|
| --source-min-area | 64 | 非保护类别低于此像素面积的源图形记为微小伪标 |
| --artifact-max-area | 1024 | 相对同帧同类主体极小、尺寸不超过80px的孤立微圈上限 |
| --artifact-max-size | 80 | 微圈包围框最长边上限 |
| --protect-small-class | bubble、blood vessel | 默认保护气泡和血管小目标，可重复追加类别 |
| --core-radius | 6 | 非 rough 人工区域的可信内部保护宽度 |
| --temporal-stride | 8 | 已标类别用于跟踪的关键帧间距；0关闭这部分跟踪 |
| --temporal-weight | 0.65 | 时序概率融合权重 |
| --boundary-smooth | 3 | 自动边界的轻微闭运算半径；0关闭 |
| --min-component-area | 64 | 无正区域支持的自动碎片面积下限 |
| --no-boundary-coordination | 默认开启协调 | 此开关关闭新增边缘交叠的联合协调 |

微圈规则还要求：同帧该类存在至少4096像素的主体，候选不超过该主体面积的2%；
超过尺寸上限的细长区域不会按这一相对规则删除。
flags.keep_small=true 或 flags.preserve=true 可显式保留小图形。
可用 --source-min-area 0 --artifact-max-area 0 禁用非空图形的尺寸过滤。

所有原始图形仍保存在 surgslot_original_shapes 中，剔除清单在 surgslot_filtered_shapes
和逐帧复核信息中；源 JSON 不修改，原始 shape 下标不重排。
第5帧从84个源图形保留10个，第20帧从78个保留8个，默认规则全序列剔除353个疑似工具微圈。
这是尺寸和相对面积启发式，真实细小区域也可能误删，需要结合剔除清单复核。

_refined 目录展示的是局部细化锚点，不包含最终的时序和跨类边缘协调；
检查最后效果请看主输出及 _visual。大范围标签矛盾仍需医生确认，
无法从像素和这些稀疏正区域中证明每条边界的医学正确性。

## temporal_v2 实测（2026-10-19）

本轮完整结果为 outputs/邢静远Segment0006_t193p000s_fps10_n90_temporal_v2，
对应 _visual 包含90张叠加图和 preview.mp4，旧输出保留。
四卡运行318.3秒。原始1311个图形保留完整来源，过滤后958个有效图形，
其中923个局部细化采用候选、35个局部回退；最终还有时序融合和类别边缘协调。
587个“类别×帧”实际采用时序证据，平均每帧6.52类。
原始图形未原样留在最终 shapes 不代表来源丢失，按类重建后仍保留完整原始来源。

- 第34帧 dura 在540–580行的错误横向断带消失，主结果中各抽查行均有614像素以上覆盖。
- 对原来相邻两帧都有人标的类别，统一比较545对非空结果：
  相邻帧平均IoU从0.725升至0.758；质心平移对齐后的平均边界距离从37.14降至32.47像素，约改善12.6%。
  指标在半分辨率计算后换算像素，只反映稳定性，受真实运动、类别切换和多区域变化影响，不能证明医学精度。
- 相对有效人工交叠之外的自动新增交叠，按类别对累加从1743254降至677829像素，约减少61.1%。
  大范围语义矛盾仍保留，不强制所有类别互斥或填满空隙。
- 非粗标的内部可信正区域在实际导出Polygon中保留约99.97%，少量差异来自轮廓简化/过滤。
- 官方LabelMe 7.5.0读取主输出和局部候选共180个JSON；MP4完整解码90帧、1920×1080、10fps。
  源JSON哈希未变，输出90张原图与源字节一致；21项契约测试通过。

仍未解决的明确问题：第20帧器械右侧强反光区漏分，retractor的平移对齐边界距离
13.81→13.83像素，基本没有改善；soft tissue的未对齐帧间IoU略降。
当前不依据亮度直接把白色区域归给器械，也不把所有未标区域强行填充。
后续可对这些指定失败帧增加器械正提示或单独测试更大的模型搜索范围，再对照实际边界复核。

只读验证器：outputs/validation/verify_temporal_v2.py；
报告：outputs/validation/temporal_v2_verification.json；
21项测试记录：outputs/validation/temporal_v2_tests.log。
三帧新旧对照图在 outputs/validation/temporal_v2_comparisons/。
