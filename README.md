# 膝关节 MRI 骨龄深度学习评估系统（网页版 v2）

基于深度学习的膝关节 MRI 骨骺发育自动评估网页：上传膝关节 MRI（NIfTI 格式），自动/手动分割骨骺 ROI 后，一键输出**预测骨龄（年龄回归）+ 股骨远端 Vieth 分级 + 胫骨近端 Vieth 分级**，结果页同时展示**当前所选模型在独立测试集上的精度指标**，并可下载内容一致的 PDF 评估报告。全流程在浏览器本地运行（ONNX Runtime WebAssembly），**不上传任何患者数据到服务器，保护隐私**。

> v2 相对上一版的变化：① 3 个原本超过 GitHub 100MB 上限的 CNN 模型已转为 **FP16 半精度**，全部缩小到 100MB 以内，6 个评估模型现可全部直接放进仓库、无需任何远程托管；② 全文措辞由“AI”统一为“深度学习”；③ 删除评估页“数据处理说明”；④ 结果页用“本模型性能指标（独立测试集）”替换原“发育阶段判断”，PDF 报告同步；⑤ PD+T1 ResNet 标注“🏆 最佳模型”。

---

## 一、功能特点

- **6 个评估模型**可选：PD+T1 双模态融合 / PD 单模态 / T1 单模态，自定义 CNN 与 ResNet 两种骨干；其中 **PD+T1 ResNet 标注为🏆最佳模型**（独立测试集综合精度最高）。
- **2 个 U-Net 自动分割模型**：只传原始图像即可自动分割股骨、胫骨骨骺 ROI；也支持上传 ITK-SNAP 等工具手工标注的掩码。
- **多任务输出**：年龄回归值（保留 2 位小数）、股骨/胫骨 Vieth 分级（2–6 级），并随所选模型与性别展示 MAE、RMSE、误差占比、Accuracy、Kappa、QWK 等精度指标。
- **PDF 报告下载**：包含性别、体位、模型、分割方式、预测骨龄、双侧 Vieth 分级、**与结果页一致的模型性能指标**及免责声明。
- **中英文双语**：右上角一键切换。
- **纯前端、离线可用**：所有模型与依赖均为本地文件，无需联网、无需后端、无需安装 Python。

---

## 二、目录结构

```
knee-mri-web-v2/
├── index.html                 # 主网页（唯一入口，经本地服务器或 GitHub Pages 打开）
├── nifti-reader-min.js        # NIfTI(.nii/.nii.gz) 医学影像读取库
├── 启动服务器.bat              # Windows 一键启动本地服务器
├── models/                    # 6 个骨龄评估模型（ONNX，全部 <100MB）
│   ├── model_pd_t1_resnet.onnx   ★最佳模型，PD+T1 双模态 ResNet（FP32，约47MB）
│   ├── model_pd_resnet.onnx      PD 单模态 ResNet（FP32，约21MB）
│   ├── model_t1_resnet.onnx      T1 单模态 ResNet（FP32，约21MB）
│   ├── model_pd_t1_cnn.onnx      PD+T1 双模态 CNN（FP16，约57MB）
│   ├── model_pd_cnn.onnx         PD 单模态 CNN（FP16，约52MB）
│   └── model_t1_cnn.onnx         T1 单模态 CNN（FP16，约52MB）
├── unet_models/               # 2 个 U-Net 自动分割模型（ONNX，各约51MB）
│   ├── knee_pd_unet.onnx
│   └── knee_t1_unet.onnx
├── ort-wasm/                  # ONNX Runtime WebAssembly 运行时（必需）
│   ├── ort.min.js
│   ├── ort-wasm.wasm
│   └── ort-wasm-simd.wasm
└── pdf-libs/                  # 生成 PDF 报告所需
    ├── jspdf.umd.min.js
    └── html2canvas.min.js
```

> 本文件夹是**纯发布版**：已剔除训练代码、Python 依赖、权重(.pth)、临时/备份文件。**所有单文件均 <100MB，可直接 `git push`（含 GitHub Pages）**。

### 关于 FP16 半精度（精度几乎无损）
3 个 CNN 由 FP32 转 FP16（对 Softmax/Sigmoid/BatchNorm/ReduceMean 等数值敏感算子保留 FP32）。在与真实输入一致的 [0,1] 归一化随机数据上、每模型 20 组（男女各 10）与 FP32 逐一对比：

| 模型 | FP32→FP16 体积 | 年龄平均绝对差 | 年龄最大单点差 | 股骨/胫骨分级一致率 |
|---|---|---|---|---|
| model_pd_cnn | 103.3→51.7MB | 0.0042 岁 | 0.0119 岁 | 20/20 |
| model_t1_cnn | 103.3→51.7MB | 0.0072 岁 | 0.0105 岁 | 20/20 |
| model_pd_t1_cnn | 113.4→56.8MB | 0.0030 岁 | 0.0089 岁 | 20/20 |

年龄误差均在 0.01 岁量级（约数天），相对模型本身约 0.75–1.3 岁的 MAE 可忽略；Vieth 分级 argmax 全部一致。FP16 模型也已在**网页真实运行时 onnxruntime-wasm（单线程、无跨域隔离，与 GitHub Pages 同环境）实测可正常加载与推理**。

---

## 三、本地运行（零基础分步）

模型推理依赖 WebAssembly，**不要直接双击 index.html 用 file:// 打开**（浏览器会因安全策略拦截 wasm/模型加载），请用本地服务器：

### 方法 A：Windows 一键脚本（最简单）
1. 进入本文件夹；
2. 双击 **`启动服务器.bat`**（需电脑已安装 Python，安装时勾选 Add to PATH）；
3. 浏览器访问脚本窗口提示的地址：**http://localhost:8000/** ；
4. 用完后在命令行窗口按 `Ctrl + C` 停止。

### 方法 B：命令行手动启动
在本文件夹地址栏输入 `cmd` 回车，然后执行：
```bash
python -m http.server 8000
```
浏览器打开 http://localhost:8000/ 即可。

### 方法 C：VS Code
安装 “Live Server” 插件，右键 index.html → Open with Live Server。

---

## 四、上传到 GitHub，生成可分享链接（GitHub Pages）

### 1. 新建仓库
登录 github.com → 右上角 `+` → `New repository`，仓库名例如 `knee-mri-web-v2`，选 Public，**不要**勾选 Add README（本地已有），创建。

### 2. 本地初始化并推送
在本文件夹内打开 Git Bash 或命令行，依次执行（把用户名换成你自己的）：
```bash
git init
git add .
git commit -m "膝关节MRI骨龄深度学习评估网页 v2（CNN转FP16，全部模型入库）"
git branch -M main
git remote add origin https://github.com/你的用户名/knee-mri-web-v2.git
git push -u origin main
```
> 拖拽上传网页时，`ort-wasm` 里的 `.wasm` 文件有时会被网页上传控件误判；遇到这种情况请改用上面的 git 命令推送，最稳妥。

### 3. 开启 GitHub Pages
仓库页面 → `Settings` → 左侧 `Pages` → `Build and deployment`：
- Source 选 `Deploy from a branch`；
- Branch 选 `main`、目录选 `/ (root)` → `Save`；
- 等 1–2 分钟，页面顶部出现链接，形如 **https://你的用户名.github.io/knee-mri-web-v2/** ，发给别人即可直接使用。

**上传后仍可随时修改**：本地改完执行
```bash
git add .
git commit -m "更新说明"
git push
```
Pages 会在一两分钟内自动更新。

> 国内直连 github.io 可能较慢，属网络问题而非程序问题；访客如遇模型加载慢，耐心等待或换网络即可。

---

## 五、网页使用说明

### 输入要求
- 影像格式：`.nii` / `.nii.gz`（2D 切片，系统自动取第一个切片，与训练一致）；自动分割模式仅需原图，手动分割模式还需对应的 ROI 掩码。
- 双模态模型需同时上传 **T1** 与 **PD** 两套图像；单模态模型只需一种。
- 临床信息：
  - **性别**：1 = 男，2 = 女（对结果影响较大，务必正确，精度指标也随性别切换）；
  - **体位**：1 = 左侧（L），2 = 右侧（R）。多数文件名带 L/R 后缀可据此判断；确实无法区分时任选其一即可，实测对年龄结果影响很小。

### 处理流程（与模型训练代码严格一致）
读取 NIfTI →（自动分割或读取手工掩码）→ 掩码二值化并与原图相乘提取 ROI → LANCZOS 缩放到模型输入尺寸（256 或 224）→ min-max 归一化到 [0,1] → ONNX 推理 → 输出结果。

### 模型选择建议
- 一般情况选默认的 **PD+T1 ResNet（双模态，🏆最佳模型）**，综合精度最好；
- 只有一种序列时，选对应的单模态模型。

---

## 六、技术栈

- 前端：原生 HTML / CSS / JavaScript（单文件，无前端框架）
- 模型推理：ONNX Runtime Web（WebAssembly + SIMD；无跨域隔离时自动单线程）
- 影像解析：nifti-reader
- PDF：jsPDF + html2canvas
- 模型：PyTorch 训练并导出为 ONNX（opset 17），U-Net 分割 + 多任务（分类+回归）评估网络；3 个 CNN 经 onnxconverter-common 转 FP16

---

## 七、常见问题

**Q：双击 index.html 打开后一直转圈 / 报模型加载失败？**
A：file:// 协议会被浏览器拦截，请按第三节用本地服务器打开。

**Q：推送时报 “exceeds GitHub's file size limit of 100 MB”？**
A：v2 已把全部模型压到 100MB 以内，正常不会再出现；若你另外放入了原始 FP32 大模型，请改用本文件夹内的 FP16 版本。

**Q：FP16 会不会让结果不准？**
A：不会有可察觉影响，对比数据见第二节，分级结果与 FP32 完全一致、年龄差在 0.01 岁量级。

**Q：网页是英文的怎么办？**
A：右上角点 “中文 / EN” 切换，选择会被浏览器记住。

**Q：患者数据会上传吗？**
A：不会。所有计算都在访问者本机浏览器内完成，本站点不含任何后端上传逻辑。

**Q：预测结果能替代医生诊断吗？**
A：不能。结果仅供科研与临床参考，最终诊断需由专业医师结合检查判断（PDF 报告内亦含免责声明）。
