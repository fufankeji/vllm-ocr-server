
# <center >《大模型Agent开发实战》（体验课）</center>
## <center>LangChain1.0 + OCR 多模态PDF解析实战</center>
### <center>MinerU & Paddle-OCR & DeepSeek-OCR 深度集成</center>

&emsp;&emsp;<font color=red>**本期公开课，我们将从零开始，手把手带大家将目前行业内性能最强的OCR解析项目：MinerU、DeepSeek-OCR和PaddleOCR-VL通过vLLM推理框架进行高性能部署，并实现具有统一解析服务接口的多模态数据分析系统。**</font>

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511071439387.png" width=80%></div>

&emsp;&emsp;核心功能一：源码部署 MinerU，使用 vLLM 启动推理服务，并以 MCP Server方式与LagnChain完成集成，支持批量解析多模态PDF、图片格式文件；

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511071444347.png" width=80%></div>


&emsp;&emsp;核心功能二：使用 vLLM 推理框架部署启动 DeepSeek-OCR 解析服务接口，并支持批量解析PDF和图像文件格式；

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511071445754.png" width=80%></div>

&emsp;&emsp;核心功能三：本地部署 PaddleOCR-VL ，并通过 PaddleOCR CLI 工具挂载 vLLM 推理服务，提供极高性能的OCR解析服务，支持批量解析PDF和图像文件格式；

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511071447976.png" width=80%></div>

&emsp;&emsp;该项目包含我们针对`DeepSeek-OCR`、`MinerU`等服务接口的独家自研代码程序，及完整的本地部署流程。同时，我们也将提供全部的前后端源码供大家学习和使用，是真正可以直接在企业中落地的项目。

# 一、PDF文件难点与解析方法总览

&emsp;&emsp;<font color=red>目前，无论是做`Agent`智能体应用，还是专注于做`RAG`知识库问答，`PDF`格式文件都是最难处理的文件类型，没有之一。</font>主要原因在于`PDF`格式文件的结构非常复杂，包含文本、图像、表格等多种类型的数据，很多上下文信息都是通过标题、图片、图表和格式等来传达的，我们需要保留这些信息并进行有效的存储，才能够保证在应用阶段正确识别出有效的文本，从而使大模型能够更好地确定如何“思考”给定内容中提供的信息。

&emsp;&emsp;我们这里以`RAG`应用方向为例。

&emsp;&emsp;`RAG`系统往往是需要依靠高质量的结构化数据来生成准确且与文本相关的输出。`PDF` 通常用于官方文件、业务报告和法律合同，包含大量信息，但其布局复杂且数据结构不合理。如果做不到精确的 `PDF` 解析，关键数据就会丢失，从而导致结果不准确并直接影响 `RAG` 应用程序的有效性。因此，`PDF`格式文件的索引和检索的流程，在`RAG`框架中的实现过程如下图所示：

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202503271109933.png" width=50%></div>

&emsp;&emsp;我们在实际构建`RAG`的第一步，往往是先做`PDF Parsing`，即`PDF`解析。所谓的`PDF Parsing`，指的是提取 `PDF` 文件中的内容并将其转换为结构化格式的过程，对于`PDF`格式的这种文件特点，就不能像处理`.txt`、`.csv`、`.json`等文件那样，直接使用`Python`的`open`函数打开文件，然后逐行读取文件内容，然后进行处理。<font color=red>它涉及分析 `PDF` 文件的结构和内容以提取有意义的信息，即把`PDF`文件中的文本、图像、表格和元数据等正确的识别出来，同时还需要解析出`PDF`文件的结构信息，比如：页眉、页脚、页码、章节、段落、标题等。</font>

&emsp;&emsp;<font color=red>在`RAG`处理`PDF`格式文件的解析过程中，将其先转换为`Markdown`文件再做后处理是目前通用的做法。</font> `Markdown` 格式文件自`2023`年起就一直是在大模型领域的最流行格式，像`ChatGPT`、`DeepSeek`等聊天机器人格式化其响应的方式都是使用的`Markdown`语法，包括我们课程中的所有项目案例`Fufan_chat`、`MateGen Pro`和`AssistGen`，采取的做法都是后端实现流式输出，通过`SSE`协议将结果返回给前端，前端再通过`Markdown`语法进行展示。如下所示：`DeepSeek`的响应会以大而粗的字体呈现标题，以及通过使用粗体文本表示关键字。


<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202503271137224.png" width=50%></div>

&emsp;&emsp;如下是 `Markdown` 语法的一些基本示例：

```markdown

        # 一级标题
        ## 二级标题
        ### 三级标题

        **加粗文本**

        *斜体文本*

        > 引用文本

        [链接文本](https://www.example.org)

```
        代码文本
        ```
    
        | 表头1   | 表头2   |
        |----------|----------|
        | 表格数据 | 表格数据 |

&emsp;&emsp;这种结构优势很明显，比如能给标题、表格、列表、链接等提供结构化的信息，添加了印刷强调元素，例如粗体或斜体，还能提供代码块、数学公式、图片、图表等，即易于编写，又易于阅读，即<font color=red>可以保留文档的原始结构，这包括保留布局、顺序以及不同部分（例如页眉、脚注、表格）之间的连接。对大模型来说，阅读和理解这种类型的输入文档上下文中是非常有效的。</font>也正是因为`Markdown`格式文件的这种优势，所以现在<font color=red>主流的处理`PDF`格式文件的库、框架、工具，都是优先针对`PDF --> Markdown` 提取策略展开研究和优化。</font>

- **OCR技术兴起的背景**

&emsp;&emsp;也正是随着这种多模态数据难处理的困境，OCR 技术也迎来了新的快速发展机遇。2025年10月16日发布的 PaddleOCR-VL 模型直接屠榜，在全球权威榜单OmniDocBench V1.5中以92.6分夺得综合性能第一，横扫文本识别、公式识别、表格理解与阅读顺序四项SOTA。

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202510211049426.png" width=60%></div>

&emsp;&emsp;而紧接其后，DeepSeek 也于 2025年10月21日发布了 DeepSeek-OCR 模型，仅需7G的显存，就能完成高精度的表格、公式识别，图片语义识别，并且在多项评测指标中一举拿下SOTA成绩。

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202510211101053.png" width=60%></div>

&emsp;&emsp;这类模型的兴起，完全是源于真实应用需求的驱动。

&emsp;&emsp;OCR（全称：Optical Character Recognition，光学字符识别）是将包含文本的图像（如扫描文档、照片、表单、书页）转换成 机器可读的文本格式的技术，如下图所示：

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202510201841979.png" width=60%></div>

&emsp;&emsp;这类模型并不是传统的OCR模型，而是拥有大模型多模态能力的OCR模型，这种模型被称为VLM（Vision-Language Model）模型、或者OCR 2.0模型。其中最本质的区别，就是多模态大模型在进行图像识别之前，会借助一个名叫视觉编码器的算法，将图像视觉信息映射到文本空间中，然后再借助大模型对文本语义的理解能力，间接的去理解图像信息。

&emsp;&emsp;简单理解就是：你有一张扫描的图片，上面写满了打印或手写的文字，OCR 技术可以把这些“文字图像”变成“真实的文本数据”（可以编辑、搜查、分析）而不是只是“一个图片”而已。

&emsp;&emsp;目前企业在做的大部分业务场景都是以正确识别不同文档类型的内容为前提，比如处理大量纸质文档（发票、合同、报表、表单、收据等），需要通过 OCR 可以把这些纸质/图像文档转换为电子数据，从而更便于存储、检索、分析。
在数字化、智能化流程（例如自动化数据输入、档案管理、人工智能分析）中，它同样也是一个基础环节。

# 二、企业级OCR项目适用场景及优劣势分析

&emsp;&emsp;就目前的发展来看，OCR 与 RAG 的技术结合并不像简单的传统 OCR 是“你给我一张图像，我把文字提取出来”就可以了，重点不能仅仅放在去提取文字，而是要在 OCR 的基础上，不仅仅是“图像中的文字”被识别，还需要结合 视觉（图像）、语言（文本）、版面结构（布局）、场景环境/上下文 等多个模态（modalities）一起分析，使得识别和理解更强、更智能。

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202510201346376.png" width=70%></div>

&emsp;&emsp;现代文档不仅仅是文本。它们包含多列布局、数学公式、半扫描表格、多语言文本以及分辨率奇数的图表。像 GPT-4o 或 Qwen-VL 这样的端到端模型可以解析它们，但它们速度慢、布局混乱，并且耗费 GPU 内存。所以企业环境下往往会选择更小、更紧凑的视觉模型来为解析工作提供支持。主流的企业级OCR项目应用如下：

- MinerU:[点击进入](https://github.com/opendatalab/MinerU)

&emsp;&emsp;MinerU 是由 OpenDataLab（上海人工智能实验室下属团队）发起的一个开源工具，目标是将 PDF（含扫描件、复杂版式、多栏、多表格、多公式）转换为可机读的结构化格式（如 Markdown、JSON）以便进一步下游使用。项目的定位更偏「文档内容抽取／结构化」而不仅仅是传统 OCR。其取向是“将 PDF → Markdown/JSON”这一流程，而不仅“图片 → 文字”。

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202503271440433.png" width=70%></div>

&emsp;&emsp;`MinerU`的主要工作流程分为以下几个阶段：

1. **输入**：接收`PDF` 格式文本，可以是简单的纯文本，也可以是包含双列文本、公式、表格或图等多模态`PDF`文件;
2. **文档预处理（Document Preprocessing）**：检查语言、页面大小、文件是否被扫描以及加密状态；
3. **内容解析（Content Parsing）**：
    - 局分析：区分文本、表格和图像。
    - 公式检测和识别：识别公式类型（内联、显示或忽略）并将其转换为 `LaTeX` 格式。
    - 表格识别：以 `HTML/LaTeX` 格式输出表格。
    - OCR：对扫描的 `PDF` 执行文本识别。
4. **内容后处理（Content Post-processing）**：修复文档解析后可能出现的问题。比如解决文本、图像、表格和公式块之间的重叠，并根据人类阅读模式重新排序内容，确保最终输出遵循自然的阅读顺序。
5. **格式转换（Format Conversion）**：以 `Markdown` 或 `JSON` 格式生成输出。
6. **输出（Output）**：高质量、结构良好的解析文档。

&emsp;&emsp;如下是`MinerU`项目官方给出的配置参考：

<div align="center">
    <img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511071459046.png" alt="MinerU 配置参考" width="70%>
</div>

&emsp;&emsp;`VLM-Transformer`则是直接利用`transformers`库中的`Vision-Language`模型处理图像+文本的多模态输入，其流程如下：

<div align="center">
    <img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202508061356866.png" alt="MinerU 配置参考" width="800">
</div>

&emsp;&emsp;而`VLM-Sglang`后端则是利用`sglang`高性能推理引擎，优化了GPU加速及分布式部署的基础上同时支持实时流式输出，其流程如下：

<div align="center">
    <img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202508061356864.png" alt="MinerU 配置参考" width="800">
</div>

&emsp;&emsp;MinerU 项目非常适用于：

- 需要从大量 PDF 文档中抽取结构化内容（例如学术文献、技术白皮书、报告）用于知识库或训练语料。

- 对版式结构（如章节、列表、表格、公式）要求较高，而不只是 OCR 文本识别。

- 希望输出 Markdown／JSON 供后续自动化流水线使用。

- PaddleOCR：[点击进入](https://github.com/PaddlePaddle/PaddleOCR/tree/main)

&emsp;&emsp;PaddleOCR 是由 Baidu (及其生态) 基于其深度学习框架 PaddlePaddle 提供的开源 OCR 工具箱。支持从 PDF 或图像文档转为结构化数据（适配 AI 场景），支持 100+ 语言。最新版本 3.0 在其技术报告中提出：PP-OCRv5、PP-StructureV3、PP-ChatOCRv4 三大解决方案，覆盖文字识别、多版式文档解析、关键 信息提取。

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202510211220985.png" width=70%></div>

&emsp;&emsp;在早期版本（如 PP-OCRv3）中，其结构可概括为：“检测 (Detection) → 分类 (Classification of orientation) → 识别 (Recognition)”。使用多种模型，例如检测模型（DBNet 等）、识别模型（如 SVTR），在 3.0 版本中，其“PP-StructureV3”整合了布局分析、表格识别、结构抽取。同时还最新推出了还推出了PaddleOCR-VL 的 Vision-Language 模型版本（0.9B 参数的 VLM），用于多语种文档解析。

&emsp;&emsp;PaddleOCR-VL 是推出的一个专注于“文档解析／视觉-语言模型 (Vision-Language Model, VLM)”功能的新模块，采用了视觉-语言模型架构以应对更高阶的需求。在解析多模态数据方面，PaddleOCR将这项工作分为两部分：

1. 首先检测并排序布局元素。
2. 使用紧凑的视觉语言模型精确识别每个元素。

&emsp;&emsp;该系统分为两个明确的阶段运行。

&emsp;&emsp;第一阶段是执行布局分析（PP-DocLayoutV2），此部分标识文本块、表格、公式和图表。它使用：
- RT-DETR 用于物体检测（基本上是边界框 + 类标签）。
- 指针网络 （6 个转换器层）可确定元素的读取顺序 ，从上到下、从左到右等。

&emsp;&emsp;最终输出统一模式的图片标注数据，如下图所示：

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202510211239047.png" width=70%></div>

&emsp;&emsp;第二阶段则是元素识别（PaddleOCR-VL-0.9B），这就是视觉语言模型发挥作用的地方。它使用：
- NaViT 风格编码器 （来自 Keye-VL），可处理动态图像分辨率。无平铺，无拉伸。
- 一个简单的 2 层 MLP， 用于将视觉特征与语言空间对齐。
- ERNIE-4.5–0.3B 作为语言模型，该模型规模虽小但速度很快，并且采用 3D-RoPE 进行位置编码

&emsp;&emsp;最终模型输出结构化 Markdown 或 JSON 格式的文件用于后续的处理。

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202510211243206.png" width=70%></div>

&emsp;&emsp;这个小小的设计决策， 将布局和识别分离，使得 PaddleOCR-VL 比通常的一体化系统更快、更稳定。同时根据实际的测试，其运行和解析速度也更快。在 A100 GPU 上， 吞吐量为 1.22 页/秒，。比 MinerU2.5 快 15.8%， VRAM 比 dots.ocr 少约 40%。

- DeepSeek-OCR：[点击进入](https://github.com/deepseek-ai/DeepSeek-OCR)

&emsp;&emsp;DeepSeek 于 2025年10月21日发布了 DeepSeek-OCR 模型，仅需7G的显存，就能完成高精度的表格、公式识别，图片语义识别，并且在多项评测指标中一举拿下SOTA成绩。其 Github 相较于其他项目相对比较"简陋"，仅仅提供了`Transformers` 和 `vLLM` 启动 `DeepSeek-OCR`的服务示例代码文件：

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511071504454.png" width=70%></div>

- dots.ocr：[点击进入](https://github.com/dots-ai/dots.ocr)


&emsp;&emsp;dots.ocr 是由 rednote‑hilab（HiLab团队）开源的多语种文档布局解析工具。官方介绍中强调：“一个统一的 Vision-Language 模型（≈1.7 B 参数）即可完成布局检测 + 内容识别 +阅读顺序排序”。支持文本、表格、公式、以及多语言输入。

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202510211151021.png" width=70%></div>

&emsp;&emsp;dots.ocr 的特点是用一个 VLM（1.7 B 参数）来统一布局解析+内容识别，而不是传统将检测、识别、结构分开。用户可通过不同 prompt 来切换任务（如“请输出版式元素的 bbox、类别、文本”）→ 即说明模型采用 prompt + VLM 的方式。

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202510211229592.png" width=70%></div>

&emsp;&emsp;非常适合需要快速处理多语种、混版式文档，且希望用一个统一模型／prompt 来搞定。虽然表现不错，但对于极复杂的表格（如跨页表、合并单元格）或特殊版式效果并不是很理想。

# 三、MinerU 项目概览与vllm 服务接口

&emsp;&emsp;`MinerU`是一个非常典型的基于管道的解决方案 (Pipeline-based solution) ，并且是一个开源文档解析项目，一共四个核心组件，通过`Pipeline`的设计无缝衔接，实现比较高效、准确的文档解析。如下图所示：（下图来源官方论文：https://arxiv.org/pdf/2409.18839v1）

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202503271440433.png" width=70%></div>

&emsp;&emsp;`MinerU`的主要工作流程分为以下几个阶段：

1. **输入**：接收`PDF` 格式文本，可以是简单的纯文本，也可以是包含双列文本、公式、表格或图等多模态`PDF`文件;
2. **文档预处理（Document Preprocessing）**：检查语言、页面大小、文件是否被扫描以及加密状态；
3. **内容解析（Content Parsing）**：
    - 局分析：区分文本、表格和图像。
    - 公式检测和识别：识别公式类型（内联、显示或忽略）并将其转换为 `LaTeX` 格式。
    - 表格识别：以 `HTML/LaTeX` 格式输出表格。
    - OCR：对扫描的 `PDF` 执行文本识别。
4. **内容后处理（Content Post-processing）**：修复文档解析后可能出现的问题。比如解决文本、图像、表格和公式块之间的重叠，并根据人类阅读模式重新排序内容，确保最终输出遵循自然的阅读顺序。
5. **格式转换（Format Conversion）**：以 `Markdown` 或 `JSON` 格式生成输出。
6. **输出（Output）**：高质量、结构良好的解析文档。

&emsp;&emsp;目前在`Github` 上，`MinerU` 的`Star` 数为`48.2K`，`Fork` 数为`4K`，拥有非常良好的社区支持和活跃的贡献者，且一直处于`active development`状态。

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511071014588.png" width=80%></div>

&emsp;&emsp;`MinerU` 提供了在线`Demo` 页面，我们可以直接线进行测试。试用地址：https://opendatalab.com/OpenSourceTools/Extractor/PDF/

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202503271502432.png" width=70%></div>

&emsp;&emsp;同时，`MinerU` 项目于`2024年07月05日`首次开源，底层主要是集成 `PDF-Extract-Kit` 开源项目做`PDF`的内容提取，`PDF-Extract-Kit`同样是一个开源项目：https://github.com/opendatalab/PDF-Extract-Kit

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202503251117693.png" width=70%></div>

&emsp;&emsp;`PDF-Extract-Kit`这个项目主要针对的是`PDF`文档的内容提取，通过集成众多`SOTA`模型对`PDF`文件实现高质量的内容提取，其中应用到的模型主要包括：

<style>
.center 
{
  width: auto;
  display: table;
  margin-left: auto;
  margin-right: auto;
}
</style>

<p align="center"><font face="黑体" size=4>PDF-Extract-Kit 应用的模型类型</font></p>
<div class="center">


| 模型类型     | 模型名称                    | GitHub 链接                                                  | 模型下载链接                                                 | 任务描述                                                   |
| ------------ | --------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ---------------------------------------------------------- |
| 布局检测模型 | LayoutLMv3 / DocLayout-YOLO | [GitHub](https://github.com/microsoft/unilm/tree/master/layoutlmv3) / [GitHub](https://github.com/opendatalab/DocLayout-YOLO) | [模型下载](https://huggingface.co/microsoft/layoutlmv3-base-chinese) / [模型下载](https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench/tree/main) | 定位文档中不同元素位置：包含图像、表格、文本、标题、公式等 |
| 公式检测模型 | YOLO                        | [GitHub](https://github.com/ultralytics/ultralytics)         | [模型下载](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt) | 定位文档中公式位置：包含行内公式和行间公式                 |
| 公式识别模型 | UniMERNet                   | [GitHub](https://github.com/opendatalab/UniMERNet)           | [模型下载](https://huggingface.co/wanderkid/unimernet_base)  | 识别公式图像为latex源码                                    |
| 表格识别模型 | StructEqTable               | [GitHub](https://github.com/Alpha-Innovator/StructEqTable-Deploy?tab=readme-ov-file) | [模型下载](https://huggingface.co/U4R/StructTable-InternVL2-1B/tree/main) | 识别表格图像为对应源码（Latex/HTML/Markdown）              |
| OCR模型      | PaddleOCR                   | [GitHub](https://github.com/PaddlePaddle/PaddleOCR)          | [模型下载](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/ocr_modules/text_detection.html) | 提取图像中的文本内容（包括定位和识别）                     |
| </div>       |                             |                                                              |                                                              |                                                            |


&emsp;&emsp;`MinerU`与`PDF-Extract-Kit`的关系是：`MinerU` 结合`PDF-Extract-Kit`输出的高质量预测结果，进行了专门的工程优化，使得文档内容提取更加便捷高效，处理底层原理的优化细节外，主要提升点在以下几点：

- 加入了自研的`doclayout_yolo(2501)`模型做布局检测，在相近解析效果情况下比原方案提速10倍以上，可以通过配置文件与 `layoutlmv3` 自由切换使用；
- 加入了自研的`unimernet(2501)` 模型做公式识别，针对真实场景下多样性公式识别的算法，可以对复杂长公式、手写公式、含噪声的截图公式均有不错的识别效果；
- 增加 `OCR` 的多语言支持，支持 `84` 种语言的检测与识别，支持列表：https://paddlepaddle.github.io/PaddleOCR/latest/ppocr/blog/multi_languages.html#5 
- 重构排序模块代码，使用 `layoutreader` 进行阅读顺序排序，确保在各种排版下都能实现极高准确率 : https://github.com/ppaanngggg/layoutreader
- 表格识别功能接入了`StructTable-InternVL2-1B`模型，大幅提升表格识别效果，模型下载地址:https://huggingface.co/U4R/StructTable-InternVL2-1B

&emsp;&emsp;在部署和使用方面，`MinerU` 支持`Linux`、`Windows`、`MacOS` 多平台部署的本地部署，并且其中用的到`布局识别模型`、`OCR` 模型、`公式识别模型`、`表格识别模型`都是开源的，我们可以直接下载到本地进行使用。而且，`MinerU` 项目是完全支持华为`昇腾`系列芯片的，可适用性非常广且符合国内用户的使用习惯。


&emsp;&emsp;因此，接下来我们将重点介绍如何在`Linux` 系统下部署`MinerU` 项目并进行`PDF` 文档解析流程实战。注意：这里强烈建议大家使用`Linux` 系统进行部署，其支持性和兼容性远高于`Windows` 系统。如果大家想选择其他操作系统，可以参考如下链接进行自行实践：[Windows 10/11 + GPU ](https://github.com/opendatalab/MinerU/blob/master/docs/README_Windows_CUDA_Acceleration_zh_CN.md)，[Docker 部署](https://github.com/opendatalab/MinerU/blob/master/README_zh-CN.md#%E4%BD%BF%E7%94%A8mps) ,


## 3.1 MinerU 源码安装

&emsp;&emsp;针对MinerU这个项目，其提供了 MCP 服务的配置文件，但是存在很多`BUG`, 我们必须手动修改相关的源码才能适配如`LangChain`等框架中，所以这里我们建议大家采用`Linux`系统下的源码安装方式。如下是`MinerU`项目官方给出的配置参考：

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511051750499.png" width=70%></div>


&emsp;&emsp;`MinerU` 实现了三个不同的处理后端，每个后端针对不同的用例和硬件配置进行了优化。其中`Pipeline`后端是默认的，也是最通用的后端，其采用传统的机器学习流程：

1. **输入**：接收`PDF` 格式文本，可以是简单的纯文本，也可以是包含双列文本、公式、表格或图等多模态`PDF`文件;
2. **文档预处理（Document Preprocessing）**：检查语言、页面大小、文件是否被扫描以及加密状态；
3. **内容解析（Content Parsing）**：
    - 局分析：区分文本、表格和图像。
    - 公式检测和识别：识别公式类型（内联、显示或忽略）并将其转换为 `LaTeX` 格式。
    - 表格识别：以 `HTML/LaTeX` 格式输出表格。
    - OCR：对扫描的 `PDF` 执行文本识别。
4. **内容后处理（Content Post-processing）**：修复文档解析后可能出现的问题。比如解决文本、图像、表格和公式块之间的重叠，并根据人类阅读模式重新排序内容，确保最终输出遵循自然的阅读顺序。
5. **格式转换（Format Conversion）**：以 `Markdown` 或 `JSON` 格式生成输出。
6. **输出（Output）**：高质量、结构良好的解析文档。

<div align="center">
    <img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202503271440433.png" alt="MinerU 配置参考" width="800">
</div>

&emsp;&emsp;`VLM-Transformer`则是直接利用`transformers`库中的`Vision-Language`模型处理图像+文本的多模态输入，其流程如下：

<div align="center">
    <img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202508061356866.png" alt="MinerU 配置参考" width="800">
</div>

&emsp;&emsp;而`VLM-Sglang`后端则是利用`sglang`高性能推理引擎，优化了GPU加速及分布式部署的基础上同时支持实时流式输出，其流程如下：

<div align="center">
    <img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202508061356864.png" alt="MinerU 配置参考" width="800">
</div>

&emsp;&emsp;各个后端的默认模型配置为：

<style>
.center 
{
  width: auto;
  display: table;
  margin-left: auto;
  margin-right: auto;
}
</style>

<p align="center"><font face="黑体" size=4>MinerU 后端默认模型配置</font></p>
<div class="center">

| 后端类型            | 默认模型            | 适用场景   | 资源需求 |
| ------------------- | ------------------- | ---------- | -------- |
| **VLM-Transformer** | MinerU2.5-2509-1.2B | 高精度解析 | 中等 GPU |
| **VLM-SGLang**      | MinerU2.5-2509-1.2B | 高速推理   | 高端 GPU |
| **Pipeline**        | PDF-Extract-Kit-1.0 | 轻量部署   | CPU 友好 |
| </div>              |                     |            |          |

&emsp;&emsp;`MinerU` 的 `VLM` 后端默认使用的是自研的 `MinerU2.5-2509-1.2B` 模型，这是一个基于 `Qwen` 架构、专门为文档解析优化的 `1.2B` 参数多模态模型。而`PDF-Extract-Kit-1.0`，则是沿用了旧版本的基于 `YOLO`、`OCR`、表格识别等传统独立 `CV` 子模型的模块化架构。

&emsp;&emsp;`MinerU` 的新版本不再严格限制`python 3.10`版本，但建议大家使用`python 3.11`或者`python 3.12`，`python 3.13`可能存在兼容问题。

- **Step 1. 确认系统版本**

&emsp;&emsp;我们使用的是`Ubuntu 22.04` 系统，可以通过`cat /etc/os-release` 命令查看系统版本，如下图所示：

<div align="center">
    <img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511061214148.png" alt="MinerU 配置参考" width="70%">
</div>


- **Step 2. 确认`CUDA` 版本**

&emsp;&emsp;在`Linux` 系统下，可以通过`nvidia-smi` 命令查看`CUDA` 版本，这里的服务器配置是四卡的`RTX 3090` 显卡，如下图所示：

<div align="center">
    <img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511061214930.png" alt="MinerU 配置参考" width="70%">
</div>

&emsp;&emsp;`CUDA Version` 显示的版本号必须 `>= 12.1`，如显示的版本号小于`12.1`，需要自行升级`CUDA` 版本。

- **Step 3. 确认 `Conda` 版本**

&emsp;&emsp;我们使用的是`Anaconda` 安装的`Conda` 环境，可以通过`conda --version` 命令查看`Conda` 版本，如下图所示：

<div align="center">
    <img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511061215266.png" alt="MinerU 配置参考" width="70%">
</div>

&emsp;&emsp;如果出现`Conda not found` 等报错，需要先安装`Conda` 环境，再执行接下来的步骤。

- **Step 4. 使用`Conda` 创建`Python 3.11` 版本的虚拟环境**

&emsp;&emsp;执行如下命令创建一个全新的虚拟环境

```bash 
    conda create --name mineru_2.5 python==3.11 -y
```

<div align="center">
    <img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511061214149.png" alt="MinerU 配置参考" width="70%">
</div>

- **Step 5. 激活虚拟环境**

&emsp;&emsp;创建完虚拟环境后，使用`Conda` 激活虚拟环境，通过`conda activate mineru_2.5` 命令激活，如下图所示：

<div align="center">
    <img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511061214150.png" alt="MinerU 配置参考" width="70%">
</div>

- **Step 6. 下载MinerU源码文件**

&emsp;&emsp;截止目前最新的`MinerU`项目源码版本为是：`2.6.4-released`，但当前版本的源码在使用`MCP Server`时存在一些`Bug`，所以我们针对源码做了一些修改。因此大家一定要下载我们课程网盘中的源码文件进行部署和使用。下载后上传至服务器中，解压文件，并进入解压后的文件夹，如下图所示：

<div align="center">
    <img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511061214151.png" alt="MinerU 配置参考" width="70%">
</div>

&emsp;&emsp;<font color> 注意，大家这里一定要使用我们课程提供的源码版本，因为我们对源码进行了非常多的修改才能适配`MCP`服务下使用`MinerU vllm`解析服务。 </font>

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511071521203.png" width=70%></div>

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511071522837.png" width=30%></div>

- **Step 7. 安装`MinerU` 项目依赖**

&emsp;&emsp;使用`Conda` 安装`MinerU` 项目依赖，需要通过如下命令在新建的`mineru` 虚拟环境中安装运行`MinerU` 项目程序的所有依赖，命令如下：

```bash
    pip install -e .[all] -i https://mirrors.aliyun.com/pypi/simple
```

<div align="center">
    <img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511061214152.png" alt="MinerU 配置参考" width="87%">
</div>

&emsp;&emsp;通过`pip show mineru` 命令查看`MinerU` 项目的版本：

<div align="center">
    <img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511061214153.png" alt="MinerU 配置参考" width="70%">
</div>

&emsp;&emsp;至此，基础的`MinerU` 项目依赖就安装完成了，接下来我们需要下载`MinerU` 项目中用到的模型文件，并进行项目配置。

- **Step 8. 下载`MinerU` 项目中用到的模型文件**

&emsp;&emsp;新版本的`MinerU` 项目提供了一键下载所有模型文件的脚本，我们只需要执行如下命令即可：

```bash
    mineru-models-download
```
&emsp;&emsp;执行该命令需要我们根据自己的需求灵活的选择要下载到本地的模型。其中`modelscope` 和 `huggingface` 是两个不同的模型下载源，我们只需要选择其中一个即可。(注意：国内用户建议使用`modelscope` 下载模型文件)。其次，`pipeline`指的是用于文档解析的一系列模型，而`vlm`指的是用于视觉语言模型的模型，即`MinerU2.0-2505-0.9B` 模型。如果是`all`，则会全部下载。


<div align="center">
    <img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511061214154.png" alt="MinerU 配置参考" width="80%">
</div>

&emsp;&emsp;等待下载完成后，所有模型文件的默认存储路径是：

```bash
    /root/.cache/modelscope/hub/models/OpenDataLab
```

<div align="center">
    <img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511061214155.png" alt="MinerU 配置参考" width="80%">
</div>

&emsp;&emsp;并且，也会自动在`/root/mineru.json` 文件中配置用于后续使用的模型路径。

<div align="center">
    <img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511061214156.png" alt="MinerU 配置参考" width="80%">
</div>

&emsp;&emsp;至此，`MinerU` 项目的本地配置就全部完成了，接下来我们可以尝试运行`MinerU` 项目并进行`PDF` 文档解析测试。

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511071534986.png" width=70%></div>

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511071522837.png" width=30%></div>

## 3.2 MinerU 启动vLLM API 推理服务

&emsp;&emsp;接下来新打开一个终端，启动`mineru-api` 服务：

```bash
    export MINERU_MODEL_SOURCE=local  # 注意：这里需要将模型源设置为本地
    export CUDA_VISIBLE_DEVICES=3  # 使用 哪一块GPU
    mineru-api --host 192.168.110.131 --port 50000
```
<div align="center">
    <img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511071609272.png" alt="MinerU 配置参考" width="70%">
</div>


&emsp;&emsp;启动后，在`192.168.110.131:50000/docs`中可以看到接口服务：

<div align="center">
    <img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511061234642.png" alt="MinerU 配置参考" width="70%">
</div>

## 3.3 MinerU vLLM API服务连接测试


```python
#!/usr/bin/env python3
"""
测试 MinerU API 的不同 backend
通过 50000 端口调用，传递不同的 backend 参数来使用不同的模型
"""

import requests
import sys
from pathlib import Path

def test_mineru_api(pdf_path: str, backend: str = "pipeline"):
    """
    测试 MinerU API

    Args:
        pdf_path: PDF 文件路径
        backend: 后端类型，可选值：
                - "pipeline" (默认，使用本地 PyTorch)
                - "vlm-vllm-async-engine" (使用 vLLM 加速)
    """
    api_url = "http://192.168.130.4:50000/file_parse"

    print(f"\n{'='*60}")
    print(f"测试 MinerU API with backend: {backend}")
    print(f"{'='*60}")
    print(f"API URL: {api_url}")
    print(f"PDF 文件: {pdf_path}")
    print(f"Backend: {backend}")
    print()

    # 检查文件是否存在
    if not Path(pdf_path).exists():
        print(f"错误: 文件不存在 - {pdf_path}")
        return None

    try:
        # 打开文件并发送请求
        with open(pdf_path, 'rb') as f:
            files = [('files', (Path(pdf_path).name, f, 'application/pdf'))]
            data = {
                'backend': backend,
                'parse_method': 'auto',
                'lang_list': 'ch',
                'return_md': 'true',
                'return_middle_json': 'false',
                'return_model_output': 'false',
                'return_content_list': 'false',
                'start_page_id': '0',
                'end_page_id': '1',  # 只处理前2页，快速测试
            }

            print("发送请求...")
            response = requests.post(
                api_url,
                files=files,
                data=data,
                timeout=300
            )

        # 检查响应
        if response.status_code != 200:
            print(f"请求失败: HTTP {response.status_code}")
            print(f"响应: {response.text[:500]}")
            return None

        # 解析 JSON 响应
        result = response.json()

        # 提取信息
        backend_used = result.get('backend', 'unknown')
        version = result.get('version', 'unknown')
        results = result.get('results', {})

        print(f"请求成功!")
        print(f"   使用的 backend: {backend_used}")
        print(f"   版本: {version}")
        print(f"   结果数量: {len(results)}")

        # 提取 markdown 内容
        if results:
            file_key = list(results.keys())[0]
            md_content = results[file_key].get('md_content', '')

            print(f"\nMarkdown 内容预览 (前500字符):")
            print("-" * 60)
            print(md_content[:500])
            print("-" * 60)

            # 保存 markdown 到文件
            output_file = f"output_{backend}.md"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(md_content)
            print(f"\n完整 Markdown 已保存到: {output_file}")

            return md_content
        else:
            print("未找到结果")
            return None

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None
```


```python
def main():
    # 默认测试文件
    pdf_path = "./2507.05595v1.pdf"


    print(f"\n开始测试 MinerU API 的不同 backend")
    print(f"测试文件: {pdf_path}\n")

    # 测试 1: pipeline backend (本地 PyTorch)
    print("\n" + "="*60)
    print("测试 1: pipeline backend (本地 PyTorch)")
    print("="*60)
    result_pipeline = test_mineru_api(pdf_path, backend="pipeline")

    # 测试 2: vLLM backend (vLLM 加速)
    print("\n" + "="*60)
    print("测试 2: vlm-vllm-async-engine backend (vLLM 加速)")
    print("="*60)
    result_vllm = test_mineru_api(pdf_path, backend="vlm-vllm-async-engine")

    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    print(f"Pipeline backend: {'成功' if result_pipeline else '失败'}")
    print(f"vLLM backend: {'成功' if result_vllm else '失败'}")

    if result_pipeline and result_vllm:
        print("\n所有测试通过! MinerU 可以通过 backend 参数切换不同模型")

    print("\n💡 提示:")
    print("  - pipeline: 使用本地 PyTorch，适合调试")
    print("  - vlm-vllm-async-engine: 使用 vLLM 加速，速度更快")


if __name__ == "__main__":
    main()
```

当然，上述代码也是`LangChain1.0 + OCR 多模态PDF解析实战`项目中`MinerU` 的核心代码逻辑：

<div align="center">
    <img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511071625811.png" alt="MinerU 配置参考" width="70%">
</div>

`MinerU` 也提供了命令行离线解析、`Gradio/WebUI`、独立 `FastAPI` 服务（mineru-api）等多种启动方式。不同方式适合的场景、端口、性能与依赖各不相同，如下所示：

<style>
.center {
  width: auto;
  display: table;
  margin-left: auto;
  margin-right: auto;
}
</style>

<p align="center"><font face="黑体" size="4"> MinerU 启动方式</font></p>
<div class="center">

| 启动方式                  | 典型命令/镜像                                                                    | 端口 & 路径            | 适用场景             | 依赖                                 |
| --------------------- | -------------------------------------------------------------------------- | ------------------ | ---------------- | ---------------------------------- |
| **SGLang 后端服务**       | `mineru-sglang-server --host 0.0.0.0 --port 30000`                         | `POST /file_parse` | 高并发、GPU 推理、本地私有化 | Python + CUDA/SGLang |
| **Docker 一键**         | `docker run -p 30000:30000 opendatalab/mineru:latest mineru-sglang-server` | 同上                 | 免环境配置、快速试用       | Docker (GPU 可选)       |
| **命令行离线解析**           | `mineru -p <输入> -o <输出>`                                                   | 无服务；直接落盘           | 批处理、无网络、脚本化      | 纯 Python CLI     |
| **Gradio WebUI**      | `mineru-gradio --enable-sglang-engine`                                     | 浏览器 UI             | 交互演示、无代码使用       | Gradio 前端       |
| **FastAPI API 服 务**   | `git clone .../mineru-api && docker compose up`                            | `PUT /api/parse`   | 需要依托自定义业务接口      | FastAPI + Uvicorn     |
| **MCP-Server（外壳）**    | `uv run mineru-mcp --transport streamable-http`                            | `/mcp` 或 `/sse`    | 给 ADK/Agent 暴露工具 | fastmcp 框架           |
| **n8n/Serverless 封装** | n8n「MCP Client」节点或 Runpod 镜像                                               | 由平台分配              | 无代码工作流           | 平台函数                     |

实际上，MinerU 会输出非常丰富的信息，主要是：

1. images:这是一个文件夹，里面存储了`PDF` 文档中出现的所有图片；
2. .origin.pdf：进行解析的原始`PDF`文件；
3. .markdown：`PDF` 文档的解析结果，输出为`xxx.md` 的 `Markdown` 格式；
4. .layout.pdf：这个文件用不同的背景色块圈定不同的内容块，以此来区分整体的布局识别结果；

5. .model.json：这个文件的主要作用是将 `PDF` 从像素级别的展示转换为结构化的数据，使计算机能够"理解"文档的组成部分。主要存储的信息是：

- 文档布局识别：存储文档中的各种元素（标题、正文、图表等）及其在页面上的精确位置
- 内容分类：将识别出的元素分类为不同类型（如标题、表格、公式等）
- 特殊内容解析：对于公式、表格等特殊内容，提供其结构化表示（如LaTeX、HTML格式）


&emsp;&emsp; 其示例数据如下：

```json
    [
        {
            "layout_dets": [
                {
                    "category_id": 2,
                    "poly": [
                        99.1906967163086,
                        100.3119125366211,
                        730.3707885742188,
                        100.3119125366211,
                        730.3707885742188,
                        245.81326293945312,
                        99.1906967163086,
                        245.81326293945312
                    ],
                    "score": 0.9999997615814209
                }
            ],
            "page_info": {
                "page_no": 0,
                "height": 2339,
                "width": 1654
            }
        },
        {
            "layout_dets": [
                {
                    "category_id": 5,
                    "poly": [
                        99.13092803955078,
                        2210.680419921875,
                        497.3183898925781,
                        2210.680419921875,
                        497.3183898925781,
                        2264.78076171875,
                        99.13092803955078,
                        2264.78076171875
                    ],
                    "score": 0.9999997019767761
                }
            ],
            "page_info": {
                "page_no": 1,
                "height": 2339,
                "width": 1654
            }
        }
    ]
```

&emsp;&emsp;其中每一个`layout_dets` 中存储了`PDF` 文档中每一页的布局识别结果，`poly` 是四边形坐标, 分别是 左上，右上，右下，左下 四点的坐标，形式为：`[x0, y0, x1, y1, x2, y2, x3, y3], 分别表示左上、右上、右下、左下四点的坐标`，`page_info` 是`PDF` 文档中每一页元数据，包含页码、高度、宽度信息，而`category_id` 为`PDF` 文档中每一页的布局识别结果的类型，分别为：

<style>
.center 
{
  width: auto;
  display: table;
  margin-left: auto;
  margin-right: auto;
}
</style>

<p align="center"><font face="黑体" size=4>category_id 的类别</font></p>
<div class="center">

| 类别 ID | 类别名称           | 描述                                   |
|---------|--------------------|----------------------------------------|
| 0       | title              | 标题                                   |
| 1       | plain_text         | 文本                                   |
| 2       | abandon            | 包括页眉页脚页码和页面注释            |
| 3       | figure             | 图片                                   |
| 4       | figure_caption      | 图片描述                               |
| 5       | table              | 表格                                   |
| 6       | table_caption      | 表格描述                               |
| 7       | table_footnote     | 表格注释                               |
| 8       | isolate_formula     | 行间公式                               |
| 9       | formula_caption     | 行间公式的标号                         |
| 13      | embedding          | 行内公式                               |
| 14      | isolated           | 行间公式                               |
| 15      | text               | OCR 识别结果                           |

</div>

&emsp;&emsp;这个`.json`文件主要可以内容提取，比如精确提取特定类型的内容（如所有表格或公式），搜索，比如实现对文档内容的结构化搜索，文档分析，比如分析文档的结构特征（如有多少图表、公式密度等）等需求场景。

6. _middle.json：这个文件的核心作用是存储 `PDF` 文档解析过程中多层次结构化数据，它比 `model.json` 提供了更详细的层次结构和内容信息, 其示例数据是这样的：

```json
    {
        "pdf_info": [
            {
                "preproc_blocks": [
                    {
                        "type": "text",
                        "bbox": [
                            52,
                            61.956024169921875,
                            294,
                            82.99800872802734
                        ],
                        "lines": [
                            {
                                "bbox": [
                                    52,
                                    61.956024169921875,
                                    294,
                                    72.0000228881836
                                ],
                                "spans": [
                                    {
                                        "bbox": [
                                            54.0,
                                            61.956024169921875,
                                            296.2261657714844,
                                            72.0000228881836
                                        ],
                                        "content": "dependent on the service headway and the reliability of the departure ",
                                        "type": "text",
                                        "score": 1.0
                                    }
                                ]
                            }
                        ]
                    }
                ],
                "layout_bboxes": [
                    {
                        "layout_bbox": [
                            52,
                            61,
                            294,
                            731
                        ],
                        "layout_label": "V",
                        "sub_layout": []
                    }
                ],
                "page_idx": 0,
                "page_size": [
                    612.0,
                    792.0
                ],
                "_layout_tree": [],
                "images": [],
                "tables": [],
                "interline_equations": [],
                "discarded_blocks": [],
                "para_blocks": [
                    {
                        "type": "text",
                        "bbox": [
                            52,
                            61.956024169921875,
                            294,
                            82.99800872802734
                        ],
                        "lines": [
                            {
                                "bbox": [
                                    52,
                                    61.956024169921875,
                                    294,
                                    72.0000228881836
                                ],
                                "spans": [
                                    {
                                        "bbox": [
                                            54.0,
                                            61.956024169921875,
                                            296.2261657714844,
                                            72.0000228881836
                                        ],
                                        "content": "dependent on the service headway and the reliability of the departure ",
                                        "type": "text",
                                        "score": 1.0
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
        ],
        "_parse_type": "txt",
        "_version_name": "0.6.1"
    }
```

&emsp;&emsp;解析结果是一个多层嵌套，从大到小依次为：<font color="red">PDF 文档 → 包含多个页面 → 包含多个区块(blocks) → 分为一级区块和二级区块 → 每个区块包含多行 → 每行包含多个最小单元， 其中`para_blocks`内存储的元素为区块信息， `span`是所有元素的最小存储单元。</font>大家理解了这个结构后，可以结合下图对每个字段进行理解，我们这里不再赘述。

<style>
.center 
{
  width: auto;
  display: table;
  margin-left: auto;
  margin-right: auto;
}
</style>

<p align="center"><font face="黑体" size=4>一级</font></p>
<div class="center">

| 字段名          | 解释                                                                                     |
|-----------------|------------------------------------------------------------------------------------------|
| pdf_info        | list，每个元素都是一个 dict，这个 dict 是每一页 PDF 的解析结果，详见下表                  |
| _parse_type     | ocr | txt，用来标识本次解析的中间态使用的模式                                           |
| _version_name   | string，表示本次解析使用的 magic-pdf 的版本号                                          |

</div>

<style>
.center 
{
  width: auto;
  display: table;
  margin-left: auto;
  margin-right: auto;
}
</style>

<p align="center"><font face="黑体" size=4>二级</font></p>
<div class="center">

| 字段名                  | 解释                                                                                     |
|-------------------------|------------------------------------------------------------------------------------------|
| preproc_blocks          | PDF 预处理后，未分段的中间结果                                                          |
| layout_bboxes           | 布局分割的结果，含有布局的方向（垂直、水平），和 bbox，按阅读顺序排序                  |
| page_idx                | 页码，从 0 开始                                                                          |
| page_size               | 页面宽度和高度                                                                          |
| _layout_tree            | 布局树状结构                                                                            |
| images                  | list，每个元素是一个 dict，每个 dict 表示一个 img_block                                  |
| tables                  | list，每个元素是一个 dict，每个 dict 表示一个 table_block                               |
| interline_equations     | list，每个元素是一个 dict，每个 dict 表示一个 interline_equation_block                  |
| discarded_blocks        | list，模型返回的需要 drop 的 block 信息                                                  |
| para_blocks             | 将 preproc_blocks 进行分段之后的结果                                                    |

</div>

<style>
.center 
{
  width: auto;
  display: table;
  margin-left: auto;
  margin-right: auto;
}
</style>

<p align="center"><font face="黑体" size=4>三级（最外层block）</font></p>
<div class="center">

| 字段名  | 解释                             |
|---------|----------------------------------|
| type    | block 类型（table 或 image）     |
| bbox    | block 矩形框坐标                |
| blocks  | list，里面的每个元素都是一个 dict 格式的二级 block |

</div>

<style>
.center 
{
  width: auto;
  display: table;
  margin-left: auto;
  margin-right: auto;
}
</style>

<p align="center"><font face="黑体" size=4>四级（内层block）</font></p>
<div class="center">

| 字段名 | 解释                             |
|--------|----------------------------------|
| type   | block 类型                       |
| bbox   | block 矩形框坐标                |
| lines  | list，每个元素都是一个 dict 表示的 line，用来描述一行信息的构成 |
|-----------------------|--------------------------|
| **其中 type 类型** | **描述**                       |
| image_body             | 图像的本体               |
| image_caption          | 图像的描述文本           |
| image_footnote         | 图像的脚注               |
| table_body             | 表格本体                 |
| table_caption          | 表格的描述文本           |
| table_footnote         | 表格的脚注               |
| text                   | 文本块                   |
| title                  | 标题块                   |
| index                  | 目录块                   |
| list                   | 列表块                   |
| interline_equation     | 行间公式块               |

</div>

<style>
.center 
{
  width: auto;
  display: table;
  margin-left: auto;
  margin-right: auto;
}
</style>

<p align="center"><font face="黑体" size=4>五级（lines）</font></p>
<div class="center">

| 字段名 | 解释                             |
|--------|----------------------------------|
| bbox   | line 的矩形框坐标                |
| spans  | list，每个元素都是一个 dict 表示的 span，用来描述一个最小组成单元的构成 |

</div>

<style>
.center 
{
  width: auto;
  display: table;
  margin-left: auto;
  margin-right: auto;
}
</style>

<p align="center"><font face="黑体" size=4>六级（spans）</font></p>
<div class="center">

| 字段名         | 解释                                                                                     |
|----------------|------------------------------------------------------------------------------------------|
| bbox           | span 的矩形框坐标                                                                        |
| type           | span 的类型                                                                               |
| content | img_path | 文本类型的 span 使用 content，图表类使用 img_path 用来存储实际的文本或者截图路径信息 |
|-----------------------|--------------------------|
| **其中 type 类型**         | **描述**  
| type                  | 描述                     |
| image                 | 图片                     |
| table                 | 表格                     |
| text                  | 文本                     |
| inline_equation       | 行内公式                 |
| interline_equation    | 行间公式                 |

</div>

&emsp;&emsp;该`.json`文件除了能像`model.json`一样提取特定类型的内容，因为其内容的丰富程度，可以做更精细化的文本分析，比如段落级别的文本分析，建立更精确的搜索索引，支持按内容类型搜索等。

&emsp;&emsp;7. span.pdf：根据 `span` 类型的不同，采用不同颜色线框绘制页面上所有 `span`。该文件可以用于质检，可以快速排查出文本丢失、行间公式未识别等问题。

&emsp;&emsp;最后一个 `.content_list.json` 文件，是`PDF` 文档中所有内容的列表，使用 JSON 格式存储数据，每个元素都是一个字典，包含了不同类型的内容（文本、图像等）。

```json
    [
        {
            // 文本内容块
            "type": "text",
            "text": "...",
            "text_level": N,  // 可选字段
            "page_idx": N
        },
        {
            // 图像内容块
            "type": "image",
            "img_path": "...",
            "img_caption": [],
            "img_footnote": [],
            "page_idx": N
        },
        {
            // 表格内容块
            "type": "table",
            "img_path": "...",
            "table_caption": [],
            "table_footnote": [],
            "table_body": "...",
            "page_idx": N
        }
    ]
```

<style>
.center 
{
  width: auto;
  display: table;
  margin-left: auto;
  margin-right: auto;
}
</style>

<p align="center"><font face="黑体" size=4>字段说明</p>
<div class="center">

| 字段名 | 出现位置 | 数据类型 | 说明 |
|--------|----------|----------|------|
| type | 所有内容块 | 字符串 | 内容块的类型，可能的值有："text"、"image"、"table" |
| page_idx | 所有内容块 | 整数 | 内容所在的页码，从0开始计数 |
| text | 文本类型块 | 字符串 | 文本内容 |
| text_level | 部分文本类型块 | 整数 | 文本的层级，通常用于标题，1表示一级标题 |
| img_path | 图像和表格类型块 | 字符串 | 图像或表格截图的文件路径 |
| img_caption | 图像类型块 | 数组 | 图像的说明文字，通常为空数组 |
| img_footnote | 图像类型块 | 数组 | 图像的脚注，通常为空数组 |
| table_caption | 表格类型块 | 数组 | 表格的说明文字，通常为空数组 |
| table_footnote | 表格类型块 | 数组 | 表格的脚注，通常为空数组 |
| table_body | 表格类型块 | 字符串 | 表格的HTML内容，包含完整的表格结构 |
</div>

&emsp;&emsp;以上是对`MinerU`解析结果的说明，通过这些文件，我们基本上可以获取到`PDF`文档中所有内容的位置、类型、内容等信息，理解各个文件的结构，也是我们后续进行自定义处理的关键，所以这里大家务必明确文件及其内部结构的组成。除此以外，除了`PDF`格式，该流程也可以支持图像（.jpg及.png）、PDF、Word（.doc及.docx）、以及PowerPoint（.ppt及.pptx）在内的多种文档格式的解析，大家可以进行尝试。

# 四、PaddleOCR项目介绍及vLLM服务启动

&emsp;&emsp;本小节课程，我们将从零开始，详细介绍如何在本地环境中完整部署 PaddleOCR-VL 并通过 vLLM 服务启动 PaddleOCR API 服务。

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202510201609587.png" width=70%></div>

&emsp;&emsp;对于硬件要求，官方给出了如上说明。其中使用原生的PaddlePaddle方式，需要GPU算力≥8.5（RTX 3090/4090、A100 等），最稳定。使用vLLM方式，需要GPU算力≥8（RTX 3060 及以上），速度最快，但算力7-8之间（T4/V100）虽可运行但不推荐，易超时或OOM。使用SGLang方式，需要GPU算力8-12之间（RTX 3060-4090），性能与稳定性的平衡选择。

&emsp;&emsp;注意：表格里的 "≥8" 指的是 GPU Compute Capability（GPU 算力版本号），不是显存大小！根据实测，根据PaddleOCR-VL-0.9B 模型（模型文件约 3.8GB）：

- 最低要求：6GB 显存（勉强够用，单张图）
- 推荐配置：8GB+ 显存（运行舒适）
- 理想配置：12GB+ 显存（可以批处理多张图）

&emsp;&emsp;所以：8GB 显存 + RTX 30 系列以上达到高效运行是完全没问题的。整个过程主要包括以下几个核心步骤：

1. **创建 Python 虚拟环境** - 隔离项目依赖，避免环境冲突；
2. **安装 PaddlePaddle 深度学习框架** - PaddleOCR 的底层依赖；
3. **安装 PaddleOCR 库** - 核心 OCR 功能库；
4. **下载预训练模型** - PaddleOCR-VL-0.9B 和 PP-DocLayoutV2；
5. **验证安装** - 运行测试确保环境正常；

&emsp;&emsp;公开课所使用的硬件环境为：Ubuntu 22.04 + 4 * 3090，共计显存96G显存，运行起来非常流畅。

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202510201346365.png" width=70%></div>

## 4.1 创建虚拟环境

&emsp;&emsp;首先，我们需要创建一个独立的 Python 虚拟环境。虚拟环境可以隔离项目依赖，避免与系统其他 Python 项目产生冲突。执行如下命令
```bash
    conda create -n ppocr-vllm python=3.11 -y
```

&emsp;&emsp;其中：
- `--name ppocr-vllm`：虚拟环境名称，可以自定义
- `python==3.11`：指定 Python 版本为 3.11（PaddleOCR 推荐版本）

&emsp;&emsp;执行效果如下图所示：

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511071213730.png" width=70%></div>

&emsp;&emsp;接下来激活虚拟环境：
```bash
    conda activate ppocr-vllm
```

&emsp;&emsp;激活后，命令行提示符前会显示 `(ppocr-vllm)`，表示已进入虚拟环境。

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511071213732.png" width=70%></div>

## 4.2 安装 PaddleOCR 工具框架

&emsp;&emsp;首先需要区分PaddleOCR 和 PaddleOCR-VL 两个之间的区别和联系。首先。PaddleOCR 是一个较为成熟、功能全面的 OCR＋文档理解开源工具库，其开源地址：https://github.com/PaddlePaddle/PaddleOCR/tree/main

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202510201635319.png" width=70%></div>

&emsp;&emsp;PaddleOCR-VL 是在这个生态内新发布的一个专注于“文档解析／视觉-语言模型 (Vision-Language Model, VLM)”功能的新模块。

<center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202510201635320.png" style="zoom:70%;" />

&emsp;&emsp;简单来说，PaddleOCR 是一个“全面且成熟”的 OCR＋文档理解工具箱，而 PaddleOCR-VL 是在这个工具箱里“专门为复杂文档解析”设计的新模块，采用了视觉-语言模型架构以应对更高阶的需求。

&emsp;&emsp;因此，在使用PaddleOCR-VL 时，我们首先需要安装 PaddleOCR 工具框架。https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202510201346364.png" width=70%></div>

&emsp;&emsp;这里安装 PaddlePaddle 3.2.0 版本。执行如下命令：

```bash
    python -m pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
```

&emsp;&emsp;其中:
- `paddlepaddle-gpu==3.2.0`：GPU 版本的 PaddlePaddle 3.2.0
- `-i https://...`：使用百度官方镜像源，下载速度更快
- `cu126`：对应 CUDA 12.6 版本

&emsp;&emsp;安装过程如下图所示：

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511071213733.png" width=70%></div>

&emsp;&emsp;安装结束后，需要验证 PaddlePaddle 安装状态，依次运行以下命令验证：

```bash
    python
    import paddle
    paddle.utils.run_check()
```

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202510201346367.png" width=70%></div>

&emsp;&emsp;如果看到如下“”信息，说明 PaddlePaddle 已成功安装。

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202510201346368.png" width=70%></div>

&emsp;&emsp;接下来要重点关注的是：PaddleOCR-VL 使用 `safetensors` 格式存储模型权重，需要额外安装，同时需要安装指定版本的，执行如下命令
```bash
    python -m pip install https://paddle-whl.bj.bcebos.com/nightly/cu126/safetensors/safetensors-0.6.2.dev0-cp38-abi3-linux_x86_64.whl
```

&emsp;&emsp;这是兼容最新PaddleOCR-Vl 的 safetensors 版本，安装很快。

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202510201346369.png" width=70%></div>

## 4.3 下载PaddleOCR-VL模型

&emsp;&emsp;使用 PaddleOCR-VL 解析功能需要两个预训练模型，其中：

1. **PaddleOCR-VL-0.9B** - 视觉语言模型（用于文本识别）
2. **PP-DocLayoutV2** - 文档布局检测模型（用于布局分析）

&emsp;&emsp;该模型权重的HuggingFace 地址为：https://huggingface.co/PaddlePaddle/PaddleOCR-VL ，

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202510201659478.png" width=70%></div>


&emsp;&emsp;此外，对于我们国内用户来说，更建议通过 ModelScope 下载（推荐）。https://modelscope.cn/models/PaddlePaddle/PaddleOCR-VL/summary 

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202510201706975.png" width=70%></div>

&emsp;&emsp;ModelScope 是阿里云的模型托管平台，国内访问速度快。首先执行如下命令：

```bash
    pip install modelscope
```

&emsp;&emsp;新建一个 download_paddleocr_vl.py 文件，写入如下代码：

```python
    from modelscope import snapshot_download

    # 下载完整模型（包含 PaddleOCR-VL-0.9B 和 PP-DocLayoutV2）
    model_dir = snapshot_download('PaddlePaddle/PaddleOCR-VL', local_dir='.')
```

&emsp;&emsp;其中：
- `'PaddlePaddle/PaddleOCR-VL'`：模型仓库 ID
- `local_dir='.'`：下载到当前目录（也可以指定其他路径，如 `'./models'`）

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202510211354046.png" width=80%></div>

<center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511071522837.png" style="zoom:80%;" />     

&emsp;&emsp;接下来执行如下代码进行模型权重安装：

```
   python download_paddleocr_vl.py
```

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202510201346361.png" width=70%></div>

&emsp;&emsp;下载完成后的模型目录结构如下所示：


<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202510201346363.png" width=70%></div>

&emsp;&emsp;其中 PaddleOCR-VL-0.9B 文件夹中存储的就是本次最新开源的超紧凑视觉语言模型，具有以下特点：

<style>
.center 
{
  width: auto;
  display: table;
  margin-left: auto;
  margin-right: auto;
}
</style>

<p align="center"><font face="黑体" size=4>PaddleOCR-VL-0.9B 模型特性</font></p>
<div class="center">

| 特性 | 说明 |
|------|------|
| **模型规模** | 0.9B 参数（极小，推理速度快） |
| **语言支持** | 109 种语言（包括中文、英文、日语、韩语等） |
| **识别能力** | 文本、表格、公式、图表等复杂元素 |
| **输出格式** | Markdown、JSON、HTML |
| **资源消耗** | GPU 显存 4GB+，推理速度 1.22 页/秒（A100） |
| **优势** | 比 MinerU 快 15.8%，显存占用比 dots.ocr 少 40% |

&emsp;&emsp;接下来，在进行运行前需要依次安装依赖包，首先是paddleocr[all], 其中：

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511071221916.png" width=70%></div>


```json
    # 只希望使用基础文字识别功能（返回文字位置坐标和文本内容），包含 PP-OCR 系列
    python -m pip install paddleocr
    # 希望使用文档解析、文档理解、文档翻译、关键信息抽取等全部功能
    # python -m pip install "paddleocr[doc-parser]"
```

&emsp;&emsp;这里我们安装doc-parser，如下图所示：

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511071213734.png" width=70%></div>

&emsp;&emsp;然后，使用 PaddleOCR CLI 安装 vLLM 的推理加速服务依赖：

```bash
    paddleocr install_genai_server_deps vllm
```

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511071213735.png" width=70%></div>

&emsp;&emsp;安装完成后，继续安装 `flash-atten` 编译包：

```bash
    pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.8cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

&emsp;&emsp;接下来，就可以启动`vLLM 服务器`服务器了。如下代码所示：

```bash
    paddlex_genai_server \
    --model_name PaddleOCR-VL-0.9B \
    --backend vllm \
    --host 0.0.0.0 \
    --port 8118
```

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511071213743.png" width=70%></div>

&emsp;&emsp;vLLM 服务启动后，保持这个终端窗口运行，不要关闭。同时打开一个新的终端，使用paddlex连接启动的paddleocr-vl服务，首先进行初始化服务配置：

```bash
    paddlex --install serving
```

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511071213736.png" width=70%></div>

&emsp;&emsp;然后生成 .yaml 配置文件：

```bash
    paddlex --get_pipeline_config PaddleOCR-VL
```

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511071213737.png" width=70%></div>

&emsp;&emsp;找到 genai_config 配置项，修改为如下所示：

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511071213739.png" width=70%></div>

&emsp;&emsp;最后，启动 PaddleOCR API 服务：

```bash
    paddlex --serve --pipeline PaddleOCR-VL.yaml --port 10800 --host 192.168.110.131 --paddle_model_dir /home/MuyuWorkSpace/02_OcrRag
```

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511071213740.png" width=70%></div>

&emsp;&emsp;成功启动后，在浏览器中访问 http://192.168.110.131:10800/docs 查看API文档，如下图所示：

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511071213741.png" width=70%></div>

&emsp;&emsp;至此，PaddleOCR API 服务启动成功。

## 4.4 PaddleOCR vLLM API服务连接测试


```python
import base64
import json
import requests
import os

# === 1. 服务端地址 ===
SERVER_URL = "http://192.168.130.4:10800/layout-parsing"

# === 2. 待处理文件路径 ===
input_path = "./course.pdf"   # 也可以是 test.png
output_md = "result.md"

# === 3. 读取文件并转为 Base64 ===
with open(input_path, "rb") as f:
    file_base64 = base64.b64encode(f.read()).decode("utf-8")

# === 4. 构造 JSON 请求体 ===
payload = {
    "file": file_base64,
    "fileType": 0 if input_path.lower().endswith(".pdf") else 1,
    "prettifyMarkdown": True,
    "visualize": False,
}

headers = {"Content-Type": "application/json"}

# === 5. 发送请求 ===
resp = requests.post(SERVER_URL, headers=headers, data=json.dumps(payload))

# === 6. 解析响应 ===
if resp.status_code == 200:
    data = resp.json()
    if data.get("errorCode") == 0:
        # PDF 的结果在 layoutParsingResults 数组中
        results = data["result"]["layoutParsingResults"]
        md_text = ""
        for i, page in enumerate(results, 1):
            md_text += f"\n\n# Page {i}\n\n"
            md_text += page["markdown"]["text"]

        with open(output_md, "w", encoding="utf-8") as f:
            f.write(md_text)
        print(f"成功生成 Markdown：{output_md}")
    else:
        print(f"服务端错误：{data.get('errorMsg')}")
else:
    print(f"HTTP 错误：{resp.status_code}")
    print(resp.text)

```

    成功生成 Markdown：result.md


&emsp;&emsp;PaddleOCR Doc Parser 命令参数如下表所示：

<style>
.center 
{
  width: auto;
  display: table;
  margin-left: auto;
  margin-right: auto;
}
</style>

<p align="center"><font face="黑体" size=4>基础参数</font></p>
<div class="center">

| 参数名 | 说明 |
|--------|------|
| `-i INPUT, --input INPUT` | 输入路径或 URL（必需） |
| `--save_path SAVE_PATH` | 输出目录路径 |
</div>

<style>
.center 
{
  width: auto;
  display: table;
  margin-left: auto;
  margin-right: auto;
}
</style>

<p align="center"><font face="黑体" size=4>布局检测参数</font></p>
<div class="center">

| 参数名 | 说明 |
|--------|------|
| `--layout_detection_model_name` | 布局检测模型名称 |
| `--layout_detection_model_dir` | 布局检测模型目录路径 |
| `--layout_threshold` | 布局检测模型的分数阈值 |
| `--layout_nms` | 是否在布局检测中使用 NMS（非极大值抑制） |
| `--layout_unclip_ratio` | 布局检测的扩展系数 |
| `--layout_merge_bboxes_mode` | 重叠框过滤方法 |
</div>


<style>
.center 
{
  width: auto;
  display: table;
  margin-left: auto;
  margin-right: auto;
}
</style>

<p align="center"><font face="黑体" size=4>VL识别模型参数</font></p>
<div class="center">

| 参数名 | 说明 |
|--------|------|
| `--vl_rec_model_name` | VL 识别模型名称 |
| `--vl_rec_model_dir` | **VL 识别模型目录路径（指定本地 PaddleOCR-VL-0.9B 模型路径）** |
| `--vl_rec_backend` | VL 识别模块使用的后端（native, vllm-server, sglang-server） |
| `--vl_rec_server_url` | VL 识别模块使用的服务器 URL |
| `--vl_rec_max_concurrency` | VLM 请求的最大并发数 |
</div>


<style>
.center 
{
  width: auto;
  display: table;
  margin-left: auto;
  margin-right: auto;
}
</style>

<p align="center"><font face="黑体" size=4>文档处理模型</font></p>
<div class="center">

| 参数名 | 说明 |
|--------|------|
| `--doc_orientation_classify_model_name` | 文档图像方向分类模型名称 |
| `--doc_orientation_classify_model_dir` | 文档图像方向分类模型目录路径 |
| `--doc_unwarping_model_name` | 文本图像矫正模型名称 |
| `--doc_unwarping_model_dir` | 图像矫正模型目录路径 |
</div>


<style>
.center 
{
  width: auto;
  display: table;
  margin-left: auto;
  margin-right: auto;
}
</style>

<p align="center"><font face="黑体" size=4>功能开关参数</font></p>
<div class="center">

| 参数名 | 说明 |
|--------|------|
| `--use_doc_orientation_classify` | 是否使用文档图像方向分类 |
| `--use_doc_unwarping` | 是否使用文本图像矫正 |
| `--use_layout_detection` | 是否使用布局检测 |
| `--use_chart_recognition` | 是否使用图表识别 |
| `--format_block_content` | 是否将块内容格式化为 Markdown |
| `--use_queues` | 是否使用队列进行异步处理 |
</div>


<style>
.center 
{
  width: auto;
  display: table;
  margin-left: auto;
  margin-right: auto;
}
</style>

<p align="center"><font face="黑体" size=4>VLM生成参数</font></p>
<div class="center">

| 参数名 | 说明 |
|--------|------|
| `--prompt_label` | VLM 的提示标签 |
| `--repetition_penalty` | VLM 采样中使用的重复惩罚系数 |
| `--temperature` | VLM 采样中使用的温度参数 |
| `--top_p` | VLM 采样中使用的 top-p 参数 |
| `--min_pixels` | VLM 图像预处理的最小像素数 |
| `--max_pixels` | VLM 图像预处理的最大像素数 |
</div>


<style>
.center 
{
  width: auto;
  display: table;
  margin-left: auto;
  margin-right: auto;
}
</style>

<p align="center"><font face="黑体" size=4>硬件与性能参数</font></p>
<div class="center">

| 参数名 | 说明 |
|--------|------|
| `--device` | 用于推理的设备，例如 `cpu`、`gpu`、`npu`、`gpu:0`、`gpu:0,1`。如果指定多个设备，将并行执行推理。注意：并非所有情况都支持并行推理。默认情况下，如果可用将使用 GPU 0，否则使用 CPU |
| `--enable_hpi` | 启用高性能推理 |
| `--use_tensorrt` | 是否使用 Paddle Inference TensorRT 子图引擎。如果模型不支持 TensorRT 加速，即使设置此标志也不会使用加速 |
| `--precision` | 使用 Paddle Inference TensorRT 子图引擎时的 TensorRT 精度（fp32, fp16） |
| `--enable_mkldnn` | 为推理启用 MKL-DNN 加速。如果 MKL-DNN 不可用或模型不支持，即使设置此标志也不会使用加速 |
| `--mkldnn_cache_capacity` | MKL-DNN 缓存容量 |
| `--cpu_threads` | 在 CPU 上用于推理的线程数 |
| `--paddlex_config` | PaddleX 管道配置文件路径 |
</div>

&emsp;&emsp;当然，上述代码也是`LangChain1.0 + OCR 多模态PDF解析实战`项目中`PaddleOCR` 的核心代码逻辑：

<div align="center">
    <img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511071644203.png" alt="MinerU 配置参考" width="70%">
</div>

# 五、DeepSeek-OCR vLLM 服务接口启动

&emsp;&emsp; `DeepSeek-OCR` 作为刚刚开源的小型多模态视觉模型，经过我们团队大量的测试，其不仅识别精度高，同时对中文的支持也非常出色，同时也能够输出结构化的 `Markdown` 格式文档。

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202510211101053.png" width=70%></div>

## 5.1 环境配置要求

&emsp;&emsp;在开始部署之前，我们首先需要确认自己的服务器是否满足运行条件。

<style>
.center 
{
  width: auto;
  display: table;
  margin-left: auto;
  margin-right: auto;
}
</style>

<p align="center"><font face="黑体" size=4>硬件配置要求（建议）</font></p>
<div class="center">

| 硬件类型 | 最低要求 | 推荐配置 | 说明 |
|---------|---------|---------|------|
| GPU | NVIDIA RTX 3090 | RTX 4090 或更高 | 必须是NVIDIA显卡，支持CUDA |
| 显存 | 24GB | 24GB+ | 显存越大，处理速度越快 |
| 内存 | 32GB | 64GB+ | 用于图像预处理和数据缓存 |
| 硬盘 | 50GB可用空间 | 100GB+ | 模型文件较大，需要足够空间 |

</div>



&emsp;&emsp;除了硬件，我们还需要准备相应的软件环境：主要包括**操作系统**：建议使用 `Ubuntu 22.04`，同时需要 `CUDA 12.0+`，而**Python 环境**建议使用 `Python 3.10+`。


&emsp;&emsp;课程使用的配置环境为：`Ubuntu 22.04` + 四卡 3090。大家在使用 vLLM 启动`DeepSeek-OCR`时，首先需要做如下配置校验：

```bash
    # 检查 GPU 是否正常工作
    nvidia-smi
```

&emsp;&emsp;执行上述命令后，如果看到类似下面的输出，说明GPU正常工作：

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202510301228311.png" width=70%></div>

&emsp;&emsp;接下来检查 Conda 是否安装，执行如下命令：

```bash
    conda --version
```

&emsp;&emsp;如果看到类似 `conda XXX` 的版本号，说明conda已经安装。如果提示命令不存在，需要先安装 miniconda 或者 anaconda3。

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202510301228312.png" width==70%></div>

## 5.2 DeepSeek-OCR模型下载

&emsp;&emsp;环境准备好之后，接下来我们需要下载 `DeepSeek-OCR` 模型文件，并进行相应的配置。这一步是整个部署过程的核心。国内用户强烈建议大家使用 `ModelScope` 下载模型，没有网络限制，同时速度也会快很多。

&emsp;&emsp;首先，我们需要创建一个独立的 Python 虚拟环境。虚拟环境可以隔离项目依赖，避免与系统其他 Python 项目产生冲突。执行如下命令
```bash
    conda create --name deepseek-ocr-vllm python==3.10
```

&emsp;&emsp;其中：
- `--name deepseek-ocr-vllm`：虚拟环境名称，可以自定义
- `python==3.10`：指定 Python 版本为 3.10（PaddleOCR 推荐版本）

&emsp;&emsp;执行效果如下图所示：

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511061607713.png" width=70%></div>

&emsp;&emsp;接下来激活虚拟环境：
```bash
    conda activate deepseek-ocr-vllm
```

&emsp;&emsp;激活后，命令行提示符前会显示 `(deepseek-ocr-vllm)`，表示已进入虚拟环境。

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202510301257910.png" width=70%></div>

&emsp;&emsp;接下来，在`ModelScope` 平台下载 `DeepSeek-OCR` 模型文件。地址：https://modelscope.cn/models/deepseek-ai/DeepSeek-OCR

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202510301300250.png" width=70%></div>

&emsp;&emsp;ModelScope 是阿里云的模型托管平台，国内访问速度快。首先执行如下命令：

```bash
    pip install modelscope
```

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202510301301452.png" width=70%></div>

&emsp;&emsp;然后执行如下代码：

```python
    from modelscope import snapshot_download
    model_dir = snapshot_download('deepseek-ai/DeepSeek-OCR',local_dir='.')
```

&emsp;&emsp;其中：
- `'deepseek-ai/DeepSeek-OCR'`：模型仓库 ID
- `local_dir='.'`：下载到当前目录（也可以指定其他路径，如 `'./models'`）

&emsp;&emsp;接下来执行如下代码进行模型权重安装：

```bash
   python download_deepseek_ocr.py
```

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202510301304455.png" width=70%></div>

&emsp;&emsp;下载完成后的模型目录结构如下所示：


<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202510301323950.png" width=70%></div>

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511071653536.png" width=70%></div>

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511071522837.png" width=30%></div>

## 5.3 Deepseek-OCR vLLM 项目文件

&emsp;&emsp;`DeepSeek-OCR`通过 vLLM 平台启动，我们是借助`DeepSeek-OCR`官方提供的项目代码，并做了一些优化和调整，其官方源码下载地址：https://github.com/deepseek-ai/DeepSeek-OCR/tree/main/DeepSeek-OCR-master/DeepSeek-OCR-vllm

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202510301228310.png" width=70%></div>

&emsp;&emsp;我们将项目文件下载到本地并上传到服务器上：

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511071049011.png" width=70%></div>

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511071654511.png" width=70%></div>

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511071522837.png" width=30%></div>

&emsp;&emsp;其中`deepseek_ocr.py` 和 `flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl` 使我们给大家提供的两个核心文件，一个用于封装 DeepSeek OCR API 接口，另一个则是离线安装的 fla_ttn 的离线安装包。

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511061607714.png" width=70%></div>

&emsp;&emsp;接下来一键安装 vLLM 运行 DeepSeek-OCR的依赖包：

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511061607715.png" width=70%></div>

&emsp;&emsp;安装结束后，使用我们给大家提供的`deepseek_ocr.py`文件，启动 `DeepSeek OCR API 接口` 服务。

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511071111948.png" width=70%></div>

&emsp;&emsp;启动命令如下：

```bash
    python ocr_client.py --model-path /home/data/nongwa/workspace/model/OCR/DeepSeek-OCR --gpu-id 3 --port 8797 --host 192.168.110.131
```

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511061710748.png" width=70%></div>

&emsp;&emsp;启动成功后，会显示如下：

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511061710749.png" width=70%></div>

&emsp;&emsp;接下来，便可以在网页端查看到 `FastAPI` 接口：

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511061710750.png" width=70%></div>

## 5.4 DeepSeek-OCR API 连接测试

&emsp;&emsp;前面我们完成了服务的部署和启动，接下来我们要学习如何在实际项目中调用这个 OCR 服务。本节将通过完整的代码示例，带领大家掌握图片识别和 PDF 文档处理的全流程。

&emsp;&emsp;在开始编写调用代码之前，我们先来了解一下 OCR 服务提供的 API 接口规范。

<style>
.center 
{
  width: auto;
  display: table;
  margin-left: auto;
  margin-right: auto;
}
</style>

<p align="center"><font face="黑体" size=4>OCR API 接口规范</font></p>
<div class="center">

| 项目 | 说明 |
|------|------|
| 接口地址 | `http://localhost:8797/ocr` |
| 请求方式 | POST |
| 内容类型 | multipart/form-data |
| 文件参数 | `file`（支持 jpg, png, pdf 等格式） |
| 可选参数 | `enable_description`（是否生成图片描述，默认 False） |

</div>

&emsp;&emsp;返回的数据格式如下：

```json
    {
        "markdown": "识别出的文本内容（Markdown格式）",
        "page_count": 1,
        "processing_time": 2.34
    }
```

&emsp;&emsp;<font color=red>返回的 `markdown` 字段就是我们需要的识别结果</font>，它保留了原文档的格式结构，包括标题、列表、表格等。


&emsp;&emsp;下面我们通过一个完整的代码示例，来演示如何调用 OCR 服务。


```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 DeepSeek-OCR API
直接调用 API 并保存返回的 markdown 和图像数据
"""

import requests
import json
import sys
from pathlib import Path


def test_deepseek_ocr(pdf_path: str, output_dir: str = "./test_output"):
    """
    测试 DeepSeek-OCR API

    Args:
        pdf_path: PDF 文件路径
        output_dir: 输出目录
    """
    # API 配置
    api_url = "http://192.168.130.4:8797/ocr"

    # 确保输出目录存在
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"测试文件: {pdf_path}")
    print(f"API 地址: {api_url}")
    print(f"输出目录: {output_dir}")
    print("-" * 60)

    # 读取 PDF 文件
    try:
        with open(pdf_path, 'rb') as f:
            files = {'file': (Path(pdf_path).name, f, 'application/pdf')}

            # API 参数
            data = {
                'enable_description': 'false',  # 是否生成图片描述
            }

            print("发送请求到 DeepSeek API...")

            # 发送请求
            response = requests.post(
                api_url,
                files=files,
                data=data,
                timeout=300
            )

            if response.status_code != 200:
                print(f"API 返回错误: {response.status_code}")
                print(f"错误信息: {response.text[:500]}")
                return False

            # 解析响应
            result = response.json()

            print(f"API 响应成功")
            print(f"响应包含的字段: {list(result.keys())}")
            print("-" * 60)

            # 提取数据
            markdown_content = result.get("markdown", "")
            page_count = result.get("page_count", 0)
            images_data = result.get("images", {})

            print(f"Markdown 长度: {len(markdown_content)} 字符")
            print(f"页数: {page_count}")
            print(f"图像数量: {len(images_data)}")

            if images_data:
                print(f"图像列表:")
                for img_key in list(images_data.keys())[:10]:
                    img_size = len(images_data[img_key])
                    print(f"   - {img_key}: {img_size} 字符 (base64)")

            print("-" * 60)

            # 保存 Markdown
            md_file = output_path / f"{Path(pdf_path).stem}_deepseek.md"
            with open(md_file, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            print(f"Markdown 已保存: {md_file}")

            # 保存完整响应
            json_file = output_path / f"{Path(pdf_path).stem}_response.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                # 为了避免文件过大，只保存图像的部分信息
                simplified_result = {
                    "markdown": markdown_content,
                    "page_count": page_count,
                    "images_count": len(images_data),
                    "image_keys": list(images_data.keys())
                }
                json.dump(simplified_result, f, ensure_ascii=False, indent=2)
            print(f"响应摘要已保存: {json_file}")

            # 保存图像数据（可选，如果需要）
            if images_data:
                images_file = output_path / f"{Path(pdf_path).stem}_images.json"
                with open(images_file, 'w', encoding='utf-8') as f:
                    json.dump(images_data, f, ensure_ascii=False, indent=2)
                print(f"图像数据已保存: {images_file}")

            # 统计信息
            print("-" * 60)
            print("统计信息:")

            # 统计表格数量
            import re
            table_count = len(re.findall(r'<table>', markdown_content, re.IGNORECASE))
            print(f"   - HTML 表格: {table_count} 个")

            # 统计图片引用
            img_ref_count = len(re.findall(r'!\[.*?\]\(.*?\)', markdown_content))
            print(f"   - Markdown 图片引用: {img_ref_count} 个")

            # 统计行数
            line_count = len(markdown_content.split('\n'))
            print(f"   - Markdown 行数: {line_count} 行")

            print("-" * 60)
            print("测试完成！")

            return True

    except FileNotFoundError:
        print(f"文件不存在: {pdf_path}")
        return False
    except requests.exceptions.Timeout:
        print(f"请求超时")
        return False
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""


    success = test_deepseek_ocr(
        pdf_path='./course.pdf', 
        output_dir='./')


if __name__ == "__main__":
    main()

```

    测试文件: ./course.pdf
    API 地址: http://192.168.110.131:8797/ocr
    输出目录: ./
    ------------------------------------------------------------
    发送请求到 DeepSeek API...
    API 响应成功
    响应包含的字段: ['markdown', 'page_count', 'images']
    ------------------------------------------------------------
    Markdown 长度: 10026 字符
    页数: 12
    图像数量: 14
    图像列表:
       - page_0_img_0.png: 161960 字符 (base64)
       - page_1_img_0.png: 9892 字符 (base64)
       - page_1_img_1.png: 111764 字符 (base64)
       - page_1_img_2.png: 43132 字符 (base64)
       - page_2_img_0.png: 66016 字符 (base64)
       - page_3_img_0.png: 57016 字符 (base64)
       - page_4_img_0.png: 53420 字符 (base64)
       - page_5_img_0.png: 166376 字符 (base64)
       - page_6_img_0.png: 34012 字符 (base64)
       - page_8_img_0.png: 227632 字符 (base64)
    ------------------------------------------------------------
    Markdown 已保存: course_deepseek.md
    响应摘要已保存: course_response.json
    图像数据已保存: course_images.json
    ------------------------------------------------------------
    统计信息:
       - HTML 表格: 9 个
       - Markdown 图片引用: 14 个
       - Markdown 行数: 264 行
    ------------------------------------------------------------
    测试完成！


&emsp;&emsp;通过不同的提示词，对输入的PDF文档的每一页做格式化提取：

```
    # 固定 Prompt
    PROMPT_OCR = "<image>\n<|grounding|>Convert the document to markdown."
    PROMPT_DESC = "<image>\nDescribe this image in detail."
```

&emsp;&emsp;当然，上述代码也是`LangChain1.0 + OCR 多模态PDF解析实战`项目中`DeepSeek-OCR` 的核心代码逻辑：

<div align="center">
    <img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511071702151.png" alt="MinerU 配置参考" width="70%">
</div>

# 六、OCR 多模态解析系统本地部署

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511071934682.png" width=70%></div>

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511071522837.png" width=30%></div>

&emsp;&emsp;完整的系统需要同时启动后端和前端两个服务。首先对于后端服务启动，需要依次执行如下操作：

```bash
    # 1. 进入后端目录
    cd backend

    # 2. 创建并激活虚拟环境
    python -m venv venv
    source venv/bin/activate  # Linux/Mac
    # 或
    venv\Scripts\activate     # Windows

    # 3. 安装依赖
    pip install -r requirements.txt

    # 4. 配置环境变量 (.env 文件)
  
    # Server Configuration
    PORT=8000
    HOST=0.0.0.0
    DEBUG=True

    # MinerU Configuration
    MINERU_API_URL=http://192.168.110.131:50000/file_parse
    VLLM_SERVER_URL=http://192.168.110.131:40000
    MINERU_BACKEND=vlm-vllm-async-engine
    MINERU_TIMEOUT=600
    MINERU_VIZ_DIR=/home/MuyuWorkSpace/05_OcrProject/backend/mineru_visualizations

    # DeepSeek OCR Configuration
    DEEPSEEK_OCR_API_URL=http://192.168.110.131:8797/ocr

    # PaddleOCR Configuration
    PADDLEOCR_API_URL=http://192.168.110.131:10800/layout-parsing

    # File Upload Limits
    MAX_FILE_SIZE=10485760
    ALLOWED_FILE_TYPES=application/pdf,image/png,image/jpeg,image/jpg,image/webp

    # CORS Settings
    ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173


&emsp;&emsp;其中：

<style>
.center 
{
  width: auto;
  display: table;
  margin-left: auto;
  margin-right: auto;
}
</style>

<p align="center"><font face="黑体" size=4>配置说明</font></p>
<div class="center">

| 配置项 | 说明 |
|--------|------|
| `MINERU_BACKEND` | `pipeline`: 使用本地 PyTorch<br>`vlm-vllm-async-engine`: 使用 vLLM 加速 |
| `VLLM_SERVER_URL` | vLLM 服务器地址（仅在使用 vLLM 时需要） |
| `MINERU_API_URL` | MinerU API 服务地址 |

```

&emsp;&emsp;启动服务执行如下代码：

```bash
    python main.py
```

<div align="center">
    <img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511071838724.png" alt="MinerU 配置参考" width="70%">
</div>

&emsp;&emsp;然后，便可以在 `http://192.168.110.131:8000/docs` 查看 API 文档：

<div align="center">
    <img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511071838725.png" alt="MinerU 配置参考" width="70%">
</div>



&emsp;&emsp;前端服务启动，需要依次执行如下操作:

```bash
    # 1. 进入前端目录
    cd frontend

    # 2. 安装依赖
    npm install
    # 或
    yarn install

    # 3. 启动开发服务器
    npm run dev
    # 或
    yarn dev
```

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511071838726.png" width=70%></div>

&emsp;&emsp;启动成功后，即可通过`http://localhost:3000`访问应用。

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511071838727.png" width=70%></div>


# 【加餐】 LangChain1.0 接入 MinerU MCP Server

&emsp;&emsp;随着`MinerU`项目的发展，除了解析性能上进一步提升，同时也支持了`MCP`协议，可以通过`stdio`、`Http + Sse`和 `Streamable Http`三种通信模型进行访问。因此，本小节我们重点讲解`MinerU`项目在`MCP`协议下的使用，并基于`LangChain 1.0` 新版本实现`MinerU MCP`的接入。

&emsp;&emsp;最新版本的`MinerU` 项目中内置了`MCP Server` 服务，我们可以通过该服务将`MinerU` 的解析服务暴露给`ADK` 或 `Agent` 使用。与常规的`MCP Server` 构建方式一样，也是通过 `FastMCP` 外放`FastAPI` 服务并支持`stdio`、`Http+Sse`和`StreamableHttp` 三种传输方式。源码位置如下：

<div align="center">
    <img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511061214157.png" alt="MinerU 配置参考" width="70%">
</div>
<div align="center">
    <img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511061214158.png" alt="MinerU 配置参考" width="70%">
</div>


&emsp;&emsp;这里我们使用`StreamableHttp` 传输方式，并且在启动前，修改下源码，使其可以加载我们通过命令行传递的`host`参数，否则只能使用默认的`localhost` 地址。如下所示：

<div align="center">
    <img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511061214159.png" alt="MinerU 配置参考" width="70%">
</div>

&emsp;&emsp;接下来创建`MCP`的`uv`环境，首先通过`pip` 安装 `uv`:

```bash
    pip install uv
```

<div align="center">
    <img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511061214160.png" alt="MinerU 配置参考" width="70%">
</div>

&emsp;&emsp;然后通过`uv` 创建`MCP` 的`uv` 环境，并安装`MinerU` 项目中用到的依赖：

```bash
    uv venv
    source .venv/bin/activate
    uv pip install -e .
```

<div align="center">
    <img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511061214161.png" alt="MinerU 配置参考" width="70%">
</div>


&emsp;&emsp;安装完成后，我们就可以通过`uv` 启动`MCP` 服务了：

```bash
    uv run mineru-mcp --transport streamable-http --host 192.168.110.131
```

<div align="center">
    <img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511061214164.png" alt="MinerU 配置参考" width="70%">
</div>

&emsp;&emsp;启动后，我们可以在浏览器中访问`http://192.168.110.131:8000/mcp`快速进行验证：

<div align="center">
    <img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511061231733.png" alt="MinerU 配置参考" width="70%">
</div>

&emsp;&emsp;需要重点注意的是：`parse_documents` 工具是`MinerU` 项目中通过`FastAPI` 外放的工具接口，其本质上，调用的是`mineru-api` 服务中的`/api/parse` 接口。所以，大家要理解的逻辑是：
`FastMCP` 负责暴露`parse_documents`和`get_ocr_languages` 工具协议，而真正做 `PDF→Markdown` 的逻辑在 `MinerU` 的“本地解析服务”里，并通过 `/file_parse` 路由完成。MCP 只是一个工具“壳”，真正干活的是`mineru-api`。因此，我们首先需要修改`MinerU MCP Server`的配置文件，让其找到解析服务。如下所示：

```bash
    mv env.example .env
    vi .env
```
<div align="center">
    <img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511061214162.png" alt="MinerU 配置参考" width="70%">
</div>

&emsp;&emsp;修改如下：
<div align="center">
    <img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511061214163.png" alt="MinerU 配置参考" width="70%">
</div>

&emsp;&emsp;其中`USE_LOCAL_API=true` 表示使用本地解析服务，`LOCAL_API_URL` 表示本地解析服务的地址。修改完成后重新启动`MinerU MCP Server` 服务。

```bash
    uv run mineru-mcp --transport streamable-http --host 192.168.110.131
```

<div align="center">
    <img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511061214164.png" alt="MinerU 配置参考" width="70%">
</div>
&emsp;&emsp;接下来新打开一个终端，启动`mineru-api` 服务：

```bash
    export MINERU_MODEL_SOURCE=local  # 注意：这里需要将模型源设置为本地
    mineru-api --host 192.168.110.131 --port 30000
```
<div align="center">
    <img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511061214165.png" alt="MinerU 配置参考" width="70%">
</div>


&emsp;&emsp;启动后，在`192.168.110.131:30000/docs`中可以看到接口服务：

<div align="center">
    <img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511061234642.png" alt="MinerU 配置参考" width="70%">
</div>

&emsp;&emsp;全部验证成功后，我们就可以在`LangChain`中使用`MCP` 服务了。代码如下：


```python
! pip install langchain>=1.0.0 langchain-core>=1.0.0 langchain-mcp-adapters mcp httpx -q
```

    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.[0m[33m
    [0m

&emsp;&emsp;安装完成后，让我们导入必要的模块：



```python
# 导入必要的库
import asyncio
from pathlib import Path
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from langchain_mcp_adapters.tools import load_mcp_tools

print("所有库导入成功！")

```

    所有库导入成功！


&emsp;&emsp;现在，配置连接到 `MinerU MCP` 服务的参数。根据服务启动日志，
- **协议类型**: streamable-http
- **服务地址**: http://192.168.110.131:8001/mcp
&emsp;&emsp;首先定义服务器配置参数：


```python
# 配置 MinerU MCP 服务器参数
MCP_SERVER_URL = "http://192.168.110.131:8001/mcp"

server_params = {
    "url": MCP_SERVER_URL,
    # 如果需要认证，可以在这里添加 headers
    # "headers": {
    #     "Authorization": "Bearer your_token"
    # }
}

print(f"MCP 服务器地址: {MCP_SERVER_URL}")
print("服务器参数配置完成")

```

    MCP 服务器地址: http://192.168.110.131:8001/mcp
    服务器参数配置完成



&emsp;&emsp;在实际解析 PDF 之前，我们先要了解 MCP 服务器提供了哪些工具。这就像进入一个工具箱，先看看里面有什么工具可用。如下函数来<font color=red>列出所有可用的 MCP 工具</font>：



```python
async def list_mcp_tools(server_params):
    """
    列出 MCP 服务器提供的所有工具
    
    Args:
        server_params: 服务器连接参数
        
    Returns:
        tools: LangChain 工具列表
    """
    print("🔌 正在连接 MCP 服务器...")
    
    # 使用 streamable-http 客户端建立连接
    async with streamablehttp_client(**server_params) as (read, write, _):
        # 创建 MCP 会话
        async with ClientSession(read, write) as session:
            # 初始化会话（握手）
            await session.initialize()
            print("MCP 会话初始化成功\n")
            
            # 加载所有 MCP 工具
            tools = await load_mcp_tools(session)
            
            print(f"发现 {len(tools)} 个可用工具：\n")
            print("=" * 80)
            
            # 打印每个工具的详细信息
            for i, tool in enumerate(tools, 1):
                print(f"\n工具 {i}: {tool.name}")
                print(f"   描述: {tool.description}")
                
                # 打印工具参数（如果有）
                if hasattr(tool, 'args_schema') and tool.args_schema:
                    print(f"   参数结构:")
                    if hasattr(tool.args_schema, 'schema'):
                        schema = tool.args_schema.schema()
                        if 'properties' in schema:
                            for param_name, param_info in schema['properties'].items():
                                param_type = param_info.get('type', 'unknown')
                                param_desc = param_info.get('description', '无描述')
                                print(f"      - {param_name} ({param_type}): {param_desc}")
                
            print("\n" + "=" * 80)
            
            return tools

# 执行函数（在 Jupyter 中运行异步代码）
tools = await list_mcp_tools(server_params)
```

    🔌 正在连接 MCP 服务器...
    MCP 会话初始化成功
    
    发现 2 个可用工具：
    
    ================================================================================
    
    工具 1: parse_documents
       描述: 统一接口，将文件转换为Markdown格式。支持本地文件和URL，会根据USE_LOCAL_API配置自动选择合适的处理方式。
    
    当USE_LOCAL_API=true时:
    - 会过滤掉http/https开头的URL路径
    - 对本地文件使用本地API进行解析
    
    当USE_LOCAL_API=false时:
    - 将http/https开头的路径使用convert_file_url处理
    - 将其他路径使用convert_file_path处理
    
    处理完成后，会自动尝试读取转换后的文件内容并返回。
    
    返回:
        成功: {"status": "success", "content": "文件内容"} 或 {"status": "success", "results": [处理结果列表]}
        失败: {"status": "error", "error": "错误信息"}
       参数结构:
    
    工具 2: get_ocr_languages
       描述: 获取 OCR 支持的语言列表。
    
    Returns:
        Dict[str, Any]: 包含所有支持的OCR语言列表的字典
       参数结构:
    
    ================================================================================


&emsp;&emsp;接下来，构建一个<font color=red>完整的客户端类</font>，封装所有的功能，使其更易用。客户端类将包含以下功能：

1. 连接管理
2. 工具列表查询
3. PDF 文档解析
4. 结果保存


```python
class MinerUClient:
    """
    MinerU MCP 客户端封装类
    提供简洁的 API 来调用 MinerU 文档解析服务
    """
    
    def __init__(self, mcp_url: str = "http://192.168.110.131:8001/mcp"):
        """
        初始化客户端
        
        Args:
            mcp_url: MCP 服务器地址
        """
        self.mcp_url = mcp_url
        self.server_params = {"url": self.mcp_url}
        print(f"MinerU 客户端已初始化")
        print(f"服务器地址: {self.mcp_url}")
    
    async def get_tools(self):
        """
        获取所有可用的 MCP 工具
        
        Returns:
            tools: 工具列表
        """
        async with streamablehttp_client(**self.server_params) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await load_mcp_tools(session)
                return tools
    
    async def parse_pdf(self, pdf_path: str, output_format: str = "markdown"):
        """
        解析 PDF 文件
        
        Args:
            pdf_path: PDF 文件路径（支持相对路径和绝对路径）
            output_format: 输出格式，默认为 markdown
            
        Returns:
            解析结果（Markdown 文本）
        """
        # 检查文件是否存在
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")
        
        print(f"开始解析 PDF: {pdf_file.name}")
        print(f"文件路径: {pdf_file.absolute()}")
        print(f"文件大小: {pdf_file.stat().st_size / 1024:.2f} KB\n")
        
        # 连接 MCP 服务器并解析
        async with streamablehttp_client(**self.server_params) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # 加载工具
                tools = await load_mcp_tools(session)
                print(f"已加载 {len(tools)} 个工具\n")
                
                # 查找合适的解析工具
                parse_tool = None
                for tool in tools:
                    # 根据工具名称特征查找
                    tool_name_lower = tool.name.lower()
                    if any(keyword in tool_name_lower for keyword in 
                           ['convert', 'parse', 'markdown', 'pdf', 'file']):
                        parse_tool = tool
                        print(f"找到解析工具: {tool.name}")
                        print(f"   工具描述: {tool.description}\n")
                        break
                
                if not parse_tool:
                    print("未找到合适的解析工具，可用工具列表：")
                    for i, tool in enumerate(tools, 1):
                        print(f"   {i}. {tool.name}")
                    # 如果没找到特定工具，尝试使用第一个工具
                    if tools:
                        parse_tool = tools[0]
                        print(f"\n尝试使用第一个工具: {parse_tool.name}")
                    else:
                        raise ValueError("MCP 服务器没有提供任何工具")
                
                # 调用工具进行解析
                print("正在调用 MCP 工具解析文档...")
                
                try:
                    # 构建工具调用参数 - 参数是字符串格式（支持单个文件或逗号分隔的多个文件）
                    tool_input = {
                        "file_sources": str(pdf_file.absolute())
                    }
                    
                    # 调用工具
                    result = await parse_tool.ainvoke(tool_input)
                    
                    print("解析完成！\n")
                    return result
                    
                except Exception as e:
                    print(f"解析失败: {str(e)}")
                    raise

# 创建客户端实例
client = MinerUClient(mcp_url=MCP_SERVER_URL)
print("客户端创建成功！")
```

    MinerU 客户端已初始化
    服务器地址: http://192.168.110.131:8001/mcp
    客户端创建成功！


&emsp;&emsp;在实际运行之前，我们需要准备一个 PDF 文件用于测试。


```python
# 使用 LangChain Agent 自动调用 MCP 工具

from pathlib import Path

print("=" * 80)
print("创建 LangChain Agent 自动调用 MCP 工具")
print("=" * 80)

# 1. 加载 MCP 工具
print("\n加载 MCP 工具...")
async with streamablehttp_client(**server_params) as (read, write, _):
    async with ClientSession(read, write) as session:
        await session.initialize()
        mcp_tools = await load_mcp_tools(session)
        
        print(f"已加载 {len(mcp_tools)} 个 MCP 工具")
        for tool in mcp_tools:
            print(f"   - {tool.name}: {tool.description[:60]}...")
        
        # 2. 直接使用工具解析 PDF
        print("\n" + "=" * 80)
        print("使用 parse_documents 工具解析 PDF")
        print("=" * 80)
        
        pdf_file_path = "./course.pdf"
        pdf_file = Path(pdf_file_path)
        
        if not pdf_file.exists():
            print(f"文件不存在: {pdf_file_path}")
        else:
            print(f"\n文件: {pdf_file.name}")
            print(f"大小: {pdf_file.stat().st_size / 1024:.2f} KB")
            
            # 找到 parse_documents 工具
            parse_tool = None
            for tool in mcp_tools:
                if tool.name == "parse_documents":
                    parse_tool = tool
                    break
            
            if not parse_tool:
                print("未找到 parse_documents 工具")
            else:
                print("\n调用 parse_documents 工具...")
                
                try:
                    # 调用工具，可以指定 backend 和 enable_ocr
                    result = await parse_tool.ainvoke({
                        "file_sources": str(pdf_file.absolute()),
                        "backend": "vlm-vllm-async-engine",  # 可选: pipeline, vlm-transformers, vlm-vllm-engine, vlm-http-client
                        "enable_ocr": True,  # 启用 OCR，对应 parse_method="ocr"
                        "language": "ch",  # 文档语言
                        "device": "cuda:3"
                    })
                    
                    print("解析完成！\n")
                    
                    # 处理结果 - 正确解析 JSON 结构
                    import json
                    import urllib.parse
                    
                    if isinstance(result, str):
                        # 尝试解析 JSON 字符串
                        try:
                            result = json.loads(result)
                        except:
                            content = result
                            result = None
                    
                    if isinstance(result, dict):
                        if result.get("status") == "success":
                            # 检查是否有 content 字段（直接返回）
                            content = result.get("content", "")
                            
                            # 如果没有 content，检查 result 字段（嵌套结构）
                            if not content:
                                result_data = result.get("result", {})
                                if result_data:
                                    results_dict = result_data.get("results", {})
                                    if results_dict:
                                        # 提取第一个结果（即使键是 URL 编码的）
                                        # results_dict 的键可能是 URL 编码的中文文件名
                                        for encoded_key, file_result in results_dict.items():
                                            md_content = file_result.get("md_content", "")
                                            if md_content:
                                                content = md_content
                                                # 解码文件名（用于日志）
                                                try:
                                                    decoded_name = urllib.parse.unquote(encoded_key)
                                                    print(f"   解码文件名: {decoded_name}")
                                                except:
                                                    pass
                                                break
                            
                            # 如果还是没有，检查是否有 results 列表
                            if not content:
                                results_list = result.get("results", [])
                                if results_list:
                                    content = results_list[0].get("content", "")
                        elif result.get("status") == "error":
                            print(f"解析失败: {result.get('error')}")
                            content = None
                        else:
                            content = str(result)
                    elif result is None:
                        # 已经是字符串了
                        pass
                    else:
                        content = str(result)
                    
                    if content:
                        print("=" * 80)
                        print("解析结果统计")
                        print("=" * 80)
                        print(f"总字符数: {len(content):,}")
                        print(f"总行数: {content.count(chr(10)) + 1:,}")
                        
                        # 保存结果到当前工作目录
                        import os
                        current_dir = Path(os.getcwd())
                        output_file = current_dir / f"{pdf_file.stem}_parsed.md"
                        output_file.write_text(content, encoding='utf-8')
                        
                        print(f"\n结果已保存到: {output_file.absolute()}")
                        print(f"当前工作目录: {current_dir.absolute()}")
                        print(f"文件大小: {output_file.stat().st_size / 1024:.2f} KB")
                        
                        # 显示预览
                        print("\n" + "-" * 80)
                        print("解析结果预览（前 1500 字符）:")
                        print("-" * 80)
                        print(content[:1500])
                        if len(content) > 1500:
                            print("\n... (后续内容已省略)")
                        print("-" * 80)
                        
                        print("\n" + "=" * 80)
                        print("成功！")
                        print("=" * 80)
                    else:
                        print("解析结果为空")
                        
                except Exception as e:
                    print(f"调用失败: {e}")
                    import traceback
                    traceback.print_exc()

```

    ================================================================================
    创建 LangChain Agent 自动调用 MCP 工具
    ================================================================================
    
    加载 MCP 工具...
    已加载 2 个 MCP 工具
       - parse_documents: 统一接口，将文件转换为Markdown格式。支持本地文件和URL，会根据USE_LOCAL_API配置自动选择合适的处理...
       - get_ocr_languages: 获取 OCR 支持的语言列表。
    
    Returns:
        Dict[str, Any]: 包含所有支持的OCR语言列表...
    
    ================================================================================
    使用 parse_documents 工具解析 PDF
    ================================================================================
    
    文件: course.pdf
    大小: 1391.70 KB
    
    调用 parse_documents 工具...
    解析完成！
    
       解码文件名: course
    ================================================================================
    解析结果统计
    ================================================================================
    总字符数: 12,693
    总行数: 364
    
    结果已保存到: /home/MuyuWorkSpace/course_parsed.md
    当前工作目录: /home/MuyuWorkSpace
    文件大小: 16.40 KB
    
    --------------------------------------------------------------------------------
    解析结果预览（前 1500 字符）:
    --------------------------------------------------------------------------------
    # 五、项目：PaddleOCR-VL构建多模态RAG系统
    
    本节内容，我们将详细介绍如何部署和运行这个基于PaddleOCR-VL的多模态AgenticRAG智能问答系统。该系统支持复杂PDF文档、图片、表格、公式等多种格式的智能分析和问答，并具备精准的溯源能力。
    
    ![](images/8b615bdc06de102e742913e121d91c5ec9905e922f853a2a988eebefa8ba496f.jpg)
    
    # 5.1 项目核心模块代码详解
    
    PaddleOCR-VL的输出格式非常适合构建多模态RAG系统，如果想要明确的区分出图像、表格、普通文本等信息，一个基本的处理流程是这样的：
    
    ```txt
    PaddleOCR-VL JSON 输出↓
    ```
    
    1. 数据预处理
    
    ```txt
    按block_order排序过滤无用内容（footer等）合并相邻同类型块
    ```
    
    2. 分类处理
    
    ```txt
    文本类  $\rightarrow$  标准 chunk 表格类  $\rightarrow$  结构化提取  $+$  文本描述 公式类  $\rightarrow$  保留格式  $+$  语义转换 图片类  $\rightarrow$  多模态向量 / 标题关联
    ```
    
    3. 元数据增强
    
    ```txt
    - block_id（溯源ID）
    - block_bbox（位置坐标）
    - block_type（元素类型）
    - page_index（页码）
    - 上下文信息（前后标题、图表编号）
    ```
    
    4. 向量化与索引
    
    ```txt
    文本 Embedding  
    表格 Embedding（多策略）建立多级索引
    ```
    
    5. 检索与溯源
    
    语义检索  
    坐标定位  
    可视化标注
    
    分享 | 下载(195.7M) | 删除 : 更多
    
    全部文件/木羽公开课资料/20251022基于PaddleOCR-VL搭建企...
    
    ![](images/409c2892525832b5465b80864b23240f91674807f62fbc6dfa25335a5dee2b19.jpg)
    
    ![](images/29ea9762cc653c55e580471be648e41292bf6730e1bf1696a59f07dcdafdfe26.jpg)
    
    首先第一步做的就是完成PaddleOCR-VL模型的接入及实现解析过程。核心代码文件为:
    
    ![](images/01aea784451e2cca4b44b64bbb05f9d9998fe4232ea886bf087b8b4157736285.jpg)
    
    这个服务的核心流程是：初始化时异步加载 PaddleOCRVL 模型到，然后在 parse_document() 中通过线程池执行阻塞的 OCR 调用，PaddleOCR 会将文档解析为多页结果并保存为 JSON/Markdown/可视化图片到磁盘。服务优先从生成的 *_res.json 文件中读取每页的 parsing_res_list，将其中的每个 block（包含 block_id、block_label、block_content、block_bbox、block_order 等字段）转换为 ParsingBlock 对象，最后通过 calculate.stats() 按 label 关键词（table/image/formula/其他）统计各类型块的数量并返回 DocumentStats。即：
    
    1. 初始化  $\rightar
    
    ... (后续内容已省略)
    --------------------------------------------------------------------------------
    
    ================================================================================
    成功！
    ================================================================================


&emsp;&emsp;同时也可以看到后端的服务运行情况：

<div align="center">
    <img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202511061214166.png" alt="MinerU 配置参考" width="70%">
</div>

