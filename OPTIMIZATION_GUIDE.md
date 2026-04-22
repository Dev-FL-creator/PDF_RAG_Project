# PDF RAG 系统优化说明

## 🎯 优化内容

本次优化按照技术面试要求，实现了以下关键功能：

### 1. ✅ 语义分块 (Semantic Chunking)
- **不再使用简单的固定大小分块**
- 使用智能语义边界识别：
  - 段落边界检测
  - 句子边界检测（使用 NLTK）
  - 表格完整性保留
  - 列表结构识别
- 自适应分块大小（默认目标 800 字符，范围 200-1500）
- 保持上下文连贯性

**实现位置**: `backend/utils/semantic_chunker.py`

### 2. ✅ 增强的表格处理
- 使用 Azure Document Intelligence 提取表格结构
- 保留行列信息
- TSV 格式存储（便于理解和检索）
- 大表自动智能分割
- 每个表格标注行数和列数

**实现位置**: `backend/routers/pdf_rag_routes.py` 中的 `extract_pages_with_docint()`

### 3. ✅ 精确的页码追踪
- 每个内容元素（段落、表格、图片）都记录准确的页码
- 从 Azure Document Intelligence 的 `bounding_regions` 获取页码
- 分块时保留页码信息
- 检索结果返回页码，方便溯源

**数据结构**:
```python
{
    "content": "...",
    "page": 3,  # 精确的页码
    "chunk_id": 0,
    "chunk_type": "paragraph"
}
```

### 4. ✅ 图片内容理解（非仅 OCR）
- 使用 **GPT-4o Vision** 理解图片语义
- 提取图片类型（图表、照片、流程图、示意图等）
- 生成详细的图片描述
- 识别关键元素和文字
- 集成到文档索引中

**实现位置**: `backend/utils/image_analyzer.py`

**功能特点**:
- 自动从 PDF 提取图片（使用 PyMuPDF）
- 分析图片内容和含义
- 识别图表、示意图、照片等
- 提取图中的文字标注
- 为每张图片生成可搜索的描述

---

## 📦 新增依赖

在 `requirements.txt` 中添加了以下包：

```
langchain==0.3.15
langchain-text-splitters==0.3.5
langchain-openai==0.2.17
nltk==3.9.1
Pillow==11.1.0
PyMuPDF==1.25.5
```

## 🔧 配置更新

在 `config.json` 中新增配置项：

```json
{
  "enable_image_analysis": true,    // 是否启用图片分析
  "chunk_target_size": 800,         // 目标分块大小
  "chunk_min_size": 200,            // 最小分块大小
  "chunk_max_size": 1500            // 最大分块大小
}
```

## 🚀 使用方法

### 1. 安装依赖

```powershell
cd backend
.\pdf_rag_venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. 启动服务

```powershell
# 在项目根目录
.\start.ps1
```

### 3. 上传 PDF

上传 PDF 时，系统会自动：
1. 提取文本、表格和图片
2. 使用 GPT-4o Vision 分析图片内容
3. 使用语义分块处理文本
4. 为每个 chunk 记录准确的页码
5. 将所有内容索引到 Azure Cognitive Search

### 4. 查询文档

查询时会返回：
- 匹配的内容片段
- **准确的页码**（来自哪一页）
- Chunk ID（方便定位）
- 如果 chunk 包含图片，会包含图片描述

---

## 🎨 技术亮点

### 语义分块示例

**原来的简单分块**:
```
固定 800 字符 → 可能在句子中间截断
```

**现在的语义分块**:
```
按段落分 → 如果段落太大，按句子分 → 保持语义完整性
表格单独处理 → 保持表格结构完整
```

### 图片理解示例

**传统 OCR**:
```
只能识别: "Temperature: 25°C"
```

**GPT-4o Vision**:
```
描述: "This is a line chart showing temperature variations over time. 
       The x-axis represents hours (0-24) and y-axis shows temperature in Celsius.
       Key elements: temperature curve, grid lines, axis labels.
       Text detected: Temperature: 25°C, Time: 12:00"
内容类型: chart
置信度: high
```

### 页码追踪示例

每个检索结果都包含准确位置：

```json
{
  "content": "系统架构包含三层...",
  "source": "technical_doc.pdf",
  "page": 5,
  "chunk_id": 2
}
```

在回答中显示为：
```
[technical_doc.pdf p.5 #2] 系统架构包含三层...
```

---

## 📊 性能优化

1. **图片分析缓存**: 同一图片不会重复分析
2. **批量处理**: Embedding 使用批量 API 调用
3. **并行处理**: 多个图片可并行分析（如果需要）
4. **智能过滤**: 过滤太小的图片（< 100px）

---

## 🔍 验证方法

### 测试语义分块

上传一个包含多个段落和表格的 PDF，检查：
- [ ] 段落是否完整（不在句子中间截断）
- [ ] 表格是否保持完整结构
- [ ] 分块大小是否合理（200-1500 字符）

### 测试图片理解

上传一个包含图表、照片的 PDF，检查：
- [ ] 是否识别到图片
- [ ] 图片描述是否准确
- [ ] 是否理解图表含义（不只是 OCR）

### 测试页码追踪

上传多页 PDF 并查询，检查：
- [ ] 返回结果是否标注正确页码
- [ ] 不同页面的内容页码是否不同
- [ ] Agent trace 中是否显示页码

---

## 🐛 故障排查

### 问题：NLTK 下载失败

```powershell
python -c "import nltk; nltk.download('punkt')"
```

### 问题：PyMuPDF 安装失败

```powershell
pip install --upgrade pip
pip install PyMuPDF==1.25.5
```

### 问题：图片分析失败

1. 检查 `config.json` 中 `openai_deployment` 是否为 `gpt-4o`
2. 确保 Azure OpenAI 部署支持 Vision 功能
3. 查看日志中的详细错误信息

### 问题：Import 错误

```powershell
# 确保在虚拟环境中
cd backend
.\pdf_rag_venv\Scripts\Activate.ps1

# 重新安装依赖
pip install -r requirements.txt
```

---

## 📝 代码结构

```
backend/
├── utils/
│   ├── __init__.py
│   ├── semantic_chunker.py      # 语义分块模块
│   └── image_analyzer.py        # 图片理解模块
├── routers/
│   └── pdf_rag_routes.py        # 优化的 PDF 处理路由
├── AI_AGENT/
│   └── pdf_agent_engine.py      # Agent 引擎
├── config.json                   # 配置文件（已更新）
└── requirements.txt              # 依赖（已更新）
```

---

## 🎓 面试问题对照

### Q1: 如何实现语义分块？
**A**: 使用 `SemanticChunker` 类，基于段落和句子边界，而非固定字符数。使用 NLTK 进行句子分割，保持表格完整性。

### Q2: 如何处理表格？
**A**: 使用 Azure Document Intelligence 提取表格结构，保留行列信息，以 TSV 格式存储，大表智能分割。

### Q3: 如何追踪页码？
**A**: 从 Azure Document Intelligence 的 `bounding_regions` 获取每个元素的页码，在分块和索引时保留，检索结果中返回。

### Q4: 如何理解图片（非 OCR）？
**A**: 使用 GPT-4o Vision API，不仅提取文字，还理解图表类型、内容含义、关键元素等语义信息。

### Q5: 检索结果如何返回页码？
**A**: 每个 chunk 在索引时存储 `page` 字段，检索后在结果中显示为 `[filename p.X #Y]` 格式。

---

## ✅ 优化检查清单

- [x] 语义分块（段落、句子边界）
- [x] 表格结构保留
- [x] 页码精确追踪
- [x] 图片内容理解（GPT-4o Vision）
- [x] 检索结果返回页码
- [x] 配置文件更新
- [x] 依赖包更新
- [x] 代码文档完善

---

## 📚 相关文档

- [Azure Document Intelligence](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/)
- [GPT-4o Vision](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/gpt-with-vision)
- [NLTK Documentation](https://www.nltk.org/)
- [PyMuPDF Documentation](https://pymupdf.readthedocs.io/)

---

## 💡 下一步优化建议

1. **多模态向量检索**: 为图片生成单独的向量
2. **表格专用索引**: 为表格创建结构化查询
3. **跨页引用**: 处理跨页的内容关联
4. **PDF 标注**: 在原 PDF 上高亮检索结果
5. **缓存优化**: 缓存分析结果避免重复处理
