# 数据处理管道

[![Language](https://img.shields.io/badge/Language-English-blue)](README.md)
[![语言](https://img.shields.io/badge/语言-中文-red)](README.zh-CN.md)

此文件夹包含ESG样本数据文件和完整数据处理管道的文档。

## 📊 数据结构

系统处理具有以下结构的ESG数据：
- **公司**：公司名称和股票代码（如"AA US Equity", "Alcoa Corp"）
- **年份**：时间数据（2005-2024）
- **指标**：ESG指标（排放、劳动力、治理）
- **数值**：定量测量
- **元数据**：额外上下文和质量分数

### 样本数据详情

`data/` 文件夹包含样本ESG数据集：
- `fact_observation.csv` - 主要ESG观测数据
- `dim_company.csv` - 公司维度数据
- `dim_field.csv` - 字段/指标维度数据
- `bge_embedding_data.csv` - 预计算嵌入
- `trend_analysis_data.csv` - 趋势分析数据
- `company_statistics.csv` - 公司统计
- `data_summary.json` - 数据摘要和元数据

## 🏗️ 完整数据管道

📊 Excel数据 → 🗄️ SQL Server → 📝 CSV导出 → 🧠 BGE嵌入 → 💾 ChromaDB → 🤖 MemoRAG

### 步骤1：将Excel数据导入SQL Server
```bash
python import_esg_to_sql.py
```

### 步骤2：创建数据库视图
```bash
python create_views.py
```

### 步骤3：提取数据用于BGE嵌入
```bash
python extract_data_for_bge.py
```

### 步骤4：从CSV构建ChromaDB
```bash
python esg_data_export/build_bge_from_csv.py
```

### 步骤5：运行MemoRAG系统
```bash
python MemoRAG-Engine-ESG-Analyst.py
```

