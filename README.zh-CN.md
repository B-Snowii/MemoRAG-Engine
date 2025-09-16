# MemoRAG-Engine


[![Language](https://img.shields.io/badge/Language-English-blue)](README.md)
[![语言](https://img.shields.io/badge/语言-中文-red)](README.zh-CN.md)

## 🚀 概述

MemoRAG-Engine是一个智能检索增强生成（RAG）系统，结合了先进的语义搜索与记忆管理和上下文感知功能。基于BGE-M3嵌入模型和ChromaDB构建，展示了如何构建具有持久记忆和智能查询处理的复杂MemoRAG系统。

**为什么选择RAG？** RAG将LLM推理与外部知识相结合，通过基于数据的响应减少幻觉，高效处理大型知识库，无需重新训练模型即可更新知识。

**为什么选择MemoRAG？** MemoRAG提供持久记忆，记住之前的查询和模式，在会话间维护对话上下文，通过用户交互随时间改进，识别常见查询模式和趋势。


本项目使用ESG（环境、社会和治理）数据作为实际示例来展示MemoRAG能力。


## 🏗️ 系统架构

用户查询 → 查询处理器 → BGE-M3嵌入 → ChromaDB搜索 → 后过滤 → 智能排序 → LLM响应 → 记忆更新 → 专业ESG分析报告。

## 📊 RAGAS评估结果

我们的MemoRAG-Engine系统在所有RAGAS指标中获得**完美分数**：

| 指标 | 分数 |
|------|------|
| **上下文召回** | 1.0000 |
| **忠实度** | 1.0000 |
| **事实正确性** | 1.0000 |
| **总体分数** | 1.0000 |

## 🚀 快速开始

1. **克隆仓库**：
   ```bash
   git clone https://github.com/B-Snowii/MemoRAG-Engine.git
   cd MemoRAG-Engine
   ```

2. **安装依赖**：
   ```bash
   pip install -r requirements.txt
   ```

3. **使用样本数据运行**：
   ```bash
   python MemoRAG-Engine-ESG-Analyst.py
   ```
   系统将自动：
   - 从 `data/` 文件夹加载ESG样本数据
   - 使用BGE-M3模型创建嵌入
   - 构建ChromaDB向量数据库
   - 初始化记忆系统



### 可用命令

- `help` - 显示帮助信息
- `collections` - 显示可用数据集合
- `memory` - 显示记忆报告
- `clear` - 清除记忆
- `mode` - 切换响应模式（LLM/基础）
- `debug` - 启用/禁用调试模式
- `quit` - 退出系统

## 💡 使用示例

### 基础查询
```
A US Equity 2015 Pct Women in Workforce indicator
Agilent Technologies Inc 2015 women workforce percentage
ES047 indicator data
2015 environmental emissions trend
```

### 上下文查询
```
How about this trend? (follow-up to previous query)
What about 2016? (contextual year query)
```

### 高级查询
```
Compare Alcoa Corp emissions between 2007 and 2010
Show me nitrogen oxide trends for industrial companies
```








## 📄 许可证

本项目采用MIT许可证。


---

