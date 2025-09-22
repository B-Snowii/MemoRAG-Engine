# MemoRAG-Engine


[![Language](https://img.shields.io/badge/Language-English-blue)](README.md)
[![语言](https://img.shields.io/badge/语言-中文-red)](README.zh-CN.md)

## 🚀 Overview

MemoRAG-Engine is a bilingual, memory-augmented Retrieval-Augmented Generation (RAG) system for ESG analytics. It ingests tabular ESG data from SQL Server into Chroma as row-level embeddings (no document chunking; optional light SQL prefilters by ticker/year/completeness). Retrieval is vector-only with BAAI/bge-m3, followed by metadata-aware re-ranking (validity, field match to extracted company/year/indicator/code, similarity, ESG category). The system supports context carryover and persistent memory, and can generate LLM responses with DeepSeek-V3.1.


**Why RAG?** RAG combines LLM reasoning with external knowledge, reduces hallucinations through data-based responses, efficiently handles large knowledge bases, and allows easy knowledge updates without retraining models.

**Why MemoRAG?** MemoRAG provides persistent memory that remembers previous queries and patterns, maintains conversation context across sessions, improves over time through user interactions, and identifies common query patterns and trends.


This project uses ESG (Environmental, Social, and Governance) data as a practical example to showcase MemoRAG capabilities.


## 🏗️ System Architecture

User Query → Query Processor → BGE-M3 Embedding → ChromaDB Search → Post-filtering → Intelligent Ranking → LLM Response → Memory Update → Professional ESG Analysis Report.

## 📊 RAGAS Evaluation Results

Pilot results on a small ESG QA set (n=5): Context Recall 0.98, Faithfulness 0.92, Factual Correctness 0.91.

## 🚀 Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/B-Snowii/MemoRAG-Engine.git
   cd MemoRAG-Engine
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run with sample data**:
   ```bash
   python MemoRAG-Engine-ESG-Analyst.py
   ```
   The system will automatically:
   - Load ESG sample data from `data/` folder
   - Create embeddings using BGE-M3 model
   - Build ChromaDB vector database
   - Initialize memory system



### Available Commands

- `help` - Display help information
- `collections` - Show available data collections
- `memory` - Display memory report
- `clear` - Clear memory
- `mode` - Toggle response mode (LLM/Basic)
- `debug` - Enable/disable debug mode
- `quit` - Exit system

## 💡 Usage Examples

### Basic Queries
```
A US Equity 2015 Pct Women in Workforce indicator
Agilent Technologies Inc 2015 women workforce percentage
ES047 indicator data
2015 environmental emissions trend
```

### Contextual Queries
```
How about this trend? (follow-up to previous query)
What about 2016? (contextual year query)
```

### Advanced Queries
```
Compare Alcoa Corp emissions between 2007 and 2010
Show me nitrogen oxide trends for industrial companies
```








## 📄 License

This project is licensed under the MIT License.


---

