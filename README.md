# MemoRAG-Engine


[![Language](https://img.shields.io/badge/Language-English-blue)](README.md)
[![语言](https://img.shields.io/badge/语言-中文-red)](README.zh-CN.md)

## 🚀 Overview

MemoRAG-Engine is an intelligent Retrieval-Augmented Generation (RAG) system that combines advanced semantic search with memory management and context awareness. Built with BGE-M3 embedding model and ChromaDB, it demonstrates how to build sophisticated MemoRAG systems with persistent memory and intelligent query processing.

**Why RAG?** RAG combines LLM reasoning with external knowledge, reduces hallucinations through data-based responses, efficiently handles large knowledge bases, and allows easy knowledge updates without retraining models.

**Why MemoRAG?** MemoRAG provides persistent memory that remembers previous queries and patterns, maintains conversation context across sessions, improves over time through user interactions, and identifies common query patterns and trends.


This project uses ESG (Environmental, Social, and Governance) data as a practical example to showcase MemoRAG capabilities.


## 🏗️ System Architecture

User Query → Query Processor → BGE-M3 Embedding → ChromaDB Search → Post-filtering → Intelligent Ranking → LLM Response → Memory Update → Professional ESG Analysis Report.

## 📊 RAGAS Evaluation Results

Our MemoRAG-Engine system achieved **perfect scores** in all RAGAS metrics:

| Metric | Score |
|--------|-------|
| **Context Recall** | 1.0000 |
| **Faithfulness** | 1.0000 |
| **Factual Correctness** | 1.0000 |
| **Overall Score** | 1.0000 |

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

