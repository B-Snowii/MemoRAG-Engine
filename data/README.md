# Data Processing Pipeline

[![Language](https://img.shields.io/badge/Language-English-blue)](README.md)
[![语言](https://img.shields.io/badge/语言-中文-red)](README.zh-CN.md)

This folder contains sample ESG data files and documentation for the complete data processing pipeline.

## 📊 Data Structure

The system processes ESG data with the following structure:
- **Company**: Company names and tickers (e.g., "AA US Equity", "Alcoa Corp")
- **Year**: Temporal data (2005-2024)
- **Indicators**: ESG metrics (emissions, workforce, governance)
- **Values**: Quantitative measurements
- **Metadata**: Additional context and quality scores

### Sample Data Details

The `data/` folder contains sample ESG datasets:
- `fact_observation.csv` - Main ESG observations data
- `dim_company.csv` - Company dimension data
- `dim_field.csv` - Field/indicator dimension data
- `bge_embedding_data.csv` - Pre-computed embeddings
- `trend_analysis_data.csv` - Trend analysis data
- `company_statistics.csv` - Company statistics
- `data_summary.json` - Data summary and metadata

## 🏗️ Complete Data Pipeline

📊 Excel Data → 🗄️ SQL Server → 📝 CSV Export → 🧠 BGE Embedding → 💾 ChromaDB → 🤖 MemoRAG

### Step 1: Import Excel Data to SQL Server
```bash
python import_esg_to_sql.py
```

### Step 2: Create Database Views
```bash
python create_views.py
```

### Step 3: Extract Data for BGE Embedding
```bash
python extract_data_for_bge.py
```

### Step 4: Build ChromaDB from CSV
```bash
python esg_data_export/build_bge_from_csv.py
```

### Step 5: Run MemoRAG System
```bash
python MemoRAG-Engine-ESG-Analyst.py
```
