import os, pyodbc, pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb

# --- Config ---
CNSTR = r"DRIVER={ODBC Driver 17 for SQL Server};SERVER=localhost;DATABASE=ESG;Trusted_Connection=yes;"
MAX_ROWS = int(os.environ.get("ESG_EMBED_MAX", "5000"))   
FILTER_TICKER = os.environ.get("ESG_FILTER_TICKER")      
FILTER_YEAR = os.environ.get("ESG_FILTER_YEAR")          
MODEL_NAME = os.environ.get("ESG_EMBED_MODEL", "intfloat/multilingual-e5-base")

FILTER_COMPLETE = os.environ.get("ESG_FILTER_COMPLETE", "0") == "1"
COLLECTION_NAME = os.environ.get("ESG_COLLECTION", "esg")

print(f"Embedding model: {MODEL_NAME}")
print(f"Filters: ticker={FILTER_TICKER}, year={FILTER_YEAR}")

# --- SQL Server connection ---
cn = pyodbc.connect(CNSTR, autocommit=True)

# WHERE
where = ["(f.value_numeric IS NOT NULL OR f.value_text IS NOT NULL)"]
params = []
if FILTER_TICKER:
    where.append("f.ticker = ?")
    params.append(FILTER_TICKER)
if FILTER_YEAR:
    where.append("f.[year] = ?")
    params.append(int(FILTER_YEAR))
if FILTER_COMPLETE:
    where.append("f.incomplete = 0")
where_sql = " AND ".join(where)

# 总数
sql_count = f"SELECT COUNT(*) FROM dbo.fact_observation f WHERE {where_sql}"
total = pd.read_sql(sql_count, cn, params=params).iloc[0, 0]
limit = min(total, MAX_ROWS)
print(f"total={total}, plan_to_index={limit}")

# 分页模板（keyset）
sql_page_tpl = """
SELECT TOP {n} f.ticker, c.company_name, f.[year], d.esg_bucket, d.field_code, d.field_name,
       COALESCE(CAST(f.value_numeric AS varchar(100)), f.value_text) AS val,
       f.source_file, f.incomplete
FROM dbo.fact_observation f
JOIN dbo.dim_field   d ON d.field_code = f.field_code
JOIN dbo.dim_company c ON c.ticker = f.ticker
WHERE {where_sql} AND (f.ticker + '|' + CAST(f.[year] AS varchar(10)) + '|' + d.field_code) > ?
ORDER BY f.ticker, f.[year], d.field_code
"""

def make_text(r):
    return f"passage: {r.company_name} ({r.ticker}) in {int(r.year)}: {r.field_name} (code={r.field_code}) = {r.val}"

# 模型与 Chroma
model = SentenceTransformer(MODEL_NAME)
persist_dir = os.path.join(os.path.dirname(__file__), "esg_chroma")
client = chromadb.PersistentClient(path=persist_dir)
coll = client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

# 迭代写入
inserted, last_key = 0, ""
BATCH = 2048
while inserted < limit:
    n = min(BATCH, limit - inserted)
    sql = sql_page_tpl.format(n=n, where_sql=where_sql)
    df = pd.read_sql(sql, cn, params=params + [last_key])
    if df.empty:
        break
    df = df.iloc[:n]
    df["text"] = df.apply(make_text, axis=1)
    df["id"] = df.apply(lambda r: f"{r.ticker}_{int(r.year)}_{r.field_code}", axis=1)
    embs = model.encode(df["text"].tolist(), normalize_embeddings=True).tolist()
    coll.add(
        ids=df["id"].tolist(),
        embeddings=embs,
        documents=df["text"].tolist(),
        metadatas=[{
            "ticker": str(r.ticker).strip().strip("'\""),
            "company": r.company_name,
            "year": int(r.year),
            "year_s": str(int(r.year)),
            "bucket": r.esg_bucket or "",
            "field_code": r.field_code,
            "field_name": r.field_name,
            "value": r.val,
            "incomplete": bool(r.incomplete),
            "source_file": r.source_file
        } for r in df.itertuples(index=False)]
    )
    inserted += len(df)
    last_key = f"{df.iloc[-1].ticker}|{int(df.iloc[-1].year)}|{df.iloc[-1].field_code}"
    print(f"indexed {inserted}/{limit}")

print("done, persisted at:", persist_dir)