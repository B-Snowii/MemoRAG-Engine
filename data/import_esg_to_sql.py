import os, math, pandas as pd, pyodbc

BASE_DIR = r"C:\Users\HKUBS\Desktop\database"
SERVER   = r"localhost"
DB       = "ESG"
DRIVERS  = ["ODBC Driver 18 for SQL Server","ODBC Driver 17 for SQL Server","SQL Server"]

def conn(db):
    last=None
    for d in DRIVERS:
        try:
            return pyodbc.connect(f"DRIVER={{{d}}};SERVER={SERVER};DATABASE={db};Trusted_Connection=yes;", timeout=5, autocommit=True)
        except Exception as e: last=e
    raise last

def init_db():
    cn = conn("master")
    cur = cn.cursor()
    cur.execute("IF DB_ID('ESG') IS NULL CREATE DATABASE ESG;")
    cur.commit(); cn.close()

    cn = conn(DB); cur = cn.cursor()
    cur.execute("""IF OBJECT_ID('dbo.dim_field','U') IS NULL
    CREATE TABLE dbo.dim_field (
      field_code varchar(20) PRIMARY KEY,
      field_name nvarchar(400) NULL,
      esg_bucket varchar(16) NULL
    );""")
    cur.execute("""IF OBJECT_ID('dbo.dim_company','U') IS NULL
    CREATE TABLE dbo.dim_company (
      ticker varchar(40) PRIMARY KEY,
      company_name nvarchar(300) NULL
    );""")
    cur.execute("""IF OBJECT_ID('dbo.fact_observation','U') IS NULL
    CREATE TABLE dbo.fact_observation (
      ticker varchar(40) NOT NULL,
      [year] int NOT NULL,
      field_code varchar(20) NOT NULL,
      value_numeric float NULL,
      value_text nvarchar(400) NULL,
      incomplete bit NOT NULL,
      source_file nvarchar(260) NULL,
      CONSTRAINT PK_fact PRIMARY KEY (ticker,[year],field_code)
    );""")
    cur.execute("""IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name='FK_fact_field')
      ALTER TABLE dbo.fact_observation ADD CONSTRAINT FK_fact_field
      FOREIGN KEY (field_code) REFERENCES dbo.dim_field(field_code);""")
    cur.execute("""IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name='FK_fact_co')
      ALTER TABLE dbo.fact_observation ADD CONSTRAINT FK_fact_co
      FOREIGN KEY (ticker) REFERENCES dbo.dim_company(ticker);""")
    cur.commit(); cn.close()

def read_book(path):
    fdesc = pd.read_excel(path, sheet_name="Field Descriptions", engine="openpyxl")
    fdesc.columns = [str(c).strip() for c in fdesc.columns]
    fields = fdesc[[fdesc.columns[0], fdesc.columns[1]]].dropna()
    fields = fields.rename(columns={fdesc.columns[0]:"field_code", fdesc.columns[1]:"field_name"})
    base = os.path.basename(path).lower()
    fields["esg_bucket"] = "ES" if "_es_" in base else ("G" if "_g_" in base else None)

    df = pd.read_excel(path, sheet_name="Data", engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]
    inc = [c for c in df.columns if c.lower().startswith("incomplete")][0]
    tic = [c for c in df.columns if c.lower().startswith("ticker")][0]
    co  = [c for c in df.columns if "company" in c.lower()][0]
    yr  = [c for c in df.columns if c.lower()=="year"][0]
    metrics = [c for c in df.columns if c not in {inc,tic,co,yr}]

    long = df.melt(id_vars=[inc,tic,co,yr], value_vars=metrics,
                   var_name="field_code", value_name="value_raw") \
            .rename(columns={inc:"incomplete",tic:"ticker",co:"company_name",yr:"year"})
    long["incomplete"] = long["incomplete"].astype(str).str.upper().eq("TRUE")
    long["field_code"] = long["field_code"].astype(str).str.strip()

    def to_f(x):
        s = str(x).strip()
        if s in ("","-","NA","N/A","None"): return math.nan
        try: return float(s)
        except: return math.nan
    long["value_numeric"] = long["value_raw"].apply(to_f)
    long["value_text"] = [None if pd.notna(n) else (None if str(r).strip() in ("","-") else str(r).strip())
                          for r,n in zip(long["value_raw"], long["value_numeric"])]
    long["source_file"] = os.path.basename(path)
    long = long[(~long["value_numeric"].isna()) | (long["value_text"].notna())]

    dim_company = long[["ticker","company_name"]].drop_duplicates()
    fact = long[["ticker","year","field_code","value_numeric","value_text","incomplete","source_file"]]
    return fields.drop_duplicates("field_code"), dim_company, fact

def upsert(conn, table, cols, keys, rows):
    if not rows: return 0
    cur = conn.cursor(); cur.fast_executemany = True
    cols_br = ", ".join(f"[{c}]" for c in cols)
    params  = ", ".join("?" for _ in cols)
    on   = " AND ".join(f"t.[{k}]=s.[{k}]" for k in keys)
    setc = ", ".join(f"t.[{c}]=s.[{c}]" for c in cols if c not in keys)
    sql = f"""MERGE {table} AS t
USING (VALUES ({params})) AS s({cols_br})
ON {on}
WHEN MATCHED THEN UPDATE SET {setc}
WHEN NOT MATCHED THEN INSERT ({cols_br}) VALUES ({cols_br});"""
    for i in range(0, len(rows), 20):
        cur.executemany(sql, rows[i:i+800]); conn.commit()
    return len(rows)

def main():
    init_db()
    conn_esg = conn(DB)
    paths = [os.path.join(BASE_DIR, f) for f in os.listdir(BASE_DIR)
             if f.lower().endswith((".xlsx",".xls",".xlsm")) and f.lower().startswith("combined_")]
    all_f, all_c, all_x = [], [], []
    for p in sorted(paths):
        f, c, x = read_book(p)
        all_f.append(f); all_c.append(c); all_x.append(x)
    fields = pd.concat(all_f, ignore_index=True).drop_duplicates("field_code").fillna("")
    comps  = pd.concat(all_c, ignore_index=True).drop_duplicates("ticker").fillna("")
    facts  = pd.concat(all_x, ignore_index=True)

    n1 = upsert(conn_esg, "dbo.dim_field", ["field_code","field_name","esg_bucket"], ["field_code"],
                fields[["field_code","field_name","esg_bucket"]].values.tolist())
    n2 = upsert(conn_esg, "dbo.dim_company", ["ticker","company_name"], ["ticker"],
                comps[["ticker","company_name"]].values.tolist())
    rows = []
    for r in facts.itertuples(index=False):
        rows.append((r.ticker, int(r.year), r.field_code,
                     None if pd.isna(r.value_numeric) else float(r.value_numeric),
                     r.value_text if r.value_text is not None else None,
                     1 if bool(r.incomplete) else 0, r.source_file))
    n3 = upsert(conn_esg, "dbo.fact_observation",
                ["ticker","year","field_code","value_numeric","value_text","incomplete","source_file"],
                ["ticker","year","field_code"], rows)
    print(f"fields={n1}, companies={n2}, facts={n3}")

if __name__ == "__main__":
    main()





