import pyodbc

cn = pyodbc.connect(r"DRIVER={ODBC Driver 17 for SQL Server};SERVER=localhost;DATABASE=ESG;Trusted_Connection=yes;", autocommit=True)
cur = cn.cursor()

# 一行=公司×年份；将 ES 与 G（以及未来其它桶）合并到一个 JSON 列 ESG_json
cur.execute("""
CREATE OR ALTER VIEW dbo.vw_company_year_esg_all AS
SELECT
  t.ticker,
  c.company_name,
  t.[year],
  (
    SELECT d.field_code,
           bucket = d.esg_bucket,
           field_name = d.field_name,
           value = COALESCE(CAST(f.value_numeric AS varchar(100)), f.value_text)
    FROM dbo.fact_observation f
    JOIN dbo.dim_field d ON d.field_code = f.field_code
    WHERE f.ticker = t.ticker AND f.[year] = t.[year]
    FOR JSON PATH
  ) AS ESG_json
FROM (SELECT DISTINCT ticker,[year] FROM dbo.fact_observation) t
JOIN dbo.dim_company c ON c.ticker = t.ticker;
""")

# 简