#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据抽取脚本 - 用于在有SSMS的电脑上运行
抽取ESG数据并生成CSV文件，带回进行BGE embedding
"""

import os
import pandas as pd
import pyodbc
from datetime import datetime

def extract_esg_data():
    """抽取ESG数据并保存为CSV文件"""
    print("🚀 ESG数据抽取脚本")
    print("=" * 50)
    
    try:
        # 1. 连接SQL Server
        print("🔌 连接SQL Server...")
        CNSTR = r"DRIVER={ODBC Driver 17 for SQL Server};SERVER=localhost;DATABASE=ESG;Trusted_Connection=yes;"
        cn = pyodbc.connect(CNSTR, autocommit=True)
        print("✅ SQL Server连接成功")
        
        # 2. 检查数据量
        print("📊 检查数据量...")
        total_count = pd.read_sql("SELECT COUNT(*) FROM dbo.fact_observation WHERE (value_numeric IS NOT NULL OR value_text IS NOT NULL)", cn).iloc[0, 0]
        print(f"📈 总记录数: {total_count}")
        
        # 3. 抽取维度表数据
        print("📋 抽取维度表数据...")
        
        # 公司维度表
        print("   📊 抽取公司维度表...")
        dim_company = pd.read_sql("SELECT ticker, company_name FROM dbo.dim_company ORDER BY ticker", cn)
        print(f"   ✅ 公司维度表: {len(dim_company)} 条记录")
        
        # 字段维度表
        print("   📊 抽取字段维度表...")
        dim_field = pd.read_sql("SELECT field_code, field_name, esg_bucket FROM dbo.dim_field ORDER BY field_code", cn)
        print(f"   ✅ 字段维度表: {len(dim_field)} 条记录")
        
        # 4. 抽取事实表数据
        print("📊 抽取事实表数据...")
        
        # 分批抽取，避免内存问题
        batch_size = 10000
        all_facts = []
        
        sql_facts = """
        SELECT 
            f.ticker, 
            f.[year], 
            f.field_code,
            COALESCE(CAST(f.value_numeric AS varchar(100)), f.value_text) AS val,
            f.incomplete, 
            f.source_file,
            d.field_name, 
            d.esg_bucket, 
            c.company_name
        FROM dbo.fact_observation f
        JOIN dbo.dim_field d ON d.field_code = f.field_code
        JOIN dbo.dim_company c ON c.ticker = f.ticker
        WHERE (f.value_numeric IS NOT NULL OR f.value_text IS NOT NULL)
        ORDER BY f.ticker, f.[year], d.field_code
        """
        
        print("   📥 开始分批抽取事实数据...")
        offset = 0
        
        while True:
            sql_batch = f"""
            {sql_facts}
            OFFSET {offset} ROWS
            FETCH NEXT {batch_size} ROWS ONLY
            """
            
            batch_df = pd.read_sql(sql_batch, cn)
            
            if batch_df.empty:
                break
                
            all_facts.append(batch_df)
            offset += batch_size
            
            print(f"   📊 已抽取 {offset} 条记录...")
            
            if len(batch_df) < batch_size:
                break
        
        # 合并所有批次
        if all_facts:
            facts = pd.concat(all_facts, ignore_index=True)
            print(f"   ✅ 事实表数据: {len(facts)} 条记录")
        else:
            print("   ❌ 没有抽取到事实数据")
            return
        
        # 5. 创建输出目录
        output_dir = "esg_data_export"
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 创建输出目录: {output_dir}")
        
        # 6. 保存数据到CSV文件
        print("💾 保存数据到CSV文件...")
        
        # 保存维度表
        dim_company.to_csv(os.path.join(output_dir, "dim_company.csv"), index=False, encoding='utf-8-sig')
        dim_field.to_csv(os.path.join(output_dir, "dim_field.csv"), index=False, encoding='utf-8-sig')
        
        # 保存事实表
        facts.to_csv(os.path.join(output_dir, "fact_observation.csv"), index=False, encoding='utf-8-sig')
        
        # 7. 生成数据摘要
        print("📋 生成数据摘要...")
        
        summary = {
            "extraction_time": datetime.now().isoformat(),
            "total_companies": len(dim_company),
            "total_fields": len(dim_field),
            "total_facts": len(facts),
            "year_range": f"{facts['year'].min()}-{facts['year'].max()}",
            "companies_sample": dim_company['ticker'].head(10).tolist(),
            "fields_sample": dim_field['field_code'].head(10).tolist()
        }
        
        # 保存摘要
        import json
        with open(os.path.join(output_dir, "data_summary.json"), 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        # 8. 生成用于BGE embedding的文本数据
        print("📝 生成BGE embedding文本数据...")
        
        def make_bge_text(r):
            """生成BGE embedding用的文本"""
            return f"Company: {r.company_name} (Ticker: {r.ticker}) in {int(r.year)}: {r.field_name} (Code: {r.field_code}) = {r.val}"
        
        facts['bge_text'] = facts.apply(make_bge_text, axis=1)
        facts['id'] = facts.apply(lambda r: f"{r.ticker}_{int(r.year)}_{r.field_code}", axis=1)
        
        # 保存BGE文本数据
        bge_data = facts[['id', 'bge_text', 'ticker', 'company_name', 'year', 'field_code', 'field_name', 'value', 'esg_bucket']].copy()
        bge_data.to_csv(os.path.join(output_dir, "bge_embedding_data.csv"), index=False, encoding='utf-8-sig')
        
        print(f"✅ BGE文本数据: {len(bge_data)} 条记录")
        
        # 9. 输出完成信息
        print(f"\n🎉 数据抽取完成！")
        print("=" * 50)
        print(f"📁 输出目录: {os.path.abspath(output_dir)}")
        print(f"📊 文件列表:")
        print(f"   - dim_company.csv ({len(dim_company)} 条)")
        print(f"   - dim_field.csv ({len(dim_field)} 条)")
        print(f"   - fact_observation.csv ({len(facts)} 条)")
        print(f"   - bge_embedding_data.csv ({len(bge_data)} 条)")
        print(f"   - data_summary.json")
        
        print(f"\n📋 数据摘要:")
        print(f"   - 公司数量: {len(dim_company)}")
        print(f"   - 字段数量: {len(dim_field)}")
        print(f"   - 事实记录: {len(facts)}")
        print(f"   - 年份范围: {facts['year'].min()}-{facts['year'].max()}")
        
        print(f"\n💡 下一步:")
        print(f"1. 将 {output_dir} 文件夹复制到目标电脑")
        print(f"2. 运行 build_bge_from_csv.py 进行embedding")
        print(f"3. 使用新的Chroma数据库")
        
        # 10. 生成BGE构建脚本
        print(f"\n🔧 生成BGE构建脚本...")
        generate_bge_build_script(output_dir)
        
    except Exception as e:
        print(f"❌ 抽取失败: {str(e)}")
        import traceback
        traceback.print_exc()

def generate_bge_build_script(output_dir):
    """生成BGE构建脚本"""
    script_content = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从CSV文件构建BGE-M3数据库
"""

import os
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

def build_bge_from_csv():
    print("🚀 从CSV文件构建BGE-M3数据库")
    print("=" * 50)
    
    try:
        # 1. 加载BGE-M3模型
        print("📥 加载BGE-M3模型...")
        MODEL_NAME = "BAAI/bge-m3"
        model = SentenceTransformer(MODEL_NAME)
        print("✅ BGE-M3模型加载完成")
        
        # 2. 读取CSV数据
        print("📊 读取CSV数据...")
        data_file = "bge_embedding_data.csv"
        
        if not os.path.exists(data_file):
            print(f"❌ 找不到数据文件: {data_file}")
            return
        
        df = pd.read_csv(data_file, encoding='utf-8-sig')
        print(f"✅ 读取到 {len(df)} 条记录")
        
        # 3. 生成embedding
        print("🔬 生成BGE-M3 embedding...")
        texts = df['bge_text'].tolist()
        embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        print(f"✅ 生成 {len(embs)} 个embedding")
        
        # 4. 创建Chroma数据库
        print("💾 创建Chroma数据库...")
        db_path = "esg_chroma_bge_m3_final"
        
        if os.path.exists(db_path):
            import shutil
            shutil.rmtree(db_path)
            print(f"🗑️ 删除旧数据库")
        
        client = chromadb.PersistentClient(path=db_path)
        collection = client.create_collection(
            name="esg_bge_m3_final",
            metadata={
                "hnsw:space": "cosine",
                "model": MODEL_NAME,
                "version": "m3_final",
                "multilingual": True
            }
        )
        print(f"✅ 创建数据库: {db_path}")
        
        # 5. 添加数据
        print("💾 添加数据到Chroma...")
        metadatas = []
        for _, row in df.iterrows():
            metadatas.append({
                "ticker": str(row['ticker']).strip(),
                "company": row['company_name'],
                "year": int(row['year']),
                "bucket": row['esg_bucket'] or "",
                "field_code": row['field_code'],
                "field_name": row['field_name'],
                "value": row['value']
            })
        
        collection.add(
            ids=df['id'].tolist(),
            embeddings=embs.tolist(),
            documents=df['bge_text'].tolist(),
            metadatas=metadatas
        )
        
        print(f"✅ 成功添加 {len(df)} 条记录")
        
        # 6. 测试查询
        print("🧪 测试查询...")
        test_query = "A US Equity 2015年女性员工比例"
        query_embedding = model.encode([test_query])[0].tolist()
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
            include=['documents', 'metadatas', 'distances']
        )
        
        print(f"🔍 测试查询: {test_query}")
        for i, (doc, meta, dist) in enumerate(zip(results['documents'][0], results['metadatas'][0], results['distances'][0])):
            similarity = 1 - dist
            print(f"  {i+1}. 相似度: {similarity:.3f}")
            print(f"     公司: {meta.get('company', 'N/A')}")
            print(f"     股票代码: {meta.get('ticker', 'N/A')}")
            print(f"     年份: {meta.get('year', 'N/A')}")
            print(f"     指标: {meta.get('field_name', 'N/A')}")
        
        print(f"\\n🎉 BGE-M3构建完成！")
        print(f"📊 数据库路径: {os.path.abspath(db_path)}")
        print(f"📊 集合名称: esg_bge_m3_final")
        print(f"🤖 使用模型: {MODEL_NAME}")
        print(f"📈 总记录数: {collection.count()}")
        
    except Exception as e:
        print(f"❌ 构建失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    build_bge_from_csv()
'''
    
    script_path = os.path.join(output_dir, "build_bge_from_csv.py")
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print(f"✅ 生成BGE构建脚本: {script_path}")

if __name__ == "__main__":
    extract_esg_data()
