#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed Interactive MemoRAG System
Resolves Chroma query syntax issues
"""

import chromadb
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import json
import re
from datetime import datetime
from collections import defaultdict
import pickle
import os
import requests
import time

class LLMResponseGenerator:
    """LLM Response Generator"""
    
    def __init__(self, api_key: str = None):
        """Initialize LLM Response Generator"""
        self.api_key = api_key
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        self.language = "en"  # Default to English
        
    def generate_response(self, query: str, results: List[Dict], context: str = "") -> str:
        """Generate intelligent response"""
        try:
            if not self.api_key:
                return self._fallback_response(query, results)
            
            # Build prompt
            prompt = self._build_prompt(query, results, context)
            
            # Call API (with retry mechanism)
            response = self._call_api_with_retry(prompt)
            
            return response
            
        except Exception as e:
            print(f"LLM call failed: {str(e)}")
            print("🔄 Switching to basic response mode...")
            return self._fallback_response(query, results)
    
    def _call_api_with_retry(self, prompt: str, max_retries: int = 3) -> str:
        """API call with retry mechanism"""
        for attempt in range(max_retries):
            try:
                return self._call_api(prompt)
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"🔄 API call failed, retrying {attempt + 1}/{max_retries}: {str(e)}")
                    time.sleep(2)  # Wait 2 seconds before retry
                else:
                    raise e
    
    def _build_prompt(self, query: str, results: List[Dict], context: str) -> str:
        """Build prompt"""
        # Extract key data
        companies = list(set([r['esg_info'].get('company', '') for r in results if r['esg_info'].get('company')]))
        years = list(set([r['esg_info'].get('year', '') for r in results if r['esg_info'].get('year')]))
        indicators = list(set([r['esg_info'].get('indicator', '') for r in results if r['esg_info'].get('indicator')]))
        
        # Build data summary (prioritize valid data)
        data_summary = []
        valid_data_count = 0
        
        for i, r in enumerate(results[:10], 1):  # Show more data for LLM analysis
            esg_info = r['esg_info']
            value = esg_info.get('value', 'N/A')
            rerank_score = r.get('rerank_score', 0)
            
            # Mark data quality
            if value and str(value).lower() != 'nan' and str(value).strip():
                data_quality = "✅Valid data"
                valid_data_count += 1
            else:
                data_quality = "❌Missing data"
            
            data_summary.append(f"{i}. {data_quality} {esg_info.get('company', 'N/A')} ({esg_info.get('year', 'N/A')}) - {esg_info.get('indicator', 'N/A')}: {value} [Quality score:{rerank_score:.1f}]")
        
        # Add data quality statistics
        data_summary.append(f"\nData quality statistics: {valid_data_count} valid data, {len(results) - valid_data_count} missing data")
        
        # Determine response language based on language setting
        response_language = "English"
        if self.language == 'zh':
            response_language = "Chinese (中文)"
        
        prompt = f"""You are a professional ESG data analyst. Please generate a professional, accurate, and understandable response based on the following query and data.

Query: {query}

Data Summary (sorted by quality, ✅ indicates valid data, ❌ indicates missing data):
{chr(10).join(data_summary)}

Context: {context}

Important Notes:
- Prioritize using ✅ valid data for analysis
- For ❌ missing data, clearly explain the data gaps
- If valid data is insufficient, explain analysis limitations

Please generate a response that includes the following elements:
1. Direct answer to user's question (based on valid data)
2. Analysis of data trends and patterns (focus on valid data)
3. Provide professional insights
4. Clearly explain data gaps and reasons
5. Give recommendations or conclusions

Response Requirements:
- Use {response_language}
- Professional but understandable
- Based on valid data facts
- Clearly distinguish between valid and missing data
- Clear structure
- Appropriate length (200-400 words)

Response:"""
        
        return prompt
    
    def _call_api(self, prompt: str) -> str:
        """Call API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1000,
            "temperature": 0.7
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=data, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                raise Exception(f"API call failed: {response.status_code}, {response.text}")
        except requests.exceptions.Timeout:
            raise Exception("API call timeout, please check network connection")
        except requests.exceptions.ConnectionError:
            raise Exception("Network connection error, please check network")
        except Exception as e:
            raise Exception(f"API call exception: {str(e)}")
    
    def _fallback_response(self, query: str, results: List[Dict]) -> str:
        """Fallback response generation"""
        if not results:
            return "Sorry, no data related to your query was found. Please try adjusting your query conditions or check if the data exists."
        
        # Extract key information
        companies = list(set([r['esg_info'].get('company', '') for r in results if r['esg_info'].get('company')]))
        years = list(set([r['esg_info'].get('year', '') for r in results if r['esg_info'].get('year')]))
        indicators = list(set([r['esg_info'].get('indicator', '') for r in results if r['esg_info'].get('indicator')]))
        
        # Analyze data quality
        complete_data = [r for r in results if r['esg_info'].get('value') and str(r['esg_info'].get('value')).lower() != 'nan']
        missing_data = len(results) - len(complete_data)
        
        # Build intelligent response
        response_parts = []
        
        # Opening
        if '趋势' in query or 'trend' in query.lower():
            response_parts.append("根据趋势分析，")
        elif '比较' in query or '对比' in query or 'vs' in query.lower():
            response_parts.append("通过对比分析，")
        elif '具体' in query or '详细' in query:
            response_parts.append("具体数据显示，")
        else:
            response_parts.append("根据查询结果，")
        
        # 数据统计
        response_parts.append(f"我找到了 {len(results)} 条相关数据。")
        
        if companies:
            response_parts.append(f"涉及的公司包括: {', '.join(companies[:3])}。")
        
        if years:
            response_parts.append(f"数据年份范围: {min(years)}-{max(years)}。")
        
        if indicators:
            response_parts.append(f"主要指标包括: {', '.join(indicators[:3])}。")
        
        # 数据质量分析
        if missing_data > 0:
            response_parts.append(f"注意：有 {missing_data} 条数据缺失或为NaN值。")
        
        # 趋势分析
        if len(results) > 1 and years and len(years) > 1:
            response_parts.append("从时间维度看，数据呈现一定的变化趋势。")
        
        # 专业建议
        if missing_data > len(results) * 0.5:
            response_parts.append("建议：数据缺失较多，建议扩大数据源或调整查询条件。")
        else:
            response_parts.append("建议：可以基于现有数据进行进一步分析。")
        
        return " ".join(response_parts)

class OptimizedQueryProcessor:
    """Optimized Query Processor"""
    
    def __init__(self):
        """Initialize Query Processor"""
        
        # ESG professional terminology dictionary
        self.esg_terms = {
            # Environmental indicators
            'environment': ['环境', 'environment', '环保', '排放', 'emission', '碳', 'carbon'],
            'nitrogen_oxide': ['氮氧化物', 'nitrogen oxide', 'NOx', 'NO2'],
            'voc_emissions': ['VOC排放', 'VOC emissions', '挥发性有机化合物', 'volatile organic compound'],
            'carbon_monoxide': ['一氧化碳', 'carbon monoxide', 'CO'],
            'methane': ['甲烷', 'methane', 'CH4'],
            'particulate': ['颗粒物', 'particulate', 'PM', '粉尘'],
            'energy_consumption': ['能源消耗', 'energy consumption', '能耗'],
            'renewable_energy': ['可再生能源', 'renewable energy', '清洁能源'],
            'water_emissions': ['水排放', 'water emissions', '废水排放'],
            'hazardous_waste': ['危险废物', 'hazardous waste', '有害废物'],
            
            # Social indicators
            'social': ['社会', 'social', '社会责任', 'social responsibility'],
            'workforce': ['劳动力', 'workforce', '员工', 'employee', '人员'],
            'women_workforce': ['女性员工', 'women workforce', '女性劳动力', 'pct women', 'women percentage', 'Pct Women in Workforce'],
            'diversity': ['多样性', 'diversity', '多元化'],
            'safety': ['安全', 'safety', '职业安全', 'occupational safety'],
            'training': ['培训', 'training', '教育', 'education'],
            'community': ['社区', 'community', '社区参与', 'community engagement'],
            'human_rights': ['人权', 'human rights', '员工权利', 'worker rights'],
            'indigenous_rights': ['原住民权利', 'indigenous rights', '土著权利'],
            'strikes': ['罢工', 'strikes', '劳资纠纷', 'labor disputes'],
            
            # Governance indicators
            'governance': ['治理', 'governance', '公司治理', 'corporate governance', 'G类', 'G类指标', 'governance指标'],
            'board_diversity': ['董事会多样性', 'board diversity', '董事会多元化'],
            'executive_compensation': ['高管薪酬', 'executive compensation', '管理层薪酬'],
            'audit': ['审计', 'audit', '审计质量', 'audit quality'],
            'transparency': ['透明度', 'transparency', '信息披露', 'disclosure'],
            'ethics': ['道德', 'ethics', '商业道德', 'business ethics'],
            'compliance': ['合规', 'compliance', '法规遵循', 'regulatory compliance'],
            'risk_management': ['风险管理', 'risk management', '风险控制'],
            'stakeholder': ['利益相关者', 'stakeholder', '股东', 'shareholder'],
            'sustainability': ['可持续性', 'sustainability', '可持续发展'],
            'financial_literacy': ['财务素养', 'financial literacy', 'Financial Literacy Programs'],
            'management_diversity': ['管理层多样性', 'management diversity', 'Pct Minorities in Management'],
            
            # ES indicators
            'es_indicators': ['ES类', 'ES类指标', 'ES指标', 'ES类表现', 'ES表现'],
            'environmental_social': ['环境社会', 'environmental social', 'ES', 'E&S']
        }
        
        # Company name patterns (support ticker and company name)
        self.company_patterns = [
            # Ticker patterns (priority matching)
            r'([A-Z]{1,6}\s+US\s+Equity)',  # General ticker pattern (AA US Equity)
            r'(A US Equity|B US Equity|C US Equity|AA US Equity)',  # Specific ticker patterns
            # Company name patterns
            r'([A-Za-z\s]+(?:Inc|Corp|Ltd|Company|Technologies|Systems|Group|Holdings))',
            r'([A-Za-z\s]+(?:Inc\.|Corp\.|Ltd\.|Company\.))',
            r'([A-Za-z\s]+(?:Technologies|Systems|Group|Holdings))',
            # Mixed patterns
            r'([A-Za-z\s]+(?:Inc|Corp|Ltd|Company|Technologies|Systems|Group|Holdings)\s*\([A-Z]{1,6}\s+US\s+Equity\))',  # Company (Ticker)
            r'([A-Z]{1,6}\s+US\s+Equity\s*\([A-Za-z\s]+\))'  # Ticker (Company)
        ]
        
        # Year patterns
        self.year_patterns = [
            r'in\s+year\s+(\d{4})',  # in year 2006
            r'in\s+(\d{4})',        # in 2006
            r'year\s+(\d{4})',      # year 2006
            r'(\d{4})年',
            r'(\d{4})',
            r'(\d{4})年数据',
            r'(\d{4})年度'
        ]
        
        # Indicator code patterns
        self.indicator_code_patterns = [
            r'ES\d{3}',
            r'ES\d{2}',
            r'code=ES\d{3}',
            r'指标代码[：:]\s*ES\d{3}',
            r'ES\d{3}指标'
        ]
        
        # Query intent classification
        self.query_intents = {
            'trend': ['趋势', 'trend', '变化', 'change', '发展', 'development', '演变', 'evolution'],
            'comparison': ['比较', 'compare', '对比', '对比分析', 'comparative', 'vs', 'versus'],
            'specific': ['具体', 'specific', '详细', 'detail', '具体数据', 'specific data'],
            'overview': ['概览', 'overview', '总体', 'overall', '整体', 'general', '综合'],
            'analysis': ['分析', 'analysis', '研究', 'research', '评估', 'evaluation']
        }
    
    def process_query(self, raw_query: str, context: Dict = None) -> Dict[str, Any]:
        """Process raw query and return structured information"""
        # 1. Basic cleaning
        cleaned_query = self._clean_query(raw_query)
        
        # 2. Extract key information
        extracted_info = self._extract_key_information(cleaned_query)
        
        # 3. Handle context (e.g., "what about 2016?")
        if context and 'previous_query' in context:
            extracted_info = self._handle_context(extracted_info, context['previous_query'])
        
        # 4. Identify query intent
        intent = self._identify_intent(cleaned_query)
        
        # 5. Generate optimized query
        optimized_query = self._generate_optimized_query(extracted_info, intent)
        
        return {
            'raw_query': raw_query,
            'cleaned_query': cleaned_query,
            'extracted_info': extracted_info,
            'intent': intent,
            'optimized_query': optimized_query,
            'confidence': self._calculate_confidence(extracted_info),
            'timestamp': datetime.now().isoformat()
        }
    
    def _handle_context(self, extracted_info: Dict, previous_query: str) -> Dict:
        """Handle contextual queries"""
        # If current query lacks company info, get from previous query
        if not extracted_info['companies']:
            prev_companies = re.findall(r'([A-Za-z\s]+(?:Inc|Corp|Ltd|Company|Technologies|Systems|Group|Holdings|US Equity))', previous_query)
            if prev_companies:
                extracted_info['companies'] = prev_companies
        
        # If current query lacks indicator info, get from previous query
        if not extracted_info['indicators'] and not extracted_info['indicator_codes']:
            prev_indicators = re.findall(r'([A-Za-z\s]+(?:Emissions|Consumption|Policy|Rights|Workforce|Diversity))', previous_query)
            if prev_indicators:
                extracted_info['indicators'] = prev_indicators
        
        return extracted_info
    
    def _clean_query(self, query: str) -> str:
        """Clean query text"""
        # Remove extra spaces
        query = re.sub(r'\s+', ' ', query.strip())
        
        # Standardize punctuation
        query = query.replace('：', ':').replace('，', ',').replace('。', '.')
        
        # Remove special characters but keep important symbols
        query = re.sub(r'[^\w\s:,.()%-]', ' ', query)
        
        # Clean spaces again
        query = re.sub(r'\s+', ' ', query.strip())
        
        return query
    
    def _extract_key_information(self, query: str) -> Dict[str, Any]:
        """提取关键信息"""
        info = {
            'companies': [],
            'years': [],
            'indicators': [],
            'indicator_codes': [],
            'values': [],
            'esg_categories': [],
            'keywords': []
        }
        
        # 智能解析查询结构
        query_lower = query.lower()
        
        # 1. 提取公司信息（优先匹配ticker）
        for pattern in self.company_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match[0] else match[1]
                if match.strip() and len(match.strip()) > 1:
                    info['companies'].append(match.strip())
        
        # 2. 提取年份信息
        for pattern in self.year_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match[0] else match[1]
                if match.isdigit() and 1900 <= int(match) <= 2030:
                    info['years'].append(match)
        
        # 3. 提取指标代码
        for pattern in self.indicator_code_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            info['indicator_codes'].extend(matches)
        
        # 4. 智能识别指标名称
        indicators = self._extract_indicator_names(query)
        info['indicators'].extend(indicators)
        
        # 5. 识别ESG类别
        for category, terms in self.esg_terms.items():
            for term in terms:
                if term.lower() in query_lower:
                    info['esg_categories'].append(category)
                    info['keywords'].append(term)
        
        # 去重并过滤
        for key in ['companies', 'years', 'indicators', 'indicator_codes', 'values', 'esg_categories', 'keywords']:
            info[key] = list(set([item for item in info[key] if item and str(item).strip()]))
        
        return info
    
    def _extract_indicator_names(self, query: str) -> List[str]:
        """智能提取指标名称"""
        indicators = []
        query_lower = query.lower()
        
        # 环境指标
        env_patterns = [
            r'(nitrogen\s+oxide\s+emissions?)',
            r'(carbon\s+dioxide\s+emissions?)',
            r'(methane\s+emissions?)',
            r'(voc\s+emissions?)',
            r'(particulate\s+matter)',
            r'(water\s+emissions?)',
            r'(energy\s+consumption)',
            r'(renewable\s+energy)',
            r'(hazardous\s+waste)'
        ]
        
        # 社会指标
        social_patterns = [
            r'(women\s+workforce)',
            r'(pct\s+women\s+in\s+workforce)',
            r'(employee\s+diversity)',
            r'(workforce\s+diversity)',
            r'(safety\s+training)',
            r'(community\s+engagement)',
            r'(human\s+rights)',
            r'(labor\s+rights)'
        ]
        
        # 治理指标
        gov_patterns = [
            r'(board\s+diversity)',
            r'(executive\s+compensation)',
            r'(audit\s+quality)',
            r'(transparency)',
            r'(corporate\s+governance)',
            r'(risk\s+management)',
            r'(stakeholder\s+engagement)'
        ]
        
        all_patterns = env_patterns + social_patterns + gov_patterns
        
        for pattern in all_patterns:
            matches = re.findall(pattern, query_lower)
            indicators.extend(matches)
        
        return indicators
    
    def _identify_intent(self, query: str) -> str:
        """识别查询意图"""
        query_lower = query.lower()
        
        for intent, keywords in self.query_intents.items():
            for keyword in keywords:
                if keyword.lower() in query_lower:
                    return intent
        
        return 'general'
    
    def _generate_optimized_query(self, extracted_info: Dict[str, Any], intent: str) -> str:
        """生成优化查询"""
        query_parts = []
        
        # 添加ESG关键词
        if not extracted_info['esg_categories']:
            query_parts.append('ESG')
        
        # 添加公司信息（支持ticker和company name查询）
        if extracted_info['companies']:
            for company in extracted_info['companies']:
                company_clean = company.strip()
                # 特殊处理：A US Equity -> Agilent Technologies Inc
                if company_clean.lower() == 'a us equity':
                    query_parts.extend(['Agilent Technologies Inc', 'A US Equity'])
                elif company_clean.lower() == 'aa us equity':
                    query_parts.extend(['AA US Equity', 'Alcoa Corporation'])
                else:
                    query_parts.append(company_clean)
                    # 如果是ticker格式，添加公司全名
                    if 'us equity' in company_clean.lower():
                        ticker = company_clean.split()[0]
                        if ticker == 'AA':
                            query_parts.append('Alcoa Corporation')
                        elif ticker == 'A':
                            query_parts.append('Agilent Technologies Inc')
        
        # 添加年份信息
        if extracted_info['years']:
            query_parts.extend(extracted_info['years'])
        
        # 添加指标信息
        if extracted_info['indicators']:
            query_parts.extend(extracted_info['indicators'])
        
        # 添加指标代码
        if extracted_info['indicator_codes']:
            query_parts.extend(extracted_info['indicator_codes'])
        
        # 添加关键词
        if extracted_info['keywords']:
            query_parts.extend(extracted_info['keywords'])
        
        # 添加意图相关词汇
        if intent == 'trend':
            query_parts.append('趋势分析')
        elif intent == 'comparison':
            query_parts.append('比较分析')
        elif intent == 'specific':
            query_parts.append('具体数据')
        elif intent == 'overview':
            query_parts.append('概览')
        elif intent == 'analysis':
            query_parts.append('分析')
        
        # 生成最终查询
        optimized_query = ' '.join(query_parts)
        
        # 添加中英文混合优化
        if any(char in optimized_query for char in '公司年指标'):
            optimized_query += ' company year indicator'
        
        # 特殊处理G类和ES类指标
        if 'G类' in optimized_query or 'governance' in optimized_query.lower():
            optimized_query += ' governance corporate governance'
        if 'ES类' in optimized_query or 'ES指标' in optimized_query:
            optimized_query += ' environmental social ES indicators'
        
        return optimized_query
    
    def _calculate_confidence(self, extracted_info: Dict[str, Any]) -> float:
        """计算查询置信度"""
        confidence = 0.0
        
        # 基础分数
        if extracted_info['companies']:
            confidence += 0.3
        if extracted_info['years']:
            confidence += 0.2
        if extracted_info['indicators'] or extracted_info['indicator_codes']:
            confidence += 0.3
        if extracted_info['esg_categories']:
            confidence += 0.2
        
        return min(confidence, 1.0)

class FixedMemoRAG:
    """Fixed MemoRAG System"""
    
    def __init__(self, db_path: str, model_name: str = "BAAI/bge-m3", 
                 memory_size: int = 1000, llm_api_key: str = None, language: str = "en"):
        """Initialize Fixed MemoRAG System"""
        self.db_path = db_path
        self.model_name = model_name
        self.memory_size = memory_size
        self.language = language  # "en" for English, "zh" for Chinese
        
        # Initialize translation dictionary
        self.translations = self._init_translations()
        
        # Initialize Chroma client
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Load BGE model
        print(f"🔬 {self.t('loading_model')}: {model_name}")
        self.model = SentenceTransformer(model_name)
        print(f"✅ {self.t('model_loaded')}")
        
        # Check embedding dimensions
        test_embedding = self.model.encode(["test"])
        print(f"📏 {self.t('embedding_dimension')}: {len(test_embedding[0])}")
        
        # Check Chroma database embedding dimensions
        try:
            # Get all collections
            collections = self.client.list_collections()
            if collections:
                # Use first collection to check dimensions
                sample = self.client.get_collection(name=collections[0].name).get(limit=1, include=['embeddings'])
                if sample['embeddings']:
                    chroma_dim = len(sample['embeddings'][0])
                    print(f"📏 {self.t('chroma_dimension')}: {chroma_dim}")
                    if chroma_dim != len(test_embedding[0]):
                        print(f"⚠️ {self.t('dimension_mismatch')} BGE: {len(test_embedding[0])}, Chroma: {chroma_dim}")
                        print("💡 This may cause inaccurate similarity search")
                        print("🔧 Suggestion: Rebuild embeddings using the same model")
                    else:
                        print(f"✅ {self.t('dimension_match')}")
                else:
                    print("❌ Chroma database has no embedding data")
            else:
                print(f"❌ {self.t('no_collections')}")
        except Exception as e:
            print(f"❌ Failed to check Chroma embedding dimensions: {str(e)}")
        
        # Initialize LLM response generator
        if llm_api_key:
            self.llm_generator = LLMResponseGenerator(llm_api_key)
            print(f"🤖 {self.t('llm_initialized')}")
        else:
            self.llm_generator = None
            print(f"⚠️ {self.t('no_llm_config')}")
        
        # Initialize smart query processor
        self.query_processor = OptimizedQueryProcessor()
        print(f"🧠 {self.t('query_processor_ready')}")
        
        # Initialize memory system
        self.memory = {
            'queries': [],
            'results': [],
            'patterns': {},
            'insights': [],
            'trends': {},
            'last_update': None
        }
        
        # Get all collections
        self.collections = self.client.list_collections()
        print(f"📊 Found {len(self.collections)} {self.t('collections_found')}")
        
        # Display collection information
        if self.collections:
            print("📋 Available collections:")
            for i, collection in enumerate(self.collections, 1):
                try:
                    count = collection.count()
                    print(f"   {i}. {collection.name} (Records: {count})")
                except Exception as e:
                    print(f"   {i}. {collection.name} (Failed to get record count: {str(e)})")
        else:
            print(f"⚠️ {self.t('no_collections')}")
        
        # Load memory
        self.load_memory()
        
        # Context management
        self.last_query = None
        self.last_results = None
        
        # Answer mode management
        self.use_llm = llm_api_key is not None
        self.debug_mode = False
    
    def _init_translations(self) -> Dict[str, Dict[str, str]]:
        """Initialize translation dictionary"""
        return {
            # System messages
            "loading_model": {"en": "Loading BGE model", "zh": "正在加载BGE模型"},
            "model_loaded": {"en": "BGE model loaded successfully!", "zh": "BGE模型加载完成！"},
            "embedding_dimension": {"en": "BGE model embedding dimension", "zh": "BGE模型嵌入维度"},
            "chroma_dimension": {"en": "Chroma database embedding dimension", "zh": "Chroma数据库嵌入维度"},
            "dimension_match": {"en": "Embedding dimensions match", "zh": "嵌入维度匹配"},
            "dimension_mismatch": {"en": "Warning: Embedding dimension mismatch!", "zh": "警告：嵌入维度不匹配！"},
            "llm_initialized": {"en": "LLM response generator initialized!", "zh": "LLM回答生成器初始化完成！"},
            "no_llm_config": {"en": "No LLM API configured, using basic response mode", "zh": "未配置LLM API，将使用基础回答模式"},
            "query_processor_ready": {"en": "Smart query processor initialized!", "zh": "智能查询处理器初始化完成！"},
            "collections_found": {"en": "ESG data collections found", "zh": "个ESG数据集合"},
            "no_collections": {"en": "Warning: No data collections found", "zh": "警告: 没有找到任何数据集合"},
            "memory_loaded": {"en": "Memory loaded successfully", "zh": "加载记忆成功"},
            "new_memory": {"en": "Creating new memory bank", "zh": "创建新的记忆库"},
            
            # Query processing
            "auto_detected_collection": {"en": "Auto-detected collection", "zh": "自动检测到集合"},
            "no_collections_found": {"en": "No collections found", "zh": "没有找到任何集合"},
            "extracted_companies": {"en": "Extracted companies", "zh": "提取到的公司"},
            "no_company_info": {"en": "No company information extracted", "zh": "未提取到公司信息"},
            
            # Results display
            "query": {"en": "Query", "zh": "查询"},
            "smart_response": {"en": "Smart Response", "zh": "智能回答"},
            "data_summary": {"en": "Data Summary", "zh": "数据摘要"},
            "found_results": {"en": "relevant data found (sorted by quality)", "zh": "条相关数据（已按质量重排序）"},
            "no_results": {"en": "No relevant results found", "zh": "没有找到相关结果"},
            "more_data": {"en": "more data", "zh": "条数据"},
            
            # Commands
            "help": {"en": "help", "zh": "帮助"},
            "collections": {"en": "collections", "zh": "集合"},
            "memory": {"en": "memory", "zh": "记忆"},
            "clear": {"en": "clear", "zh": "清空"},
            "mode": {"en": "mode", "zh": "模式"},
            "debug": {"en": "debug", "zh": "调试"},
            "quit": {"en": "quit/exit", "zh": "退出"},
            
            # Interactive mode
            "welcome": {"en": "Welcome to Smart MemoRAG ESG System!", "zh": "欢迎使用智能回答版MemoRAG ESG系统！"},
            "enter_query": {"en": "Enter your ESG query, or type 'help' for help", "zh": "输入您的ESG查询，或输入 'help' 查看帮助"},
            "enter_query_prompt": {"en": "Please enter your query", "zh": "请输入您的查询"},
            "invalid_query": {"en": "Please enter a valid query", "zh": "请输入有效的查询"},
            "processing_error": {"en": "Error processing query", "zh": "处理查询时出错"},
            "try_again": {"en": "Please try entering the query again", "zh": "请尝试重新输入查询"},
            "goodbye": {"en": "Thank you for using! Goodbye!", "zh": "感谢使用！再见！"},
            
            # Help system
            "usage_help": {"en": "Usage Help", "zh": "使用帮助"},
            "query_examples": {"en": "Query Examples", "zh": "查询示例"},
            "commands": {"en": "Commands", "zh": "命令"},
            "tips": {"en": "Tips", "zh": "提示"},
            "supports_natural_language": {"en": "Supports natural language queries", "zh": "支持自然语言查询"},
            "supports_multilingual": {"en": "Supports multilingual queries", "zh": "支持中英文混合查询"},
            "supports_context": {"en": "Supports contextual queries", "zh": "支持上下文查询"},
            "generates_smart_responses": {"en": "Generates smart responses", "zh": "系统会生成智能回答"},
            "remembers_history": {"en": "Remembers your query history", "zh": "系统会记住您的查询历史"},
            "auto_detects_collections": {"en": "Auto-detects available data collections", "zh": "系统会自动检测可用的数据集合"},
            
            # Memory system
            "memory_empty": {"en": "Memory bank is empty", "zh": "记忆库为空"},
            "memory_report": {"en": "Memory Report", "zh": "记忆报告"},
            "total_queries": {"en": "Total queries", "zh": "总查询数"},
            "recent_queries": {"en": "Recent queries", "zh": "最近查询"},
            "popular_companies": {"en": "Popular companies", "zh": "热门公司"},
            "popular_years": {"en": "Popular years", "zh": "热门年份"},
            "pattern_count": {"en": "Pattern count", "zh": "模式数量"},
            "trend_count": {"en": "Trend count", "zh": "趋势数量"},
            "memory_cleared": {"en": "Memory cleared", "zh": "记忆已清空"},
            
            # Collections
            "available_collections": {"en": "Available Data Collections", "zh": "可用数据集合"},
            "collection_name": {"en": "Collection Name", "zh": "集合名称"},
            "record_count": {"en": "Record Count", "zh": "记录数量"},
            "sample_metadata": {"en": "Sample Metadata", "zh": "样本元数据"},
            "failed_to_get_info": {"en": "Failed to get information", "zh": "获取信息失败"},
            
            # Debug mode
            "debug_mode_on": {"en": "Debug mode enabled", "zh": "调试模式已开启"},
            "debug_mode_off": {"en": "Debug mode disabled", "zh": "调试模式已关闭"},
            "debug_info": {"en": "Debug Information", "zh": "调试信息"},
            "raw_query": {"en": "Raw Query", "zh": "原始查询"},
            "cleaned_query": {"en": "Cleaned Query", "zh": "清理后查询"},
            "optimized_query": {"en": "Optimized Query", "zh": "优化查询"},
            "extracted_info": {"en": "Extracted Information", "zh": "提取信息"},
            "query_intent": {"en": "Query Intent", "zh": "查询意图"},
            "confidence": {"en": "Confidence", "zh": "置信度"},
            "original_document": {"en": "Original Document", "zh": "原始文档"},
            "metadata": {"en": "Metadata", "zh": "元数据"},
            "extraction_result": {"en": "Extraction Result", "zh": "提取结果"},
            
            # LLM mode
            "llm_mode": {"en": "LLM Smart Response", "zh": "LLM智能回答"},
            "basic_mode": {"en": "Basic Response", "zh": "基础回答"},
            "switched_to_mode": {"en": "Switched to", "zh": "已切换到"},
            "no_llm_api": {"en": "No LLM API configured, cannot switch modes", "zh": "未配置LLM API，无法切换模式"},
            
            # Main function
            "system_title": {"en": "Smart MemoRAG + BGE-M3 ESG System", "zh": "智能回答版MemoRAG + BGE-M3 ESG系统"},
            "use_llm_prompt": {"en": "Use LLM to generate smart responses? (y/n)", "zh": "是否使用LLM生成智能回答？(y/n)"},
            "api_key_prompt": {"en": "Enter DeepSeek API Key", "zh": "请输入DeepSeek API Key"},
            "no_api_key": {"en": "No API Key provided, using basic response mode", "zh": "未提供API Key，将使用基础回答模式"},
            
            # Language selection
            "language_selection": {"en": "Language Selection", "zh": "语言选择"},
            "select_language": {"en": "Please select language / 请选择语言:", "zh": "请选择语言 / Please select language:"},
            "english_option": {"en": "1. English", "zh": "1. English"},
            "chinese_option": {"en": "2. Chinese (中文)", "zh": "2. Chinese (中文)"},
            "language_set": {"en": "Language set to", "zh": "语言已设置为"},
            
            # Search results display
            "searched_companies": {"en": "Searched companies", "zh": "搜索到的公司"},
            "unique_companies_found": {"en": "Unique companies found", "zh": "发现的唯一公司"},
            "similarity": {"en": "Similarity", "zh": "相似度"},
            "quality_score": {"en": "Quality score", "zh": "分数"},
            "more_data_available": {"en": "more data available", "zh": "条数据"},
        }
    
    def t(self, key: str) -> str:
        """Get translated text"""
        if key in self.translations:
            return self.translations[key].get(self.language, self.translations[key]["en"])
        return key
    
    def intelligent_query(self, raw_query: str, collection_name: str = None, 
                        n_results: int = 10, use_memory: bool = True) -> Dict[str, Any]:
        """Intelligent query (supports natural language)"""
        # Auto-detect collection name
        if collection_name is None:
            collections = self.client.list_collections()
            if collections:
                collection_name = collections[0].name
                print(f"🔍 {self.t('auto_detected_collection')}: {collection_name}")
            else:
                print(f"❌ {self.t('no_collections_found')}")
                return {
                    'raw_query': raw_query,
                    'query_analysis': {},
                    'results': [],
                    'insights': [self.t('no_collections_found')],
                    'smart_response': f"Sorry, no data collections found in the database.",
                    'timestamp': datetime.now().isoformat()
                }
        
        # 1. Process query (with context)
        context = {'previous_query': self.last_query} if self.last_query else None
        query_analysis = self.query_processor.process_query(raw_query, context)
        
        # Debug mode: display query analysis
        if self.debug_mode:
            print(f"\n🔍 {self.t('debug_info')}:")
            print(f"   {self.t('raw_query')}: {raw_query}")
            print(f"   {self.t('cleaned_query')}: {query_analysis['cleaned_query']}")
            print(f"   {self.t('optimized_query')}: {query_analysis['optimized_query']}")
            print(f"   {self.t('extracted_info')}: {query_analysis['extracted_info']}")
            print(f"   {self.t('query_intent')}: {query_analysis['intent']}")
            print(f"   {self.t('confidence')}: {query_analysis['confidence']:.2f}")
        
        # Always display extracted company info (for debugging)
        extracted_companies = query_analysis['extracted_info'].get('companies', [])
        if extracted_companies:
            print(f"\n🏢 {self.t('extracted_companies')}: {extracted_companies}")
        else:
            print(f"\n⚠️ {self.t('no_company_info')}")
        
        # 2. Update context
        self.last_query = raw_query
        
        # 3. Execute query
        results = self._execute_query_with_post_filter(query_analysis, collection_name, n_results)
        
        # 4. Generate insights
        insights = self._generate_insights(results, query_analysis)
        
        # 5. Generate smart response
        smart_response = self._generate_smart_response(raw_query, results, query_analysis)
        
        # 6. Add to memory
        self.add_to_memory(query_analysis['optimized_query'], results, insights)
        
        # 7. Update context results
        self.last_results = results
        
        return {
            'raw_query': raw_query,
            'query_analysis': query_analysis,
            'results': results,
            'insights': insights,
            'smart_response': smart_response,
            'timestamp': datetime.now().isoformat()
        }
    
    def _generate_smart_response(self, query: str, results: List[Dict], query_analysis: Dict) -> str:
        """Generate smart response"""
        if self.use_llm and self.llm_generator:
            # Use LLM to generate response
            context = f"Query intent: {query_analysis['intent']}, Confidence: {query_analysis['confidence']:.2f}"
            # Pass language setting to LLM generator
            self.llm_generator.language = self.language
            return self.llm_generator.generate_response(query, results, context)
        else:
            # Use basic response generation
            return self._generate_basic_response(query, results, query_analysis)
    
    def _generate_basic_response(self, query: str, results: List[Dict], query_analysis: Dict) -> str:
        """Generate basic response"""
        if not results:
            if self.language == 'zh':
                return "很抱歉，没有找到与您查询相关的数据。请尝试调整查询条件或检查数据是否存在。"
            else:
                return "Sorry, no data related to your query was found. Please try adjusting your query conditions or check if the data exists."
        
        # Analyze query intent
        intent = query_analysis['intent']
        extracted_info = query_analysis['extracted_info']
        
        response_parts = []
        
        # Opening
        if self.language == 'zh':
            if intent == 'trend':
                response_parts.append("根据趋势分析，")
            elif intent == 'comparison':
                response_parts.append("通过对比分析，")
            elif intent == 'specific':
                response_parts.append("具体数据显示，")
            else:
                response_parts.append("根据查询结果，")
        else:
            if intent == 'trend':
                response_parts.append("Based on trend analysis,")
            elif intent == 'comparison':
                response_parts.append("Through comparative analysis,")
            elif intent == 'specific':
                response_parts.append("Specific data shows,")
            else:
                response_parts.append("Based on query results,")
        
        # Data statistics
        companies = list(set([r['esg_info'].get('company', '') for r in results if r['esg_info'].get('company')]))
        years = list(set([r['esg_info'].get('year', '') for r in results if r['esg_info'].get('year')]))
        indicators = list(set([r['esg_info'].get('indicator', '') for r in results if r['esg_info'].get('indicator')]))
        
        if self.language == 'zh':
            response_parts.append(f"我找到了 {len(results)} 条相关数据。")
            
            if companies:
                response_parts.append(f"涉及的公司包括: {', '.join(companies[:3])}。")
            
            if years:
                response_parts.append(f"数据年份范围: {min(years)}-{max(years)}。")
            
            if indicators:
                response_parts.append(f"主要指标包括: {', '.join(indicators[:3])}。")
            
            # Trend analysis
            if len(results) > 1 and years:
                response_parts.append("从时间维度看，数据呈现一定的变化趋势。")
            
            # Suggestions
            response_parts.append("建议您查看具体数据详情以获取更准确的信息。")
        else:
            response_parts.append(f"I found {len(results)} relevant data records.")
            
            if companies:
                response_parts.append(f"Companies involved include: {', '.join(companies[:3])}.")
            
            if years:
                response_parts.append(f"Data year range: {min(years)}-{max(years)}.")
            
            if indicators:
                response_parts.append(f"Main indicators include: {', '.join(indicators[:3])}.")
            
            # Trend analysis
            if len(results) > 1 and years:
                response_parts.append("From a temporal perspective, the data shows certain trends.")
            
            # Suggestions
            response_parts.append("I recommend reviewing specific data details for more accurate information.")
        
        return " ".join(response_parts)
    
    def toggle_llm_mode(self):
        """Toggle LLM mode"""
        if self.llm_generator and self.llm_generator.api_key:
            self.use_llm = not self.use_llm
            mode = self.t('llm_mode') if self.use_llm else self.t('basic_mode')
            print(f"🔄 {self.t('switched_to_mode')} {mode}")
        else:
            print(f"❌ {self.t('no_llm_api')}")
    
    def _execute_query_with_post_filter(self, query_analysis: Dict, collection_name: str, n_results: int) -> List[Dict[str, Any]]:
        """执行查询并进行后过滤"""
        try:
            collection = self.client.get_collection(name=collection_name)
            query_embedding = self.model.encode([query_analysis['optimized_query']])[0].tolist()
            
            # 先执行基础查询，获取更多结果
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(50, n_results * 5),  # 获取更多结果用于过滤
                include=['documents', 'metadatas', 'distances']
            )
            
            # 处理结果
            processed_results = []
            for i in range(len(results['documents'][0])):
                document = results['documents'][0][i]
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i]
                similarity = 1 - distance
                
                esg_info = self.extract_esg_info(document)
                
                result = {
                    'document': document,
                    'metadata': metadata,
                    'distance': distance,
                    'similarity': similarity,
                    'esg_info': esg_info
                }
                
                processed_results.append(result)
            
            # Debug: display all searched companies
            print(f"\n🔍 {self.t('searched_companies')} ({len(processed_results)} records):")
            companies_found = set()
            for i, r in enumerate(processed_results[:10]):
                esg_info = r['esg_info']
                metadata = r['metadata']
                company = esg_info.get('company', metadata.get('company', 'N/A'))
                companies_found.add(company)
                print(f"   {i+1}. {company} ({self.t('similarity')}: {r['similarity']:.3f})")
                
                # Debug: display original document and metadata
                if self.debug_mode:
                    print(f"      📄 {self.t('original_document')}: {r['document'][:100]}...")
                    print(f"      📋 {self.t('metadata')}: {metadata}")
                    print(f"      🔍 {self.t('extraction_result')}: {esg_info}")
                    print()
            
            print(f"\n📊 {self.t('unique_companies_found')}: {list(companies_found)}")
            
            # 后过滤
            filtered_results = self._post_filter_results(processed_results, query_analysis)
            
            # 按相似度排序
            filtered_results.sort(key=lambda x: x['similarity'], reverse=True)
            
            return filtered_results[:n_results]
            
        except Exception as e:
            print(f"Query error: {str(e)}")
            return []
    
    def _post_filter_results(self, results: List[Dict], query_analysis: Dict) -> List[Dict]:
        """后过滤结果并进行智能重排序"""
        extracted_info = query_analysis['extracted_info']
        
        # 如果没有提取到任何过滤条件，直接返回原始结果
        if not extracted_info['years'] and not extracted_info['companies'] and not extracted_info['indicator_codes']:
            return self._rerank_results(results, query_analysis)
        
        # 计算每个结果的匹配分数
        scored_results = []
        for result in results:
            score = self._calculate_match_score(result, extracted_info)
            if score > 0:  # 只保留有匹配的结果
                result['match_score'] = score
                scored_results.append(result)
        
        # 如果没有匹配的结果，返回原始结果
        if not scored_results:
            return self._rerank_results(results, query_analysis)
        
        # 智能重排序
        reranked_results = self._rerank_results(scored_results, query_analysis)
        
        return reranked_results
    
    def _calculate_match_score(self, result: Dict, extracted_info: Dict) -> int:
        """计算匹配分数"""
        metadata = result['metadata']
        esg_info = result['esg_info']
        score = 0
        
        # 年份匹配分数
        if extracted_info['years']:
            for year in extracted_info['years']:
                if metadata.get('year') and str(metadata.get('year')) == str(year):
                    score += 10
                elif esg_info.get('year') and str(esg_info.get('year')) == str(year):
                    score += 10
        
        # 公司匹配分数（更灵活的匹配）
        if extracted_info['companies']:
            for company in extracted_info['companies']:
                # 检查metadata中的company字段
                if metadata.get('company'):
                    metadata_company = str(metadata.get('company')).lower()
                    if company.lower() in metadata_company:
                        score += 10
                    # 特殊处理：如果查询的是"A US Equity"，也匹配包含"Agilent"的记录
                    elif company.lower() == 'a us equity' and 'agilent' in metadata_company:
                        score += 10
                
                # 检查esg_info中的company字段
                if esg_info.get('company'):
                    esg_company = str(esg_info.get('company')).lower()
                    if company.lower() in esg_company:
                        score += 10
                    # 特殊处理：如果查询的是"A US Equity"，也匹配包含"Agilent"的记录
                    elif company.lower() == 'a us equity' and 'agilent' in esg_company:
                        score += 10
        
        # 指标代码匹配分数
        if extracted_info['indicator_codes']:
            for code in extracted_info['indicator_codes']:
                if metadata.get('field_code') and code.upper() in str(metadata.get('field_code')).upper():
                    score += 10
                elif esg_info.get('code') and code.upper() in str(esg_info.get('code')).upper():
                    score += 10
        
        return score
    
    def _rerank_results(self, results: List[Dict], query_analysis: Dict) -> List[Dict]:
        """智能重排序结果"""
        extracted_info = query_analysis['extracted_info']
        
        # 为每个结果计算重排序分数
        for result in results:
            score = 0
            esg_info = result['esg_info']
            metadata = result['metadata']
            
            # 1. 数据完整性分数（最重要）
            value = esg_info.get('value', '')
            if value and str(value).lower() != 'nan' and str(value).strip():
                score += 100  # 有效数据高分
            else:
                score -= 20  # 无效数据轻微扣分（不要完全排除）
            
            # 2. 匹配分数（如果有的话）
            if 'match_score' in result:
                score += result['match_score'] * 5
            
            # 3. 年份匹配分数
            if extracted_info['years']:
                for year in extracted_info['years']:
                    if metadata.get('year') and str(metadata.get('year')) == str(year):
                        score += 50
                    elif esg_info.get('year') and str(esg_info.get('year')) == str(year):
                        score += 50
            
            # 4. 公司匹配分数（更灵活的匹配）
            if extracted_info['companies']:
                for company in extracted_info['companies']:
                    # 检查metadata中的company字段
                    if metadata.get('company'):
                        metadata_company = str(metadata.get('company')).lower()
                        if company.lower() in metadata_company:
                            score += 30
                        # 特殊处理：如果查询的是"A US Equity"，也匹配包含"Agilent"的记录
                        elif company.lower() == 'a us equity' and 'agilent' in metadata_company:
                            score += 30
                    
                    # 检查esg_info中的company字段
                    if esg_info.get('company'):
                        esg_company = str(esg_info.get('company')).lower()
                        if company.lower() in esg_company:
                            score += 30
                        # 特殊处理：如果查询的是"A US Equity"，也匹配包含"Agilent"的记录
                        elif company.lower() == 'a us equity' and 'agilent' in esg_company:
                            score += 30
            
            # 5. 指标代码匹配分数
            if extracted_info['indicator_codes']:
                for code in extracted_info['indicator_codes']:
                    if metadata.get('field_code') and code.upper() in str(metadata.get('field_code')).upper():
                        score += 20
                    elif esg_info.get('code') and code.upper() in str(esg_info.get('code')).upper():
                        score += 20
            
            # 6. 相似度分数
            score += result.get('similarity', 0) * 10
            
            # 7. ESG类别匹配分数
            if extracted_info['esg_categories']:
                score += 10
            
            result['rerank_score'] = score
        
        # 按重排序分数排序
        results.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        return results
    
    def extract_esg_info(self, document: str) -> Dict[str, str]:
        """从文档中提取ESG信息"""
        info = {}
        
        try:
            # Debug: print original document
            if hasattr(self, 'debug_mode') and self.debug_mode:
                print(f"🔍 {self.t('original_document')}: {document[:200]}...")
            
            # Extract company name - support multiple formats
            company_patterns = [
                r'([^（]+)（',  # Chinese format: company name（
                r'([A-Za-z\s]+(?:Inc|Corp|Ltd|Company|Technologies|Systems|Group|Holdings))',  # English format
                r'([A-Z]{1,6}\s+US\s+Equity)',  # Ticker format
            ]
            
            for pattern in company_patterns:
                company_match = re.search(pattern, document)
                if company_match:
                    info['company'] = company_match.group(1).strip()
                    break
            
            # Extract year - support multiple formats
            year_patterns = [
                r'在(\d{4})年',  # Chinese format: 在2006年
                r'(\d{4})',      # Simple number format
                r'year\s+(\d{4})',  # year 2006
                r'in\s+(\d{4})',   # in 2006
            ]
            
            for pattern in year_patterns:
                year_match = re.search(pattern, document, re.IGNORECASE)
                if year_match:
                    info['year'] = year_match.group(1)
                    break
            
            # 提取指标名 - 支持多种格式
            indicator_patterns = [
                r'：([^（]+)（',  # 中文格式：指标名（
                r'([A-Za-z\s]+(?:Emissions|Consumption|Policy|Rights|Workforce|Diversity|Governance))',  # 英文指标
                r'(nitrogen\s+oxide\s+emissions?)',  # 具体指标
                r'(carbon\s+dioxide\s+emissions?)',
                r'(methane\s+emissions?)',
                r'(voc\s+emissions?)',
                r'(women\s+workforce)',
                r'(pct\s+women\s+in\s+workforce)',
            ]
            
            for pattern in indicator_patterns:
                indicator_match = re.search(pattern, document, re.IGNORECASE)
                if indicator_match:
                    info['indicator'] = indicator_match.group(1).strip()
                    break
            
            # 提取指标代码
            code_patterns = [
                r'code=([^）]+)',  # code=ES001
                r'(ES\d{3})',      # ES001
                r'(ES\d{2})',     # ES01
            ]
            
            for pattern in code_patterns:
                code_match = re.search(pattern, document, re.IGNORECASE)
                if code_match:
                    info['code'] = code_match.group(1).strip()
                    break
            
            # 提取数值 - 支持多种格式
            value_patterns = [
                r'= ([^,]+)',     # = 42.6
                r':\s*([^,\s]+)', # : 42.6
                r'(\d+\.?\d*)',   # 简单数字
                r'(True|False)',  # 布尔值
            ]
            
            for pattern in value_patterns:
                value_match = re.search(pattern, document)
                if value_match:
                    info['value'] = value_match.group(1).strip()
                    break
            
            # Debug: print extraction results
            if hasattr(self, 'debug_mode') and self.debug_mode:
                print(f"🔍 {self.t('extraction_result')}: {info}")
            
        except Exception as e:
            print(f"Error extracting information: {str(e)}")
        
        return info
    
    def _generate_insights(self, results: List[Dict], query_analysis: Dict) -> List[str]:
        """生成洞察"""
        insights = []
        
        if not results:
            return ["没有找到相关数据"]
        
        # 基于相似度分析
        high_similarity_count = len([r for r in results if r['similarity'] > 0.8])
        if high_similarity_count > 0:
            insights.append(f"发现 {high_similarity_count} 条高相关性数据（相似度>0.8）")
        
        # 基于数据完整性分析
        complete_data = [r for r in results if not r['metadata'].get('incomplete', False)]
        if complete_data:
            insights.append(f"数据完整性良好，{len(complete_data)} 条记录完整")
        
        # 基于公司分析
        companies = [r['esg_info'].get('company', '') for r in results if r['esg_info'].get('company')]
        if companies:
            unique_companies = list(set(companies))
            insights.append(f"涉及 {len(unique_companies)} 家公司: {', '.join(unique_companies[:3])}")
        
        # 基于年份分析
        years = [r['esg_info'].get('year') for r in results if r['esg_info'].get('year')]
        if years:
            unique_years = sorted(list(set(years)))
            insights.append(f"数据年份范围: {min(unique_years)}-{max(unique_years)}")
        
        return insights
    
    def add_to_memory(self, query: str, results: List[Dict], insights: List[str] = None):
        """添加查询到记忆系统"""
        memory_entry = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'result_count': len(results),
            'top_similarity': max([r.get('similarity', 0) for r in results]) if results else 0,
            'insights': insights or [],
            'companies': list(set([r.get('metadata', {}).get('company', '') for r in results])),
            'years': list(set([r.get('metadata', {}).get('year') for r in results if r.get('metadata', {}).get('year')])),
            'indicators': list(set([r.get('metadata', {}).get('field_name', '') for r in results]))
        }
        
        self.memory['queries'].append(memory_entry)
        
        # 保持记忆大小
        if len(self.memory['queries']) > self.memory_size:
            self.memory['queries'] = self.memory['queries'][-self.memory_size:]
        
        # 保存记忆
        self.save_memory()
    
    def save_memory(self):
        """保存记忆到文件"""
        memory_file = os.path.join(self.db_path, 'memorag_memory.pkl')
        try:
            with open(memory_file, 'wb') as f:
                pickle.dump(self.memory, f)
        except Exception as e:
            print(f"Failed to save memory: {str(e)}")
    
    def load_memory(self):
        """从文件加载记忆"""
        memory_file = os.path.join(self.db_path, 'memorag_memory.pkl')
        if os.path.exists(memory_file):
            try:
                with open(memory_file, 'rb') as f:
                    self.memory = pickle.load(f)
                print(f"✅ {self.t('memory_loaded')}, contains {len(self.memory['queries'])} historical queries")
            except Exception as e:
                print(f"Failed to load memory: {str(e)}")
        else:
            print(f"📝 {self.t('new_memory')}")
    
    def display_results(self, result: Dict[str, Any]):
        """Display query results (smart response version)"""
        print(f"\n🔍 {self.t('query')}: {result['raw_query']}")
        print("=" * 60)
        
        # Display smart response
        print(f"🤖 {self.t('smart_response')}:")
        print(result['smart_response'])
        
        # Display data summary
        results = result['results']
        if results:
            print(f"\n📊 {self.t('data_summary')}:")
            print(f"   Found {len(results)} {self.t('found_results')}")
            
            # Display top 5 key data (including rerank score)
            for i, r in enumerate(results[:5], 1):
                esg_info = r['esg_info']
                rerank_score = r.get('rerank_score', 0)
                value = esg_info.get('value', 'N/A')
                value_status = "✅" if value and str(value).lower() != 'nan' and str(value).strip() else "❌"
                print(f"   {i}. {value_status} {esg_info.get('company', 'N/A')} ({esg_info.get('year', 'N/A')}) - {esg_info.get('indicator', 'N/A')}: {value} [{self.t('quality_score')}:{rerank_score:.1f}]")
            
            if len(results) > 5:
                print(f"   ... {len(results) - 5} {self.t('more_data_available')}")
        else:
            print(f"❌ {self.t('no_results')}")
    
    def show_help(self):
        """Display help information"""
        print(f"\n📖 {self.t('usage_help')}:")
        print("=" * 50)
        print(f"💡 {self.t('query_examples')}:")
        print("   • A US Equity 2015 Pct Women in Workforce indicator")
        print("   • Agilent Technologies Inc 2015 women workforce percentage")
        print("   • ES047 indicator data")
        print("   • 2015 environmental emissions trend")
        print("   • How about this trend? (contextual query)")
        print(f"\n🔧 {self.t('commands')}:")
        print(f"   • {self.t('help')} - Display help information")
        print(f"   • {self.t('collections')} - Display available data collections")
        print(f"   • {self.t('memory')} - Display memory report")
        print(f"   • {self.t('clear')} - Clear memory")
        print(f"   • {self.t('mode')} - Toggle response mode (LLM/Basic)")
        print(f"   • {self.t('debug')} - Enable/disable debug mode")
        print(f"   • {self.t('quit')} - Exit system")
        print(f"\n💭 {self.t('tips')}:")
        print(f"   • {self.t('supports_natural_language')}")
        print(f"   • {self.t('supports_multilingual')}")
        print(f"   • {self.t('supports_context')}")
        print(f"   • {self.t('generates_smart_responses')}")
        print(f"   • {self.t('remembers_history')}")
        print(f"   • {self.t('auto_detects_collections')}")
    
    def show_collections(self):
        """Display available data collections"""
        print(f"\n📊 {self.t('available_collections')}:")
        print("=" * 50)
        
        if not self.collections:
            print(f"❌ {self.t('no_collections_found')}")
            return
        
        for i, collection in enumerate(self.collections, 1):
            try:
                count = collection.count()
                print(f"{i}. {self.t('collection_name')}: {collection.name}")
                print(f"   📈 {self.t('record_count')}: {count}")
                
                # Get sample data
                sample = collection.get(limit=1, include=['metadatas'])
                if sample['metadatas']:
                    metadata = sample['metadatas'][0]
                    print(f"   📋 {self.t('sample_metadata')}: {metadata}")
                
                print()
                
            except Exception as e:
                print(f"{i}. {self.t('collection_name')}: {collection.name}")
                print(f"   ⚠️ {self.t('failed_to_get_info')}: {str(e)}")
                print()
    
    def show_memory_report(self):
        """Display memory report"""
        if not self.memory['queries']:
            print(f"\n📝 {self.t('memory_empty')}")
            return
        
        # Statistics
        total_queries = len(self.memory['queries'])
        recent_queries = [q for q in self.memory['queries'] if 
                         (datetime.now() - datetime.fromisoformat(q['timestamp'])).days < 7]
        
        # Popular companies
        all_companies = []
        for query in self.memory['queries']:
            all_companies.extend(query['companies'])
        company_counts = pd.Series(all_companies).value_counts()
        
        # Popular years
        all_years = []
        for query in self.memory['queries']:
            all_years.extend(query['years'])
        year_counts = pd.Series(all_years).value_counts()
        
        print(f"\n📊 {self.t('memory_report')}:")
        print("=" * 50)
        print(f"{self.t('total_queries')}: {total_queries}")
        print(f"{self.t('recent_queries')}: {len(recent_queries)}")
        print(f"{self.t('popular_companies')}: {', '.join(company_counts.head(5).index.tolist())}")
        print(f"{self.t('popular_years')}: {', '.join(map(str, year_counts.head(5).index.tolist()))}")
        print(f"{self.t('pattern_count')}: {len(self.memory['patterns'])}")
        print(f"{self.t('trend_count')}: {len(self.memory['trends'])}")
    
    def clear_memory(self):
        """Clear memory"""
        self.memory = {
            'queries': [],
            'results': [],
            'patterns': {},
            'insights': [],
            'trends': {},
            'last_update': None
        }
        self.save_memory()
        print(f"\n🗑️ {self.t('memory_cleared')}")
    
    def interactive_mode(self):
        """Interactive mode"""
        print(f"\n🚀 {self.t('welcome')}")
        print("=" * 60)
        print(f"💡 {self.t('enter_query')}")
        print(f"💡 {self.t('quit')} to exit system")
        print("=" * 60)
        
        while True:
            try:
                # Get user input
                user_input = input(f"\n🔍 {self.t('enter_query_prompt')}: ").strip()
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', '退出']:
                    print(f"\n👋 {self.t('goodbye')}")
                    break
                elif user_input.lower() in ['help', '帮助']:
                    self.show_help()
                    continue
                elif user_input.lower() in ['collections', '集合']:
                    self.show_collections()
                    continue
                elif user_input.lower() in ['memory', '记忆']:
                    self.show_memory_report()
                    continue
                elif user_input.lower() in ['clear', '清空']:
                    self.clear_memory()
                    continue
                elif user_input.lower() in ['mode', '模式']:
                    self.toggle_llm_mode()
                    continue
                elif user_input.lower() in ['debug', '调试']:
                    self.debug_mode = not self.debug_mode
                    status = self.t('debug_mode_on') if self.debug_mode else self.t('debug_mode_off')
                    print(f"🔍 {status}")
                    continue
                elif not user_input:
                    print(f"❌ {self.t('invalid_query')}")
                    continue
                
                # Execute query
                result = self.intelligent_query(user_input)
                self.display_results(result)
                
            except KeyboardInterrupt:
                print(f"\n\n👋 {self.t('goodbye')}")
                break
            except Exception as e:
                print(f"\n❌ {self.t('processing_error')}: {str(e)}")
                print(f"💡 {self.t('try_again')}")

def main():
    """Main function - Smart Interactive MemoRAG"""
    
    # Use relative path to ensure code can run in different environments
    db_path = "."  # Current directory
    
    print("🚀 Smart MemoRAG + BGE-M3 ESG System")
    print("=" * 60)
    
    # Language selection
    print("\n🌐 Language Selection / 语言选择")
    print("=" * 40)
    print("1. English")
    print("2. Chinese (中文)")
    
    while True:
        lang_choice = input("\nPlease select language / 请选择语言 (1/2): ").strip()
        if lang_choice == "1":
            language = "en"
            break
        elif lang_choice == "2":
            language = "zh"
            break
        else:
            print("Invalid choice. Please enter 1 or 2. / 无效选择，请输入1或2。")
    
    print(f"\n✅ Language set to {'English' if language == 'en' else 'Chinese (中文)'}")
    print("=" * 60)
    
    # Ask whether to use LLM
    use_llm = input("🤖 Use LLM to generate smart responses? (y/n): ").strip().lower()
    llm_api_key = None
    
    if use_llm == 'y':
        llm_api_key = input("🔑 Enter DeepSeek API Key: ").strip()
        if not llm_api_key:
            print("⚠️ No API Key provided, using basic response mode")
            llm_api_key = None
    
    # Initialize system
    memorag = FixedMemoRAG(db_path, llm_api_key=llm_api_key, language=language)
    
    # Start interactive mode
    memorag.interactive_mode()

if __name__ == "__main__":
    main()
