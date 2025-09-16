"""
RAGAS Evaluation Script for MemoRAG-Engine
Professional RAG system evaluation based on official documentation
Reference: https://docs.ragas.io/en/stable/getstarted/rag_eval/
"""

import pandas as pd
import numpy as np
from typing import List, Dict
import json
from datetime import datetime

try:
    import ragas
    from ragas import evaluate, EvaluationDataset
    from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
    from ragas.llms import LangchainLLMWrapper
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

class MemoRAGEvaluator:
    """MemoRAG系统评估器"""
    
    def __init__(self, rag_system):
        """
        初始化评估器
        
        Args:
            rag_system: MemoRAG系统实例
        """
        self.rag_system = rag_system
        self.evaluation_results = []
        
    def create_test_dataset(self) -> List[Dict]:
        """创建ESG测试数据集"""
        test_questions = [
            {
                "question": "What were Alcoa Corp's nitrogen oxide emissions in 2007?",
                "ground_truth": "Alcoa Corp reported nitrogen oxide emissions of 32.8 kilotons in 2007.",
                "contexts": ["Alcoa Corp 2007 NOx emissions data"]
            },
            {
                "question": "Tell me about Alcoa Corp's carbon dioxide emissions performance in 2010",
                "ground_truth": "Alcoa Corp's carbon dioxide emissions in 2010 were 29.5 units, showing a 13.5% increase from 2009.",
                "contexts": ["Alcoa Corp 2010 CO2 emissions data"]
            },
            {
                "question": "How did Agilent Technologies Inc perform in women workforce percentage in 2015?",
                "ground_truth": "Agilent Technologies Inc had specific women workforce percentage data for 2015.",
                "contexts": ["Agilent Technologies Inc 2015 workforce data"]
            },
            {
                "question": "What is the trend of environmental emissions for industrial companies?",
                "ground_truth": "Environmental emissions trends vary by company and year, with some companies showing reduction efforts.",
                "contexts": ["Environmental emissions trend data"]
            },
            {
                "question": "Compare Alcoa Corp emissions between 2007 and 2010",
                "ground_truth": "Alcoa Corp's emissions changed between 2007 and 2010, with specific values for each year.",
                "contexts": ["Alcoa Corp emissions comparison data"]
            }
        ]
        
        return test_questions
    
    def evaluate_single_query(self, question: str, ground_truth: str) -> Dict:
        """评估单个查询"""
        try:
            result = self.rag_system.process_query(question)
            
            answer = result.get('answer', '')
            contexts = result.get('contexts', [])
            retrieved_docs = result.get('retrieved_docs', [])
            
            evaluation_result = {
                'question': question,
                'ground_truth': ground_truth,
                'answer': answer,
                'contexts': contexts,
                'retrieved_docs': retrieved_docs,
                'timestamp': datetime.now().isoformat()
            }
            
            return evaluation_result
            
        except Exception as e:
            return {
                'question': question,
                'ground_truth': ground_truth,
                'answer': '',
                'contexts': [],
                'retrieved_docs': [],
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def run_ragas_evaluation(self, test_data: List[Dict]) -> Dict:
        """运行RAGAS评估"""
        if not RAGAS_AVAILABLE:
            return {"error": "RAGAS not available. Install with: pip install ragas"}
        
        try:
            ragas_data = []
            for item in test_data:
                result = self.evaluate_single_query(item['question'], item['ground_truth'])
                
                ragas_item = {
                    'user_input': item['question'],
                    'retrieved_contexts': result['contexts'],
                    'response': result['answer'],
                    'reference': item['ground_truth']
                }
                ragas_data.append(ragas_item)
            
            evaluation_dataset = EvaluationDataset.from_list(ragas_data)
            
            result = evaluate(
                dataset=evaluation_dataset,
                metrics=[
                    LLMContextRecall(),
                    Faithfulness(), 
                    FactualCorrectness()
                ]
            )
            
            return result
            
        except Exception as e:
            return {"error": f"RAGAS evaluation failed: {str(e)}"}
    
    def manual_evaluation(self, test_data: List[Dict]) -> Dict:
        """手动评估（当RAGAS不可用时）"""
        results = []
        
        for item in test_data:
            result = self.evaluate_single_query(item['question'], item['ground_truth'])
            results.append(result)
        
        total_questions = len(results)
        successful_answers = len([r for r in results if r.get('answer', '') != ''])
        error_count = len([r for r in results if 'error' in r])
        
        evaluation_summary = {
            'total_questions': total_questions,
            'successful_answers': successful_answers,
            'success_rate': successful_answers / total_questions if total_questions > 0 else 0,
            'error_count': error_count,
            'error_rate': error_count / total_questions if total_questions > 0 else 0,
            'detailed_results': results
        }
        
        return evaluation_summary
    
    def save_evaluation_results(self, results: Dict, filename: str = None):
        """保存评估结果"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ragas_evaluation_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"Evaluation results saved to: {filename}")
    
    def run_full_evaluation(self):
        """运行完整评估"""
        print("🚀 Starting MemoRAG-Engine Evaluation...")
        
        test_data = self.create_test_dataset()
        print(f"📊 Created {len(test_data)} test questions")
        
        if RAGAS_AVAILABLE:
            print("🔍 Running RAGAS evaluation...")
            ragas_results = self.run_ragas_evaluation(test_data)
            
            if 'error' not in ragas_results:
                print("✅ RAGAS evaluation completed successfully!")
                print(f"📈 RAGAS Results: {ragas_results}")
                self.save_evaluation_results(ragas_results, "ragas_evaluation_results.json")
            else:
                print(f"❌ RAGAS evaluation failed: {ragas_results['error']}")
        
        print("🔍 Running manual evaluation...")
        manual_results = self.manual_evaluation(test_data)
        print("✅ Manual evaluation completed!")
        print(f"📈 Manual Results: {manual_results}")
        
        self.save_evaluation_results(manual_results, "manual_evaluation_results.json")
        
        return manual_results

def main():
    """主函数"""
    print("MemoRAG-Engine RAGAS Evaluation")
    print("=" * 50)
    
    print("\n📋 To use this evaluation script:")
    print("1. Install RAGAS: pip install ragas")
    print("2. Import your RAG system")
    print("3. Create evaluator instance")
    print("4. Run evaluation")
    
    print("\n🎯 RAGAS Metrics Explained:")
    print("- Context Recall: How well retrieved context covers ground truth")
    print("- Faithfulness: How faithful is the response to retrieved context")
    print("- Factual Correctness: How factually correct is the response")

if __name__ == "__main__":
    main()