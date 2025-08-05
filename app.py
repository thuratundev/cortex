import psycopg2
import re
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from typing import Dict, List, Tuple, Any
import json
import openai  # or use any other LLM API

class BurmeseERPRAGSystem:
    def __init__(self, db_config: Dict, openai_api_key: str = None):
        """
        Initialize the Burmese ERP RAG System
        
        Args:
            db_config: PostgreSQL connection configuration
            openai_api_key: OpenAI API key for LLM responses
        """
        self.db_config = db_config
        self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        if openai_api_key:
            openai.api_key = openai_api_key
            
        # Initialize components
        self.intent_classifier = BurmeseIntentClassifier()
        self.entity_extractor = BurmeseEntityExtractor()
        self.sql_generator = BurmeseSQLGenerator()
        self.response_generator = BurmeseResponseGenerator()
        
        # Database schema knowledge
        self.schema_knowledge = self._load_schema_knowledge()
    
    def _load_schema_knowledge(self) -> Dict:
        """Load database schema knowledge with Burmese mappings"""
        return {
            "tables": {
                "sales": {
                    "burmese_names": ["အရောင်း", "ရောင်းချမှု", "အရောင်းဇယား"],
                    "description": "ရောင်းချမှုလုပ်ငန်းများ၏ဒေတာများ",
                    "columns": {
                        "sale_id": {"burmese": "အရောင်းကုဒ်", "type": "SERIAL"},
                        "product_id": {"burmese": "ပစ္စည်းကုဒ်", "type": "INTEGER"},
                        "product_name": {"burmese": "ပစ္စည်းနာမည်", "type": "VARCHAR"},
                        "quantity": {"burmese": "အရေအတွက်", "type": "INTEGER"},
                        "unit_price": {"burmese": "တစ်ခုချင်းစျေးနှုန်း", "type": "DECIMAL"},
                        "total_amount": {"burmese": "စုစုပေါင်းပမာဏ", "type": "DECIMAL"},
                        "sale_date": {"burmese": "ရောင်းချသည့်ရက်စွဲ", "type": "DATE"},
                        "customer_id": {"burmese": "ဝယ်သူကုဒ်", "type": "INTEGER"}
                    }
                },
                "products": {
                    "burmese_names": ["ပစ္စည်း", "ကုန်ပစ္စည်း", "ပစ္စည်းများ"],
                    "description": "ပစ္စည်းများ၏အချက်အလက်",
                    "columns": {
                        "product_id": {"burmese": "ပစ္စည်းကုဒ်", "type": "SERIAL"},
                        "product_name": {"burmese": "ပစ္စည်းနာမည်", "type": "VARCHAR"},
                        "category": {"burmese": "အမျိုးအစား", "type": "VARCHAR"},
                        "cost_price": {"burmese": "ဝယ်ဈေး", "type": "DECIMAL"},
                        "selling_price": {"burmese": "ရောင်းဈေး", "type": "DECIMAL"}
                    }
                },
                "expenses": {
                    "burmese_names": ["ကုန်ကျစရိတ်", "အသုံးစရိတ်", "ကုန်ကျမှု"],
                    "description": "ကုန်ကျစရိတ်နှင့်အသုံးစရิတ်ဒေတာများ",
                    "columns": {
                        "expense_id": {"burmese": "စရိတ်ကုဒ်", "type": "SERIAL"},
                        "expense_type": {"burmese": "စရိတ်အမျိုးအစား", "type": "VARCHAR"},
                        "amount": {"burmese": "ပမာဏ", "type": "DECIMAL"},
                        "expense_date": {"burmese": "စရိတ်ရက်စွဲ", "type": "DATE"},
                        "description": {"burmese": "ဖော်ပြချက်", "type": "TEXT"}
                    }
                }
            }
        }
    
    def process_query(self, burmese_query: str) -> str:
        """
        Main method to process Burmese queries
        
        Args:
            burmese_query: User query in Burmese
            
        Returns:
            Generated response in Burmese
        """
        try:
            # Step 1: Classify intent and extract entities
            intent, confidence = self.intent_classifier.classify_intent(burmese_query)
            
            if confidence < 0.5:
                return "စာလုံးရေးသားမှုကို နားမလည်ပါ။ နောက်တစ်ကြိမ် ရှင်းရှင်းလင်းလင်း ပြန်မေးပေးပါ။"
            
            # Step 2: Extract entities (dates, product names, etc.)
            entities = self.entity_extractor.extract_entities(burmese_query)
            
            # Step 3: Generate SQL query
            sql_query = self.sql_generator.generate_sql(intent, entities, self.schema_knowledge)
            
            if not sql_query:
                return "ဒေတာဘေ့စ်မှ အချက်အလက်ရှာဖွေရန် မအောင်မြင်ပါ။"
            
            
            # # Step 4: Execute query and get data
            # data = self._execute_query(sql_query)
            
            # # Step 5: Generate natural language response
            # response = self.response_generator.generate_response(
            #     burmese_query, data, intent, entities
            # )
        
            # return response
            return sql_query
            
        except Exception as e:
            return f"အမှားအယွင်းတစ်ခုဖြစ်ပွားခဲ့သည်: {str(e)}"
    
    def _execute_query(self, sql_query: str) -> List[Dict]:
        """Execute SQL query against PostgreSQL database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute(sql_query)
            columns = [desc[0] for desc in cursor.description]
            data = []
            
            for row in cursor.fetchall():
                data.append(dict(zip(columns, row)))
            
            cursor.close()
            conn.close()
            
            return data
            
        except Exception as e:
            print(f"Database error: {e}")
            return []


class BurmeseIntentClassifier:
    def __init__(self):
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # Intent templates in Burmese
        self.intent_templates = {
            "top_selling_products": [
                "အရောင်းရဆုံး ပစ္စည်း",
                "ရောင်းချမှုကောင်းဆုံး ပစ္စည်း",
                "အများဆုံးရောင်းချ ပစ္စည်း",
                "ရောင်းအားကောင်းဆုံး ပစ္စည်း"
            ],
            "lowest_selling_products": [
                "အရောင်းနည်းဆုံး ပစ္စည်း",
                "ရောင်းချမှုနည်းဆုံး ပစ္စည်း",
                "အနည်းဆုံးရောင်းချ ပစ္စည်း",
                "ရောင်းအားညံ့ဆုံး ပစ္စည်း"
            ],
            "profit_loss_report": [
                "အမြတ်အရှုံး",
                "အမြတ်အရှုံးအစီရင်ခံစာ",
                "ဝင်ငွေ ထွက်ငွေ",
                "အမြတ်အစီရင်ခံစာ"
            ],
            "sales_report": [
                "ရောင်းချမှုအစီရင်ခံစာ",
                "အရောင်းအစီရင်ခံစာ",
                "ရောင်းချမှုဒေတာ",
                "အရောင်းဒေတာ"
            ],
            "product_analysis": [
                "ပစ္စည်းခွဲခြမ်းစိတ်ဖြာမှု",
                "ပစ္စည်းအချက်အလက်",
                "ကုန်ပစ္စည်းအစီရင်ခံစာ"
            ]
        }
    
    def classify_intent(self, query: str) -> Tuple[str, float]:
        """Classify the intent of Burmese query"""
        query_embedding = self.model.encode([query])
        
        best_intent = None
        highest_score = 0
        
        for intent, examples in self.intent_templates.items():
            example_embeddings = self.model.encode(examples)
            
            similarities = torch.cosine_similarity(
                torch.tensor(query_embedding),
                torch.tensor(example_embeddings)
            )
            
            max_similarity = similarities.max().item()
            if max_similarity > highest_score:
                highest_score = max_similarity
                best_intent = intent
        
        return best_intent, highest_score


class BurmeseEntityExtractor:
    def __init__(self):
        # Burmese number mapping
        self.burmese_numbers = {
            '၀': '0', '၁': '1', '၂': '2', '၃': '3', '၄': '4',
            '၅': '5', '၆': '6', '၇': '7', '၈': '8', '၉': '9'
        }
        
        # Time period patterns
        self.time_patterns = {
            'last_week': ['အရင်အပတ်', 'ပြီးခဲ့သည့်အပတ်', 'လွန်ခဲ့သည့်အပတ်'],
            'last_month': ['အရင်လ', 'ပြီးခဲ့သည့်လ', 'လွန်ခဲ့သည့်လ'],
            'this_week': ['ဒီအပတ်', 'ယခုအပတ်'],
            'this_month': ['ဒီလ', 'ယခုလ'],
            'last_year': ['အရင်နှစ်', 'ပြီးခဲ့သည့်နှစ်']
        }
    
    def extract_entities(self, query: str) -> Dict:
        """Extract entities from Burmese query"""
        entities = {
            'time_period': None,
            'date_range': None,
            'product_name': None,
            'numbers': []
        }
        
        # Extract time periods
        for period, patterns in self.time_patterns.items():
            for pattern in patterns:
                if pattern in query:
                    entities['time_period'] = period
                    entities['date_range'] = self._get_date_range(period)
                    break
        
        # Extract Burmese numbers
        converted_query = self._convert_burmese_numbers(query)
        number_matches = re.findall(r'\d+', converted_query)
        entities['numbers'] = [int(num) for num in number_matches]
        
        return entities
    
    def _convert_burmese_numbers(self, text: str) -> str:
        """Convert Burmese numbers to Arabic numbers"""
        for burmese, arabic in self.burmese_numbers.items():
            text = text.replace(burmese, arabic)
        return text
    
    def _get_date_range(self, period: str) -> Tuple[str, str]:
        """Get date range for time period"""
        today = datetime.now()
        
        if period == 'last_week':
            end_date = today - timedelta(days=today.weekday() + 1)
            start_date = end_date - timedelta(days=6)
        elif period == 'last_month':
            if today.month == 1:
                start_date = datetime(today.year - 1, 12, 1)
                end_date = datetime(today.year, 1, 1) - timedelta(days=1)
            else:
                start_date = datetime(today.year, today.month - 1, 1)
                end_date = datetime(today.year, today.month, 1) - timedelta(days=1)
        elif period == 'this_week':
            start_date = today - timedelta(days=today.weekday())
            end_date = today
        elif period == 'this_month':
            start_date = datetime(today.year, today.month, 1)
            end_date = today
        else:
            return None, None
        
        return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')


class BurmeseSQLGenerator:
    def __init__(self):
        self.sql_templates = {
            "top_selling_products": """
                SELECT 
                    p.product_name,
                    SUM(s.quantity) as total_quantity,
                    SUM(s.total_amount) as total_sales
                FROM sales s
                JOIN products p ON s.product_id = p.product_id
                WHERE s.sale_date BETWEEN '{start_date}' AND '{end_date}'
                GROUP BY p.product_id, p.product_name
                ORDER BY total_quantity DESC
                LIMIT 10
            """,
            
            "lowest_selling_products": """
                SELECT 
                    p.product_name,
                    COALESCE(SUM(s.quantity), 0) as total_quantity,
                    COALESCE(SUM(s.total_amount), 0) as total_sales
                FROM products p
                LEFT JOIN sales s ON p.product_id = s.product_id 
                    AND s.sale_date BETWEEN '{start_date}' AND '{end_date}'
                GROUP BY p.product_id, p.product_name
                ORDER BY total_quantity ASC
                LIMIT 10
            """,
            
            "profit_loss_report": """
                SELECT 
                    COALESCE(sales_data.total_revenue, 0) as total_revenue,
                    COALESCE(sales_data.total_cost, 0) as total_cost,
                    COALESCE(expense_data.total_expenses, 0) as total_expenses,
                    (COALESCE(sales_data.total_revenue, 0) - COALESCE(sales_data.total_cost, 0) - COALESCE(expense_data.total_expenses, 0)) as net_profit
                FROM 
                    (SELECT 
                        SUM(s.total_amount) as total_revenue,
                        SUM(s.quantity * p.cost_price) as total_cost
                     FROM sales s
                     JOIN products p ON s.product_id = p.product_id
                     WHERE s.sale_date BETWEEN '{start_date}' AND '{end_date}'
                    ) as sales_data
                CROSS JOIN
                    (SELECT 
                        SUM(amount) as total_expenses
                     FROM expenses
                     WHERE expense_date BETWEEN '{start_date}' AND '{end_date}'
                    ) as expense_data
            """,
            
            "sales_report": """
                SELECT 
                    DATE(s.sale_date) as sale_date,
                    COUNT(*) as total_transactions,
                    SUM(s.quantity) as total_quantity,
                    SUM(s.total_amount) as total_amount
                FROM sales s
                WHERE s.sale_date BETWEEN '{start_date}' AND '{end_date}'
                GROUP BY DATE(s.sale_date)
                ORDER BY sale_date DESC
            """
        }
    
    def generate_sql(self, intent: str, entities: Dict, schema: Dict) -> str:
        """Generate SQL query based on intent and entities"""
        if intent not in self.sql_templates:
            return None
        
        sql_template = self.sql_templates[intent]
        
        # Get date range
        if entities.get('date_range'):
            start_date, end_date = entities['date_range']
        else:
            # Default to last month if no date specified
            today = datetime.now()
            start_date = datetime(today.year, today.month - 1, 1).strftime('%Y-%m-%d')
            end_date = (datetime(today.year, today.month, 1) - timedelta(days=1)).strftime('%Y-%m-%d')
        
        # Format SQL with parameters
        formatted_sql = sql_template.format(
            start_date=start_date,
            end_date=end_date
        )
        
        return formatted_sql


class BurmeseResponseGenerator:
    def __init__(self):
        self.response_templates = {
            "top_selling_products": "အရောင်းရဆုံး ပစ္စည်းများမှာ:",
            "lowest_selling_products": "အရောင်းနည်းဆုံး ပစ္စည်းများမှာ:",
            "profit_loss_report": "အမြတ်အရှုံး အစီရင်ခံစာ:",
            "sales_report": "ရောင်းချမှု အစီရင်ခံစာ:"
        }
    
    def generate_response(self, query: str, data: List[Dict], intent: str, entities: Dict) -> str:
        """Generate natural language response in Burmese"""
        if not data:
            return "မေးမြန်းထားသည့် ကာလအတွင်း ဒေတာများ မတွေ့ရှိပါ။"
        
        base_response = self.response_templates.get(intent, "ရလဒ်များမှာ:")
        
        if intent == "top_selling_products":
            return self._format_product_sales_response(data, base_response, True)
        elif intent == "lowest_selling_products":
            return self._format_product_sales_response(data, base_response, False)
        elif intent == "profit_loss_report":
            return self._format_profit_loss_response(data, base_response)
        elif intent == "sales_report":
            return self._format_sales_report_response(data, base_response)
        else:
            return f"{base_response}\n\nဒေတာများကို ရရှိပြီးပါပြီ။"
    
    def _format_product_sales_response(self, data: List[Dict], base_response: str, is_top: bool) -> str:
        """Format product sales response"""
        response = f"{base_response}\n\n"
        
        for i, item in enumerate(data[:5], 1):  # Show top 5
            product_name = item.get('product_name', 'အမည်မသိ')
            quantity = item.get('total_quantity', 0)
            sales = item.get('total_sales', 0)
            
            response += f"{i}. {product_name}\n"
            response += f"   အရေအတွက်: {quantity:,} ခု\n"
            response += f"   ရောင်းချမှု: {sales:,.0f} ကျပ်\n\n"
        
        return response
    
    def _format_profit_loss_response(self, data: List[Dict], base_response: str) -> str:
        """Format profit/loss response"""
        if not data:
            return "အမြတ်အရှုံး ဒေတာများ မတွေ့ရှိပါ။"
        
        item = data[0]
        revenue = item.get('total_revenue', 0) or 0
        cost = item.get('total_cost', 0) or 0
        expenses = item.get('total_expenses', 0) or 0
        profit = item.get('net_profit', 0) or 0
        
        response = f"{base_response}\n\n"
        response += f"စုစုပေါင်း ဝင်ငွေ: {revenue:,.0f} ကျပ်\n"
        response += f"ပစ္စည်း ဝယ်ဈေး: {cost:,.0f} ကျပ်\n"
        response += f"အခြား ကုန်ကျစရိတ်: {expenses:,.0f} ကျပ်\n"
        response += f"{'─' * 30}\n"
        
        if profit >= 0:
            response += f"စုစုပေါင်း အမြတ်: {profit:,.0f} ကျပ်"
        else:
            response += f"စုစုပေါင်း အရှုံး: {abs(profit):,.0f} ကျပ်"
        
        return response
    
    def _format_sales_report_response(self, data: List[Dict], base_response: str) -> str:
        """Format sales report response"""
        response = f"{base_response}\n\n"
        
        total_amount = sum(item.get('total_amount', 0) for item in data)
        total_transactions = sum(item.get('total_transactions', 0) for item in data)
        
        response += f"စုစုပေါင်း ရောင်းချမှု: {total_amount:,.0f} ကျပ်\n"
        response += f"စုစုပေါင်း အရောင်းအရေအတွက်: {total_transactions:,} ကြိမ်\n\n"
        
        response += "နေ့စဉ် ရောင်းချမှု:\n"
        for item in data[:7]:  # Show last 7 days
            date = item.get('sale_date', 'N/A')
            amount = item.get('total_amount', 0)
            transactions = item.get('total_transactions', 0)
            
            response += f"{date}: {amount:,.0f} ကျပ် ({transactions} ကြိမ်)\n"
        
        return response


# Usage Example
if __name__ == "__main__":
    # Database configuration
    db_config = {
        'host': 'localhost',
        'database': 'erp_system',
        'user': 'your_username',
        'password': 'your_password',
        'port': 5432
    }
    
    # Initialize RAG system
    rag_system = BurmeseERPRAGSystem(db_config)
    
    # Test queries
    test_queries = [
        "အရင်အပတ်က ဘာပစ္စည်းအရောင်းရဆုံးလဲ",  # "What products sold the most last week?"
        "အရောင်းနည်းဆုံး ပစ္စည်းကဘာလဲ",              # "What are the lowest selling products?"
        "အရင်လ အရှုံးအမြတ်ကို ပြပါ",                # "Show last month's profit/loss"
        "ဒီလ ရောင်းချမှုအစီရင်ခံစာ ပေးပါ"            # "Give me this month's sales report"
    ]
    
    print("=== Burmese ERP RAG System Testing ===\n")
    
    for query in test_queries:
        print(f"Query: {query}")
        response = rag_system.process_query(query)
        print(f"Response: {response}")
        print("-" * 50)