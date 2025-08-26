import os
from dotenv import load_dotenv

load_dotenv()


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# --- File Paths ---
MASTER_DATA_FILE = os.path.join(DATA_DIR, "feed_master_data.json")

# Output file for the generated embeddings
OUTPUT_EMBEDDINGS_FILE = os.path.join(DATA_DIR, "embedded_data.json")

GOOGLE_API_KEY = os.getenv('GEMINIAI_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

EMBEDDED_MODEL_CONFIG = {
    "google": {
        "model_name": "models/gemini-embedding-001",
        "api_key": GOOGLE_API_KEY
    },
    "openai": {
        "model_name": "text-embedding-3-large",
        "api_key": OPENAI_API_KEY
    }
}

# Database connection parameters
DB_PARAMS = {
    "dbname": "cygnusdb",
    "user": "postgres",
    "password": "015427",
    "host": "localhost",
    "port": "5432"
}