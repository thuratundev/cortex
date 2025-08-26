
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from app_config import EMBEDDED_MODEL_CONFIG
import warnings
warnings.filterwarnings('ignore')


# Configuration Models
EMBEDDING_MODEL_NAME = "google"  # Options: "google", "openai"

MODEL_CONFIG = EMBEDDED_MODEL_CONFIG

def get_embedding_model():
    if EMBEDDING_MODEL_NAME not in MODEL_CONFIG:
        raise ValueError(f"Unsupported embedding model: {EMBEDDING_MODEL_NAME}. Choose from: {list(MODEL_CONFIG.keys())}")
    
    config = MODEL_CONFIG[EMBEDDING_MODEL_NAME]
    
    if EMBEDDING_MODEL_NAME == "google":
        if not config["api_key"]:
            raise ValueError("Google API key is required")
        return GoogleGenerativeAIEmbeddings(
            google_api_key=config["api_key"],
            model=config["model_name"]
        )
    
    elif EMBEDDING_MODEL_NAME == "openai":
        if not config["api_key"]:
            raise ValueError("OpenAI API key is required")
        return OpenAIEmbeddings(
            openai_api_key=config["api_key"],
            model=config["model_name"]
        )

if __name__ == "__main__":
    embedding = get_embedding_model()
    print("Initialized embedding model:", embedding)

    test_text = "ကုန်ပစ္စည်းလက်ကျန် စာရင်းကို ဘယ်လိုကြည့်ရမလဲ"
    embedded_text = embedding.embed_query(test_text)
    print("Embedding for test text:", embedded_text)