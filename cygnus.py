from google import genai
from google.genai.types import EmbedContentConfig

# Only run this block for Gemini Developer API
client = genai.Client(api_key='AIzaSyDiidatii6YDYpp-gR-aRdhx7u79x0MK3I')

response = client.models.embed_content(
    model="gemini-embedding-001",
    contents="How do I get a driver's license/learner's permit?",
    config=EmbedContentConfig(
        task_type="RETRIEVAL_DOCUMENT",  # Optional
        output_dimensionality=3072,  # Optional
        title="Driver's License",  # Optional
    ),
)
print(response)