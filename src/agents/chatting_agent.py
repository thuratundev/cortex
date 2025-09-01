from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import app_config
import asyncio

# --- Prompt template definition ---
# Define your system prompt as a string.
SYSTEM_PROMPT = """
သင်သည် ERP စနစ်အတွက် အထူးပြုမေးခွန်းများဖြေဆိုပေးသော Retrieval-Augmented Generation (RAG) AI တစ်ဦးဖြစ်ပါသည်။
သင်၏ အဓိကတာဝန်မှာ အောက်တွင်ပေးထားသော "ရှာဖွေတွေ့ရှိထားသော အချက်အလက် (Context)" ကိုသာ အသုံးပြု၍ "အသုံးပြုသူ၏ မေးခွန်း" ကို တိကျစွာဖြေဆိုရန် ဖြစ်သည်။

**လိုက်နာရန် စည်းမျဉ်းများ:**

1.  **Context ကိုသာ အခြေခံပါ:** သင်၏အဖြေသည် ပေးထားသော "ရှာဖွေတွေ့ရှိထားသော အချက်အလက် (Context)" ၏ ဘောင်အတွင်း၌သာ ရှိရမည်။ သင်၏ ကိုယ်ပိုင်အသိပညာ သို့မဟုတ် အခြားသော အချက်အလက်ရင်းမြစ်များကို **လုံးဝအသုံးမပြုပါနှင့်**။
2.  **တိုက်ရိုက်နှင့် တိကျပါ:** မေးခွန်းကို တိုက်ရိုက်ဖြေဆိုပါ။ အဖြေကို တိုတို၊ လိုရင်းနှင့် တိကျအောင်ရေးပါ။ မလိုအပ်သော နိဒါန်းပျိုးခြင်း၊ နှုတ်ဆက်ခြင်း သို့မဟုတ် အပိုစကားများ ထည့်သွင်းရန်မလိုအပ်ပါ။
3.  **အချက်အလက်မရှိက ဝန်ခံပါ:** အကယ်၍ ပေးထားသော Context သည် မေးခွန်းကိုဖြေဆိုရန် လုံလောက်သော အချက်အလက်မပါဝင်ပါက **"ပေးထားသော အချက်အလက်များအရ ဤမေးခွန်းကို ဖြေဆိုနိုင်ခြင်းမရှိပါ"** ဟု တိကျစွာဖြေဆိုပါ။ မမှန်ကန်သော အချက်အလက်များကို မှန်းဆ၍ **လုံးဝမဖြေဆိုပါနှင့်**။
4.  **ပေါင်းစပ်ဖြေဆိုခြင်း:** မေးခွန်းက အချက်အလက်များကို ပေါင်းစပ်သုံးသပ်ရန် လိုအပ်ပါက (ဥပမာ - နှိုင်းယှဉ်ခိုင်းခြင်း)၊ Context ထဲမှ သက်ဆိုင်ရာ အချက်အလက်များကိုသာ ပေါင်းစပ်ပြီး အဖြေတစ်ခုအဖြစ် ပြန်လည်ဖွဲ့စည်းပါ။ Context တွင်မပါသော အချက်အလက်အသစ်များကို မထည့်သွင်းပါနှင့်။
"""

# Define the user's message with placeholders.
USER_PROMPT = """
=========================
**အသုံးပြုသူ၏ မေးခွန်း:** "{user_query}"
=========================

**ရှာဖွေတွေ့ရှိထားသော Report များ:**
{context_for_prompt}
=========================

**သင်၏ အဖြေ:**
"""


template = ChatPromptTemplate([
    ("system", SYSTEM_PROMPT),
    ("user", USER_PROMPT)
])


CHAT_MODEL = "gemini" #  gemini, openai

async def chat(user_query: str, similar_results: list[dict]) -> str:
    try:
        print(f"Finding best response for query: {user_query}")

        if not similar_results:
            return "No similar results found."

        context_for_prompt = ""
        for i, report in enumerate(similar_results):
            report_name = report['reportname']
            description = report["description"]

            context_for_prompt += f"--- Report {i+1} ---\n"
            context_for_prompt += f"Report အမည်: {report_name}\n"
            context_for_prompt += f"Report ဖော်ပြချက်: {description}\n\n"

        template_variables = {
            "user_query": user_query.strip(),
            "context_for_prompt": context_for_prompt.strip()
        }

        
        formatted_message = template.format_messages(**template_variables)

        chat_model = get_chat_model()

        
        response = chat_model.invoke(formatted_message)

        return response.content

    except Exception as e:
        print(f"ERROR: Failed to generate chat response. Error: {e}")
        return "An error occurred while generating the response."
    
def get_chat_model():

    if CHAT_MODEL == "gemini":
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-001",
            api_key=app_config.GOOGLE_API_KEY,
        )
    elif CHAT_MODEL == "openai":
        return ChatOpenAI(
            model="gpt-5-mini-2025-08-07",
            api_key=app_config.OPENAI_API_KEY,
        )