from google import genai
from google.genai.types import EmbedContentConfig
import query_finder

if __name__ == "__main__":

    # Sample query in Burmese
    # test_query_burmese = "ကုန်ပစ္စည်း အလိုက် အရောင်းစာရင်းကို ကြည့်ချင်တယ်"
    # test_query_burmese = "အသုံးစရိတ် စားရင်း ဘယ်လိုကြည့်ရမလဲ"
    # test_query_burmese = "Invoice အလိုက် ရောင်းအားစာရင်း ထုတ်ပေးပါ။"
    # test_query_burmese = "ရောင်းအားအကောင်းဆုံး ပစ္စည်းကို ဘယ်လိုသိနိုင်မလဲ။"
    # test_query_burmese = "Expense Report နဲ့ SalesByEachInvoice Report က ဘာကွာခြားသလဲ။"
    # test_query_burmese = "SalesByCode နဲ့ SalesByEachInvoice Report က ဘာကွာခြားသလဲ။ ရှင်းပြပါ။"
    # test_query_burmese = "ExpenseDetail Report ကို ရှင်းပြပါ။"


    # Invalid Query
    # test_query_burmese = "ဖောက်သည်အချက်အလက်ကို ဘယ်လိုကြည့်ရမလဲ"
    test_query_burmese = "ကုန်ပစ္စည်းလက်ကျန် စာရင်းကို ဘယ်လိုကြည့်ရမလဲ"

    # Advanced Query
    # test_query_burmese = "နေ့စဉ် အရောင်းစာရင်းကို ကြည့်ချင်တယ် ဆိုရင် SaleByEachInvoce Report ကို သုံးရမလား။ TopSales Report ကို သုံးရမလား။"
    # test_query_burmese = "အရောင်းရဆုံး Item 10 ခု စာရင်းကို ကြည့်ချင်တယ်။ ဘယ် Report ကို သုံးရမလဲ။"

    # Non Sense Query
    # test_query_burmese = "အိမ်မှာ ဘာလုပ်ရမလဲ"
    # test_query_burmese = "ကျောင်းမှန်မှန်တတ် စာမခက်"
    # test_query_burmese = "အလုပ်လုပ်ဖို့ ဘယ်လိုလုပ်ရမလဲ"
    query_finder.run_rag_query(test_query_burmese)