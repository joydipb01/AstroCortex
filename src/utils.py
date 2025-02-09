from datetime import date
import re

def extract_currency(query):
    # List of known currency keywords.
    # You can add more currencies as needed.
    currency_keywords = [
        "rupees",
        "euros",
        "dollars",
        "pounds",
        "yen",
        "won",
        "aud",
        "cad",
        "inr",
        "pkr",
        "gbp",
        "krw",
        "usd"
    ]

    pattern = re.compile(r'\b(' + '|'.join(currency_keywords) + r')\b', re.IGNORECASE)
    
    match = pattern.search(query)
    if match:
        return match.group(1).lower()
    else:
        return "us dollars"


def get_current_date():
    return date.today()

def get_search_prompts(lst: str, query: str):
    list_lst = lst.split(',')

    currency = extract_currency(query)

    date_today = get_current_date()

    search_prompts = []

    for res in list_lst:
        search_prompt = f"What is the cost of {res} in {currency} as of {date_today}?"
        search_prompts.append(search_prompt)
    
    return search_prompts