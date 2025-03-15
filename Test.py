#!/usr/bin/env python3
import os
import glob
import re
import PyPDF2
import openai
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

def read_pdfs(folder):
    all_text = ""
    pdf_files = glob.glob(os.path.join(folder, "*.pdf"))
    for pdf_file in pdf_files:
        with open(pdf_file, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    all_text += text + "\n"
    return all_text

def extract_monthly_segments(text):
    pattern = r"((?:[A-Za-z]+\s+\d{1,2},\s*\d{4}\s+to\s+[A-Za-z]+\s+\d{1,2},\s*\d{4}):)(.*?)(?=(?:[A-Za-z]+\s+\d{1,2},\s*\d{4}\s+to)|$)"
    segments = re.findall(pattern, text, re.DOTALL)
    monthly_data = []
    for period, content in segments:
        period = period.strip().rstrip(':')
        content = content.strip()
        expense_matches = re.findall(r"(ATM and Debit Card Subtractions|Other Subtractions):\s*-\$([\d,]+\.\d{2})", content)
        deposit_match = re.search(r"Deposits:\s*\$([\d,]+\.\d{2})", content)
        ending_match = re.search(r"Ending Balance:\s*\$([\d,]+\.\d{2})", content)
        total_deposit = float(deposit_match.group(1).replace(",", "")) if deposit_match else 0.0
        total_expense = sum(float(m[1].replace(",", "")) for m in expense_matches) if expense_matches else 0.0
        ending_balance = float(ending_match.group(1).replace(",", "")) if ending_match else None
        monthly_data.append({
            "period": period,
            "total_deposit": total_deposit,
            "total_expense": total_expense,
            "ending_balance": ending_balance,
            "raw_content": content
        })
    return monthly_data

def preprocess_pdf_text(text):
    monthly_data = extract_monthly_segments(text)
    deposit_matches = re.findall(r"Deposits:\s*\$([\d,]+\.\d{2})", text)
    expense_matches = re.findall(r"(ATM and Debit Card Subtractions|Other Subtractions):\s*-\$([\d,]+\.\d{2})", text)
    total_deposits = sum(float(match.replace(",", "")) for match in deposit_matches) if deposit_matches else 0.0
    total_expenses = sum(float(match[1].replace(",", "")) for match in expense_matches) if expense_matches else 0.0

    monthly_summary = "### Monthly Trends:\n\n"
    for data in monthly_data:
        monthly_summary += f"- **{data['period']}**:\n"
        monthly_summary += f"  - Total Deposits: ${data['total_deposit']:.2f}\n"
        monthly_summary += f"  - Total Expenses: ${data['total_expense']:.2f}\n"
        if data['ending_balance'] is not None:
            monthly_summary += f"  - Ending Balance: ${data['ending_balance']:.2f}\n"
        monthly_summary += "\n"
        
    structured_summary = (
        f"### Overall Summary:\n"
        f"- Total Deposits: ${total_deposits:.2f}\n"
        f"- Total Expenses: ${total_expenses:.2f}\n\n"
        f"{monthly_summary}"
    )
    return structured_summary

def get_summary_and_advice(pdf_text, api_key):
    openai.api_key = api_key
    structured_summary = preprocess_pdf_text(pdf_text)
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert financial advisor. Analyze the detailed bank statement data provided below. "
                "Focus on monthly trends, recurring subscriptions, and unusual expenses. Categorize expenses into fixed and variable costs. "
                "Provide specific, actionable advice on budgeting and saving money, including concrete steps and strategies. "
                "If more details are needed, ask clarifying questions."
            )
        },
        {
            "role": "user",
            "content": (
                f"Here is a structured summary of key financial figures extracted from my bank statements:\n\n{structured_summary}\n\n"
                "Below is the full raw data from the bank statements:\n\n" + pdf_text
            )
        }
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.2,
        max_tokens=1500
    )
    return response["choices"][0]["message"]["content"]

def interactive_session(api_key):
    print("Interactive session started. Type 'exit' to quit.")
    while True:
        question = input("You: ")
        if question.strip().lower() == "exit":
            break
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a detailed and specific financial advisor."},
                {"role": "user", "content": question}
            ],
            temperature=0.2,
            max_tokens=1500
        )
        print("ChatGPT:", response["choices"][0]["message"]["content"])

def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        api_key = input("Enter your OpenAI API key: ").strip()

    pdf_folder = "./statements"
    if not os.path.exists(pdf_folder):
        print("Folder './statements' not found!")
        return

    print("Reading PDF bank statements from", pdf_folder)
    pdf_text = read_pdfs(pdf_folder)
    if not pdf_text:
        print("No text was extracted from the PDFs.")
        return

    print("Sending the bank statements to ChatGPT for analysis...")
    try:
        summary_advice = get_summary_and_advice(pdf_text, api_key)
        print("\nSummary and Financial Advice:\n")
        print(summary_advice)
    except openai.error.AuthenticationError:
        print("Authentication error: Please check your OpenAI API key.")
    except openai.error.RateLimitError:
        print("Rate limit error: You have exceeded the API rate limit. Please try again later.")
    except openai.error.APIConnectionError:
        print("API connection error: There was an issue connecting to the OpenAI API.")
    except openai.error.InvalidRequestError as e:
        print(f"Invalid request error: {e}")
    except openai.error.OpenAIError as e:
        print(f"An error occurred while contacting the ChatGPT API: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    interactive_session(api_key)

if __name__ == "__main__":
    main()
