import os

# Fix OpenMP duplicate library crash (FAISS + PyTorch conflict on macOS)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import time
import shutil
import json
import csv
import pandas as pd
import re
import numpy as np
from datetime import datetime
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import hashlib
from streamlit_lottie import st_lottie
import requests
import random
from collections import defaultdict

# Import timing analysis module
from timing_analysis import reset_timing, time_step, display_timing_table, QUERY_TIMING

# Load API key from Streamlit secrets (Cloud) or environment variable (local)
if "OPENAI_API_KEY" not in os.environ:
    try:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass  # User will enter key in sidebar

# Page configuration
st.set_page_config(
    page_title="Taxmen AI v6 - Complete Edition",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Global Directories and Files Setup
# -------------------------------
DOCUMENTS_DIR = "documents"
VECTORSTORE_DIR = "vectorstore"
TEMP_DIR = "temp_uploads"
CERTIFICATES_DIR = "certificates"
CERTIFICATES_DATA_FILE = "certificates_data.csv"
QA_HISTORY_FILE = "qa_history.csv"
USER_RESPONSES_FILE = "user_responses.csv"
USER_CONTEXT_FILE = "user_context.json"
QUESTIONNAIRE_CONFIG_FILE = "questionnaire_config.json"

# Create all necessary directories
os.makedirs(VECTORSTORE_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(CERTIFICATES_DIR, exist_ok=True)


def clear_temp_directory():
    """Clear temporary upload directory"""
    for filename in os.listdir(TEMP_DIR):
        file_path = os.path.join(TEMP_DIR, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")


# -------------------------------
# COMPLETE Questionnaire Configuration - ALL 13 Categories
# -------------------------------
COMPLETE_QUESTIONNAIRE = {
    "Small Business Relief - SBR Filing": {
        "description": "Assessment for Small Business Relief Eligibility (AED 3 million threshold)",
        "methodology": """
CALCULATION METHODOLOGY:
1. Gross Receipts Basis (NOT net income or profits)
2. Include: Regular revenue + Asset sales + Out-of-scope/Exempt VAT transactions + Dividends from shareholdings
3. Important: Revenue from VAT returns must match; if not, add exempt/out-of-scope amounts
4. All bank receipts must align with reported revenue
5. Threshold: AED 3,000,000 total gross receipts
        """,
        "questions": [
            {"id": "sbr_1", "question": "What is your total revenue/turnover for the tax period (AED)?",
             "type": "number"},
            {"id": "sbr_2",
             "question": "Do you have any asset sales in the tax period? If yes, what is the sale amount (AED)?",
             "type": "number"},
            {"id": "sbr_3", "question": "Does your revenue figure match the VAT return revenues?", "type": "select",
             "options": ["Yes", "No", "Not VAT Registered"]},
            {"id": "sbr_4",
             "question": "If revenue doesn't match VAT, what is the total amount of exempt/out-of-scope transactions (AED)?",
             "type": "number"},
            {"id": "sbr_5", "question": "Is the company a shareholder in another company locally or internationally?",
             "type": "select", "options": ["Yes", "No"]},
            {"id": "sbr_6", "question": "Did you receive any dividends? If yes, what is the amount (AED)?",
             "type": "number"},
            {"id": "sbr_7", "question": "Do your receipts in all bank accounts align with the reported revenue?",
             "type": "select", "options": ["Yes", "No", "Not Sure"]},
            {"id": "sbr_8", "question": "Did you make profits in excess of AED 375,000, or are you in losses?",
             "type": "select", "options": ["Profit > 375K", "Profit < 375K", "Losses"]},
            {"id": "sbr_9",
             "question": "If you are in losses, do you plan to carry forward tax losses for future tax benefits?",
             "type": "select", "options": ["Yes", "No", "N/A"]},
            {"id": "sbr_10", "question": "Is the company part of any unincorporated partnerships?", "type": "select",
             "options": ["Yes", "No"]},
            {"id": "sbr_11", "question": "Is the company a tax resident in another country?", "type": "select",
             "options": ["Yes", "No"]},
            {"id": "sbr_12",
             "question": "Does the company have any property, intangible assets, or financial assets purchased before the corporate tax period?",
             "type": "select", "options": ["Yes", "No"]},
            {"id": "sbr_13",
             "question": "Do you have any assets valued using fair value method in financial statements?",
             "type": "select", "options": ["Yes", "No"]}
        ]
    },
    "Transitional Tax Benefits - Property": {
        "description": "Transitional relief calculations for real estate assets",
        "methodology": """
CALCULATION METHODOLOGY:
1. WITH TITLE DEED: Straightforward calculation - compare market value method vs time apportionment
2. OFF-PLAN PROPERTY: 
   - Check control: If user has control (can use/lease/sell), capitalize under IAS 40
   - Oqood (approved contract) = control if economic rewards can be achieved through sale
3. CALCULATIONS REQUIRED:
   - Market Value Method: Use FMV on first day of tax period
   - Time Apportionment Method: (Days held pre-tax / Total days held) × Gain
4. Note: Tax savings for unsold units cannot use time apportionment (only on sale date)
5. Provide comparison and recommend best method
        """,
        "questions": [
            {"id": "tp_1",
             "question": "Do you own property under the company name purchased before your first corporate tax period?",
             "type": "select", "options": ["Yes", "No"]},
            {"id": "tp_2", "question": "When did you purchase? (Date of Oqood / Property Title Deed)", "type": "date"},
            {"id": "tp_3",
             "question": "Is the property ready or off-plan? If under construction, do you have title deed or direct access?",
             "type": "select",
             "options": ["Ready with Title Deed", "Off-plan with Oqood", "Off-plan no control", "Under Construction"]},
            {"id": "tp_4", "question": "Are you still holding the property, or has it been sold?", "type": "select",
             "options": ["Still Holding", "Sold"]},
            {"id": "tp_5", "question": "If sold, please provide the sale date and selling price (AED):",
             "type": "text"},
            {"id": "tp_6", "question": "How was the property accounted for in the books?", "type": "select",
             "options": ["Historical Cost Method", "Fair Value Method (IAS 40)", "Inventory (IAS 2)"]},
            {"id": "tp_7",
             "question": "What was the market value of the property on the first day of the tax period (AED)?",
             "type": "number"},
            {"id": "tp_8", "question": "Are you able to provide a valuation certificate at this value?",
             "type": "select", "options": ["Yes", "No", "Can Obtain"]},
            {"id": "tp_9",
             "question": "What was the original purchase price including all directly attributable costs (registration, trustee fees, commissions) (AED)?",
             "type": "number"},
            {"id": "tp_10", "question": "What's the selling price if the property is sold (AED)?", "type": "number"}
        ]
    },
    "Transitional Tax Benefits - Intangible Assets": {
        "description": "Transitional relief for intangible assets (same methodology as property, no control issues)",
        "methodology": """
CALCULATION METHODOLOGY:
Same as property transitional rules EXCEPT control issue doesn't apply.
1. Compare Market Value Method vs Time Apportionment Method
2. Market Value Method: FMV on first day of tax period
3. Time Apportionment: (Days held pre-tax / Total days held) × Gain
4. Recommend optimal method based on calculations
        """,
        "questions": [
            {"id": "tia_1",
             "question": "Do you own intangible assets under company name purchased before first corporate tax period?",
             "type": "select", "options": ["Yes", "No"]},
            {"id": "tia_2", "question": "If yes, provide the purchase date:", "type": "date"},
            {"id": "tia_3", "question": "Are you still holding the intangible assets, or have they been sold?",
             "type": "select", "options": ["Still Holding", "Sold"]},
            {"id": "tia_4", "question": "If sold, provide the date:", "type": "date"},
            {"id": "tia_5", "question": "How were intangible assets accounted for in the books?", "type": "select",
             "options": ["Historical Cost Method", "Fair Value Method (IAS 40)"]},
            {"id": "tia_6",
             "question": "What was the market value of intangible assets on first day of tax period (AED)?",
             "type": "number"},
            {"id": "tia_7",
             "question": "Are you able to reliably measure the value or provide a valuation certificate?",
             "type": "select", "options": ["Yes", "No", "Can Obtain"]},
            {"id": "tia_8",
             "question": "What was the original purchase price including all directly attributable costs (AED)?",
             "type": "number"},
            {"id": "tia_9", "question": "What's the selling price if sold (AED)?", "type": "number"}
        ]
    },
    "Transitional Tax Benefits - Financial Assets": {
        "description": "Transitional relief for financial assets",
        "methodology": """
CALCULATION METHODOLOGY:
Same as Intangible Assets - control issues don't apply.
1. Compare Market Value Method vs Time Apportionment Method
2. Calculate tax savings under both methods
3. Provide recommendation on optimal method
        """,
        "questions": [
            {"id": "tfa_1",
             "question": "Do you own financial assets under company name purchased before first corporate tax period?",
             "type": "select", "options": ["Yes", "No"]},
            {"id": "tfa_2", "question": "If yes, provide the date:", "type": "date"},
            {"id": "tfa_3", "question": "Are you still holding financial assets, or have they been sold?",
             "type": "select", "options": ["Still Holding", "Sold"]},
            {"id": "tfa_4", "question": "If sold, provide the date:", "type": "date"},
            {"id": "tfa_5", "question": "How were financial assets accounted for in the books?", "type": "select",
             "options": ["Historical Cost Method", "Fair Value Method (IAS 40)"]},
            {"id": "tfa_6",
             "question": "What was the market value of financial assets on first day of tax period (AED)?",
             "type": "number"},
            {"id": "tfa_7",
             "question": "Are you able to reliably measure the value or provide a valuation certificate?",
             "type": "select", "options": ["Yes", "No", "Can Obtain"]},
            {"id": "tfa_8",
             "question": "What was the original purchase price including all directly attributable costs (AED)?",
             "type": "number"},
            {"id": "tfa_9", "question": "What's the selling price if the asset is sold (AED)?", "type": "number"}
        ]
    },
    "Connected Person Payments": {
        "description": "Payments to owners, shareholders, directors, relatives",
        "methodology": """
CALCULATION METHODOLOGY:
1. Identify ALL payments to connected parties (shareholders, directors, officers, partners, relatives)
2. Calculate EXCESS AMOUNT: Actual payment - Market rate (arm's length)
3. IMPLICATIONS:
   - Excess amounts are NOT tax deductible
   - May trigger transfer pricing documentation requirements
   - Subject to additional scrutiny by FTA
4. ADDITIONAL REQUIREMENTS:
   - If total payments > AED 500,000: Enhanced disclosure required
   - Need contracts/agreements supporting payment terms
   - Justification for any excess (special skills, risks, expertise)
   - Must demonstrate arm's length principle
5. Consider: Salaries, commissions, benefits, non-monetary benefits
        """,
        "questions": [
            {"id": "cp_1",
             "question": "Were there any payments, benefits, or transactions to shareholders, directors, officers, partners, or related parties?",
             "type": "select", "options": ["Yes", "No"]},
            {"id": "cp_2", "question": "What was the total amount of these payments (AED)?", "type": "number"},
            {"id": "cp_3",
             "question": "What would the payment have been if made to an unconnected party with similar role, responsibility, risk profile, and expertise (AED)?",
             "type": "number"},
            {"id": "cp_4", "question": "If actual payment differs from market rate, why is there a difference?",
             "type": "text"},
            {"id": "cp_5",
             "question": "Does the total payment to connected parties exceed AED 500,000 for the Tax Period?",
             "type": "select", "options": ["Yes", "No"]},
            {"id": "cp_6",
             "question": "Was this payment recurring, and did it start before Corporate Tax became applicable?",
             "type": "select", "options": ["Yes - Recurring before CT", "Yes - New payment", "No"]},
            {"id": "cp_7",
             "question": "Are there contracts or agreements supporting the payment terms to the connected party?",
             "type": "select", "options": ["Yes", "No", "Partial"]},
            {"id": "cp_8", "question": "Were there any non-monetary benefits provided to connected parties?",
             "type": "select", "options": ["Yes", "No"]},
            {"id": "cp_9", "question": "If yes, describe and estimate value of non-monetary benefits:", "type": "text"}
        ]
    },
    "Related Party Transactions": {
        "description": "Transfer pricing and related party transaction assessment",
        "questions": [
            {"id": "rpt_1",
             "question": "Is there any transaction with a person, family member, or company that owns, controls, or significantly influences 50% or more of your business?",
             "type": "select", "options": ["Yes", "No"]},
            {"id": "rpt_2",
             "question": "Is there any transaction with another business commonly owned or controlled by the same person or group?",
             "type": "select", "options": ["Yes", "No"]},
            {"id": "rpt_3", "question": "What types of transactions do you have with related parties?",
             "type": "multiselect",
             "options": ["Sales", "Purchases", "Loans", "Services", "Assets", "Management Fees", "Royalties", "Other"]},
            {"id": "rpt_4", "question": "How were the prices or charges for these transactions determined?",
             "type": "text"},
            {"id": "rpt_5", "question": "Would the price be the same if the transaction was with an independent party?",
             "type": "select", "options": ["Yes", "No", "Similar", "Different"]},
            {"id": "rpt_6", "question": "If different, what special skills, risks, or benefits justify the difference?",
             "type": "text"},
            {"id": "rpt_7",
             "question": "Do you have agreements, contracts, or board approvals supporting related party transactions?",
             "type": "select", "options": ["Yes", "No", "Partial"]},
            {"id": "rpt_8",
             "question": "Is the total value of all transactions with related parties during the year less than AED 40 million?",
             "type": "select", "options": ["Yes", "No"]},
            {"id": "rpt_9", "question": "Does any single type of related party transaction exceed AED 4 million?",
             "type": "select", "options": ["Yes", "No"]},
            {"id": "rpt_10",
             "question": "Is the company part of a multinational group with global revenues above AED 3.15 billion, or does the company itself have annual revenues above AED 200 million?",
             "type": "select", "options": ["Yes", "No"]}
        ]
    },
    "Real Estate": {
        "description": "Real estate investment and taxation assessment",
        "methodology": """
CRITICAL NOTES:
1. Real estate investment income of INDIVIDUALS is NOT taxable in UAE (clearly stated in rules)
2. Must refer to datasets and provide specific, accurate answers (not generic ChatGPT responses)
3. Consider: Property type, ownership structure, business activity, accounting treatment
        """,
        "questions": [
            {"id": "re_1", "question": "Who owns the property?", "type": "select",
             "options": ["Individual", "Company", "Joint Ownership", "Freezone Company"]},
            {"id": "re_2", "question": "If it's an individual, what is the nationality of the owner?", "type": "text"},
            {"id": "re_3", "question": "What's the nature of the property?", "type": "select",
             "options": ["Commercial", "Residential", "Mixed Use"]},
            {"id": "re_4", "question": "What's the licensed activity of the Company?", "type": "select",
             "options": ["Buying & Selling of Real Estate", "Holding Co / Investment Company",
                         "Real Estate Development", "Other"]},
            {"id": "re_5",
             "question": "Is the company intending to buy and sell in the short term, or hold for rentals and long-term appreciation?",
             "type": "select",
             "options": ["Short-term trading", "Long-term rental income", "Long-term appreciation", "Mixed strategy"]},
            {"id": "re_6", "question": "Do you have the Title of the property, or is it still under SPA / Oqood?",
             "type": "select", "options": ["Title Deed", "SPA", "Oqood", "Under Construction"]},
            {"id": "re_7", "question": "When did you purchase? (Date of Oqood / Property Title Deed)", "type": "date"},
            {"id": "re_8", "question": "Do you already have books of accounts prepared?", "type": "select",
             "options": ["Yes", "No", "In Progress"]},
            {"id": "re_9", "question": "What's the concern you would like to address?", "type": "text"},
            {"id": "re_10", "question": "Do you have any document to upload for analysis?", "type": "select",
             "options": ["Yes - will upload", "No"]}
        ]
    },
    "Qualifying Freezone Persons (QFZP)": {
        "description": "Assessment for Qualifying Free Zone Person status (0% tax rate eligibility)",
        "questions": [
            {"id": "qfzp_1", "question": "Is your company registered in a freezone?", "type": "select",
             "options": ["Yes", "No"]},
            {"id": "qfzp_2", "question": "Provide total revenue for the tax period (AED):", "type": "number"},
            {"id": "qfzp_3",
             "question": "Income from transactions with natural persons (except aircraft leasing, ship operations, fund management, wealth management) - Amount (AED) and %:",
             "type": "text"},
            {"id": "qfzp_4", "question": "Income from banking activities - Amount (AED) and %:", "type": "text"},
            {"id": "qfzp_5",
             "question": "Income from insurance activities (excluding reinsurance or captive insurance) - Amount (AED) and %:",
             "type": "text"},
            {"id": "qfzp_6",
             "question": "Income from financial and leasing activities (except ships, aircraft, treasury, and related party financing) - Amount (AED) and %:",
             "type": "text"},
            {"id": "qfzp_7",
             "question": "Income from immovable property other than commercial property in freezone - Amount (AED) and %:",
             "type": "text"},
            {"id": "qfzp_8", "question": "Income from qualifying activities with mainland/international clients:",
             "type": "multiselect", "options": ["Manufacturing and processing", "Trading of commodities",
                                                "Ship management/ownership/operation", "Aircraft finance and leasing",
                                                "Group treasury services", "Holding of shares/securities",
                                                "Reinsurance", "Fund management/wealth management",
                                                "Headquarter services", "Logistics services"]},
            {"id": "qfzp_9", "question": "Provide total income from above qualifying activities (AED):",
             "type": "number"},
            {"id": "qfzp_10", "question": "How much of your income (amount and %) is with other freezone persons?",
             "type": "text"},
            {"id": "qfzp_11",
             "question": "If you outsource any activity, is it to freezone persons and do you supervise that work?",
             "type": "select",
             "options": ["Yes - to FZ with supervision", "Yes - to FZ no supervision", "No outsourcing",
                         "Outsourced to non-FZ"]},
            {"id": "qfzp_12",
             "question": "Are your core income generating activities and key decisions carried out in the freezone?",
             "type": "select", "options": ["Yes", "No", "Partially"]},
            {"id": "qfzp_13",
             "question": "Do you incur expenses in the company that are as per usual benchmark in the same industry?",
             "type": "select", "options": ["Yes", "No", "Not Sure"]},
            {"id": "qfzp_14",
             "question": "Do you have real operations in the freezone? (e.g., space, adequate staff to run your activities)",
             "type": "select", "options": ["Yes - Full operations", "Yes - Minimal", "No"]}
        ]
    },
    "Tax Residency Certificate - Companies": {
        "description": "Eligibility assessment for UAE Tax Residency Certificate for Companies",
        "questions": [
            {"id": "trc_c_1", "question": "Is your company incorporated or legally registered in the UAE?",
             "type": "select", "options": ["Yes", "No"]},
            {"id": "trc_c_2",
             "question": "Is the place of effective management of the company in the UAE (i.e., key management and commercial decisions are made in the UAE)?",
             "type": "select", "options": ["Yes", "No", "Partially"]},
            {"id": "trc_c_3", "question": "Does your company maintain proper books and records in the UAE?",
             "type": "select", "options": ["Yes", "No", "Partial"]},
            {"id": "trc_c_4",
             "question": "Has your company filed UAE Corporate Tax returns (if applicable) and complied with other tax obligations?",
             "type": "select", "options": ["Yes", "No", "Not yet required", "In progress"]}
        ]
    },
    "Tax Residency Certificate - Individuals": {
        "description": "Eligibility assessment for UAE Tax Residency Certificate for Individuals",
        "questions": [
            {"id": "trc_i_1",
             "question": "Is your usual home and main financial/personal interests located in the UAE?",
             "type": "select", "options": ["Yes", "No"]},
            {"id": "trc_i_2", "question": "Do you own or rent a home in the UAE?", "type": "select",
             "options": ["Own", "Rent", "Neither"]},
            {"id": "trc_i_3",
             "question": "Are your main family members (spouse, children, dependents) living in the UAE?",
             "type": "select", "options": ["Yes - All", "Yes - Some", "No"]},
            {"id": "trc_i_4",
             "question": "Do you have bank accounts, investments, or other financial assets in the UAE?",
             "type": "select", "options": ["Yes - Significant", "Yes - Some", "No"]},
            {"id": "trc_i_5",
             "question": "Is your primary source of income derived from employment or business activities in the UAE?",
             "type": "select", "options": ["Yes", "No", "Partially"]},
            {"id": "trc_i_6",
             "question": "Are your personal and professional documents (ID, driver's license, health insurance, etc.) registered in the UAE?",
             "type": "select", "options": ["Yes - All", "Yes - Most", "No"]},
            {"id": "trc_i_7",
             "question": "Do you spend the majority of your free time in the UAE compared to other countries?",
             "type": "select", "options": ["Yes", "No"]},
            {"id": "trc_i_8",
             "question": "Are your legal or financial obligations (taxes, loans, insurance) mainly in the UAE?",
             "type": "select", "options": ["Yes", "No", "Partially"]},
            {"id": "trc_i_9",
             "question": "Do you have any significant tax residency in another country that may conflict with UAE residency?",
             "type": "select", "options": ["Yes", "No"]},
            {"id": "trc_i_10", "question": "How many days have you been in the UAE in the last 12 months?",
             "type": "number"},
            {"id": "trc_i_11",
             "question": "If less than 183 days, do you meet the 90-day rule criteria? (UAE/GCC nationality OR UAE Resident Permit AND permanent residence OR employment/business in UAE)",
             "type": "select", "options": ["Yes", "No", "N/A - Over 183 days"]}
        ]
    },
    "Designated Freezone Transactions": {
        "description": "VAT treatment for Designated Free Zone transactions",
        "questions": [
            {"id": "dfz_1",
             "question": "Do you provide services from the Designated Zone to customers in the mainland or abroad?",
             "type": "select", "options": ["Yes", "No"]},
            {"id": "dfz_2",
             "question": "Do you receive services from mainland or foreign suppliers while in the Designated Zone?",
             "type": "select", "options": ["Yes", "No"]},
            {"id": "dfz_3",
             "question": "Have you had any lost, missing, or unaccounted-for goods in the Designated Zone?",
             "type": "select", "options": ["Yes", "No"]},
            {"id": "dfz_4",
             "question": "When selling to mainland, is the buyer a VAT-registered business (B2B) or an individual consumer (B2C)?",
             "type": "select", "options": ["B2B", "B2C", "Both", "N/A"]},
            {"id": "dfz_5", "question": "Do you sell or move goods from the Designated Zone to the UAE mainland?",
             "type": "select", "options": ["Yes", "No"]},
            {"id": "dfz_6", "question": "Do you purchase goods from the UAE mainland into the Designated Zone?",
             "type": "select", "options": ["Yes", "No"]},
            {"id": "dfz_7", "question": "Do you import goods from abroad directly into the Designated Zone?",
             "type": "select", "options": ["Yes", "No"]},
            {"id": "dfz_8",
             "question": "When transferring, do the goods remain under customs suspension and not altered?",
             "type": "select", "options": ["Yes", "No", "N/A"]},
            {"id": "dfz_9", "question": "Do you transfer goods between two different Designated Zones?",
             "type": "select", "options": ["Yes", "No"]},
            {"id": "dfz_10", "question": "Are any of those goods consumed, used, or written off inside the zone?",
             "type": "select", "options": ["Yes", "No"]}
        ]
    },
    "Profit Margin Scheme (Used Goods Sales VAT)": {
        "description": "VAT Profit Margin Scheme for used goods, antiques, collectibles",
        "questions": [
            {"id": "pms_1", "question": "What type of goods are you selling?", "type": "multiselect",
             "options": ["Used goods", "Antiques", "Collectibles"]},
            {"id": "pms_2", "question": "From whom did you purchase these goods?", "type": "select",
             "options": ["VAT-registered supplier", "Non-registered person", "Mixed"]},
            {"id": "pms_3", "question": "Did you claim input VAT on the purchase of these goods?", "type": "select",
             "options": ["Yes", "No"]},
            {"id": "pms_4", "question": "Does the purchase invoice you received show VAT separately?", "type": "select",
             "options": ["Yes", "No"]},
            {"id": "pms_5",
             "question": "Have you included any additional costs (like transport, commission, or repairs) in the purchase price of these goods?",
             "type": "select", "options": ["Yes", "No"]},
            {"id": "pms_6",
             "question": "Do you keep a record or stock ledger of each item purchased and sold under Profit Margin Scheme?",
             "type": "select", "options": ["Yes", "No", "In Progress"]},
            {"id": "pms_7",
             "question": "When you issue invoices to customers, do you include the wording 'Tax charged with reference to the profit margin'?",
             "type": "select", "options": ["Yes", "No"]},
            {"id": "pms_8", "question": "Is your selling price lower than your purchase cost (making a loss)?",
             "type": "select", "options": ["Yes - Loss on some items", "No - All profitable"]}
        ]
    },
    "Voluntary Disclosures (VDS) - VAT": {
        "description": "Voluntary disclosure for VAT errors or underpayments",
        "questions": [
            {"id": "vds_vat_1",
             "question": "Did you discover any error in your VAT return or refund application that needs correction?",
             "type": "select", "options": ["Yes", "No"]},
            {"id": "vds_vat_2", "question": "Has this type of error happened before (first time or repeated)?",
             "type": "select", "options": ["First time", "Repeated"]},
            {"id": "vds_vat_3", "question": "Is the voluntary disclosure already submitted?", "type": "select",
             "options": ["Yes", "No", "In Progress"]},
            {"id": "vds_vat_4", "question": "When did the error occur?", "type": "select",
             "options": ["Within last 1 year", "Within last 2 years", "Within last 3 years", "Within last 4 years",
                         "More than 4 years"]},
            {"id": "vds_vat_5", "question": "How much is the underpaid / overpaid VAT amount (AED)?", "type": "number"},
            {"id": "vds_vat_6",
             "question": "Are you prepared to settle the underpaid VAT within 20 business days to avoid late payment interest?",
             "type": "select", "options": ["Yes", "No", "N/A - Overpaid"]}
        ]
    },
    "Voluntary Disclosures (VDS) - Corporate Tax": {
        "description": "Voluntary disclosure for Corporate Tax errors or underpayments",
        "questions": [
            {"id": "vds_ct_1", "question": "Did you discover any mistake or underpayment in your Corporate Tax return?",
             "type": "select", "options": ["Yes", "No"]},
            {"id": "vds_ct_2", "question": "When was the original tax return due, and how long ago was it filed?",
             "type": "text"},
            {"id": "vds_ct_3", "question": "What is the underpaid Corporate Tax amount (Tax Difference) (AED)?",
             "type": "number"},
            {"id": "vds_ct_4", "question": "Did you submit a Voluntary Disclosure yet, or is it still pending?",
             "type": "select", "options": ["Submitted", "Pending", "Planning to submit"]}
        ]
    }
}


def create_default_questionnaire():
    """Create default questionnaire configuration file"""
    if not os.path.exists(QUESTIONNAIRE_CONFIG_FILE):
        with open(QUESTIONNAIRE_CONFIG_FILE, 'w') as f:
            json.dump(COMPLETE_QUESTIONNAIRE, f, indent=4)


def load_questionnaire():
    """Load questionnaire from configuration file"""
    create_default_questionnaire()
    try:
        with open(QUESTIONNAIRE_CONFIG_FILE, 'r') as f:
            return json.load(f)
    except:
        return COMPLETE_QUESTIONNAIRE


# -------------------------------
# Persistent User Context Management
# -------------------------------
def save_persistent_context(user_id, category, responses):
    """Save user context persistently (one-time questionnaire)"""
    try:
        context_data = {}
        if os.path.exists(USER_CONTEXT_FILE):
            with open(USER_CONTEXT_FILE, 'r') as f:
                context_data = json.load(f)

        if user_id not in context_data:
            context_data[user_id] = {}

        context_data[user_id][category] = {
            'responses': responses,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'completed': True
        }

        with open(USER_CONTEXT_FILE, 'w') as f:
            json.dump(context_data, f, indent=4)

        return True
    except Exception as e:
        st.error(f"Error saving context: {str(e)}")
        return False


def load_persistent_context(user_id):
    """Load user's persistent context (auto-load on login)"""
    try:
        if not os.path.exists(USER_CONTEXT_FILE):
            return {}

        with open(USER_CONTEXT_FILE, 'r') as f:
            context_data = json.load(f)

        return context_data.get(user_id, {})
    except Exception as e:
        st.error(f"Error loading context: {str(e)}")
        return {}


def check_questionnaire_completed(user_id, category):
    """Check if user has completed a specific questionnaire"""
    context = load_persistent_context(user_id)
    return category in context and context[category].get('completed', False)


def get_all_completed_categories(user_id):
    """Get list of all completed questionnaire categories"""
    context = load_persistent_context(user_id)
    return [cat for cat, data in context.items() if data.get('completed', False)]


def build_comprehensive_user_context(user_id):
    """Build complete user context from persistent storage"""
    context = load_persistent_context(user_id)
    questionnaire = load_questionnaire()

    if not context:
        return ""

    context_parts = ["=== COMPREHENSIVE USER PROFILE AND CONTEXT ==="]
    context_parts.append("This information was provided by the user during initial questionnaire completion.")
    context_parts.append("CRITICAL: Use this context for ALL responses. DO NOT ask for this information again.\n")

    for category, data in context.items():
        if not data.get('completed', False):
            continue

        context_parts.append(f"\n{'=' * 60}")
        context_parts.append(f"CATEGORY: {category}")
        context_parts.append(f"Completed: {data['timestamp']}")
        context_parts.append(f"{'=' * 60}")

        if category in questionnaire and 'methodology' in questionnaire[category]:
            context_parts.append(f"\nCALCULATION METHODOLOGY:")
            context_parts.append(questionnaire[category]['methodology'])

        context_parts.append(f"\nUSER'S RESPONSES:")
        for q_id, answer in data['responses'].items():
            context_parts.append(f"  • {q_id}: {answer}")

    context_parts.append(f"\n{'=' * 60}")
    context_parts.append("=== END USER CONTEXT ===\n")

    return "\n".join(context_parts)


# -------------------------------
# Q&A History Management
# -------------------------------
def save_qa_to_history(question, answer, category="General", user_id="default_user"):
    """Save Q&A to history CSV"""
    try:
        qa_data = {
            'user_id': user_id,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'category': category,
            'question': question.replace('\n', ' ').strip(),
            'answer': answer.replace('\n', ' ').strip()[:1000]
        }

        file_exists = os.path.isfile(QA_HISTORY_FILE)
        with open(QA_HISTORY_FILE, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['user_id', 'timestamp', 'category', 'question', 'answer']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)

            if not file_exists:
                writer.writeheader()

            writer.writerow(qa_data)
        return True
    except Exception as e:
        st.error(f"Error saving Q&A history: {str(e)}")
        return False


def load_qa_history(user_id="default_user", limit=10):
    """Load recent Q&A history"""
    try:
        if not os.path.exists(QA_HISTORY_FILE):
            return []

        df = pd.read_csv(QA_HISTORY_FILE)
        df = df[df['user_id'] == user_id]
        df = df.sort_values('timestamp', ascending=False).head(limit)

        history = []
        for _, row in df.iterrows():
            history.append({
                'timestamp': row['timestamp'],
                'category': row['category'],
                'question': row['question'],
                'answer': row['answer']
            })

        return list(reversed(history))
    except Exception as e:
        return []


def build_qa_summary(user_id="default_user"):
    """Build summary of previous Q&A for continuity"""
    history = load_qa_history(user_id, limit=5)

    if not history:
        return ""

    summary_parts = ["=== PREVIOUS CONVERSATION HISTORY ==="]
    summary_parts.append("User has asked these questions before. Use this for continuity.")
    summary_parts.append("DO NOT ask user to repeat information from previous conversations.\n")

    for i, qa in enumerate(history, 1):
        summary_parts.append(f"{i}. [{qa['category']}] {qa['timestamp']}")
        summary_parts.append(f"   Q: {qa['question'][:200]}")
        summary_parts.append(f"   A: {qa['answer'][:200]}...\n")

    summary_parts.append("=== END CONVERSATION HISTORY ===\n")

    return "\n".join(summary_parts)


# -------------------------------
# Enhanced Document Processor with Hash Caching
# -------------------------------
class EnhancedDocumentProcessor:
    """Enhanced document processor with hash-based caching"""

    def __init__(self, documents_dir, vectorstore_dir):
        self.documents_dir = documents_dir
        self.vectorstore_dir = vectorstore_dir
        self.embeddings = OpenAIEmbeddings()

    def extract_text_with_metadata(self, pdf_file, filename):
        """Extract text with metadata"""
        pdf_reader = PdfReader(pdf_file)
        pages_data = []

        doc_info = pdf_reader.metadata or {}
        creation_date = doc_info.get('/CreationDate', '')
        title = doc_info.get('/Title', filename)
        author = doc_info.get('/Author', '')

        for page_num, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text.strip():
                pages_data.append({
                    'text': text,
                    'page': page_num + 1,
                    'filename': filename,
                    'title': title,
                    'author': author,
                    'creation_date': creation_date,
                    'file_size': len(text),
                    'char_count': len(text)
                })

        return pages_data

    def smart_chunking(self, text, metadata, chunk_strategy='adaptive'):
        """Smart chunking based on content length"""
        if chunk_strategy == 'adaptive':
            if len(text) < 500:
                chunk_size = 300
                chunk_overlap = 50
            elif len(text) < 2000:
                chunk_size = 600
                chunk_overlap = 100
            else:
                chunk_size = 1200
                chunk_overlap = 200
        else:
            chunk_size = 1000
            chunk_overlap = 200

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        chunks = splitter.split_text(text)

        chunk_docs = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'chunk_id': i,
                'chunk_size': len(chunk),
                'total_chunks': len(chunks)
            })
            chunk_docs.append({
                'text': chunk,
                'metadata': chunk_metadata
            })

        return chunk_docs

    def create_document_summary(self, text, filename):
        """Create document summary"""
        if len(text) > 4000:
            text = text[:4000] + "..."

        summary_prompt = f"""Create a comprehensive summary of this document covering:
1. Main topics and subjects
2. Key concepts and terminology
3. Document type and purpose

Document: {filename}
Content: {text}

Summary:"""

        try:
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            summary = llm.invoke(summary_prompt).content
            return f"Document: {filename}\nSummary: {summary}"
        except:
            return f"Document: {filename}\nContent: {text[:500]}..."

    def create_hierarchical_vectorstore_with_progress(self, progress_bar, status_text):
        """Create hierarchical vectorstore with progress tracking"""
        doc_summaries = []
        doc_metadata = []
        all_chunks = []
        all_chunk_metadata = []

        pdf_files = [f for f in os.listdir(self.documents_dir) if f.lower().endswith('.pdf')]
        total_files = len(pdf_files)

        for i, pdf_file in enumerate(pdf_files):
            progress = int((i / total_files) * 70)
            progress_bar.progress(progress)
            status_text.text(f"Processing {i + 1}/{total_files}: {pdf_file}")

            try:
                with open(os.path.join(self.documents_dir, pdf_file), "rb") as f:
                    pages_data = self.extract_text_with_metadata(f, pdf_file)

                if not pages_data:
                    continue

                full_text = " ".join([page['text'] for page in pages_data])
                if len(full_text.strip()) < 50:
                    continue

                doc_summary = self.create_document_summary(full_text, pdf_file)
                doc_summaries.append(doc_summary)
                doc_metadata.append({
                    'filename': pdf_file,
                    'type': 'document_summary',
                    'page_count': len(pages_data),
                    'total_chars': len(full_text)
                })

                for page_data in pages_data:
                    if len(page_data['text'].strip()) < 100:
                        continue

                    chunks = self.smart_chunking(page_data['text'], page_data, 'adaptive')

                    for chunk_doc in chunks:
                        if len(chunk_doc['text'].strip()) > 50:
                            all_chunks.append(chunk_doc['text'])
                            all_chunk_metadata.append(chunk_doc['metadata'])

            except Exception as e:
                st.warning(f"Error processing {pdf_file}: {str(e)}")
                continue

        if not doc_summaries or not all_chunks:
            raise ValueError("No valid documents could be processed")

        status_text.text("Creating document-level vectorstore...")
        progress_bar.progress(75)

        doc_vectorstore = FAISS.from_texts(
            texts=doc_summaries,
            embedding=self.embeddings,
            metadatas=doc_metadata
        )

        status_text.text(f"Creating chunk-level vectorstore ({len(all_chunks)} chunks)...")
        progress_bar.progress(85)

        if len(all_chunks) > 1000:
            batch_size = 500
            chunk_vectorstore = None

            for batch_idx in range(0, len(all_chunks), batch_size):
                batch_chunks = all_chunks[batch_idx:batch_idx + batch_size]
                batch_metadata = all_chunk_metadata[batch_idx:batch_idx + batch_size]

                if chunk_vectorstore is None:
                    chunk_vectorstore = FAISS.from_texts(
                        texts=batch_chunks,
                        embedding=self.embeddings,
                        metadatas=batch_metadata
                    )
                else:
                    batch_vectorstore = FAISS.from_texts(
                        texts=batch_chunks,
                        embedding=self.embeddings,
                        metadatas=batch_metadata
                    )
                    chunk_vectorstore.merge_from(batch_vectorstore)

                batch_progress = 85 + int((batch_idx / len(all_chunks)) * 15)
                progress_bar.progress(batch_progress)
        else:
            chunk_vectorstore = FAISS.from_texts(
                texts=all_chunks,
                embedding=self.embeddings,
                metadatas=all_chunk_metadata
            )

        progress_bar.progress(100)
        return doc_vectorstore, chunk_vectorstore


# -------------------------------
# Advanced Retriever
# -------------------------------
class AdvancedRetriever:
    """Advanced retriever for large collections"""

    def __init__(self, doc_vectorstore, chunk_vectorstore):
        self.doc_vectorstore = doc_vectorstore
        self.chunk_vectorstore = chunk_vectorstore

    def create_ensemble_retriever(self):
        """Create ensemble retriever"""
        doc_retriever = self.doc_vectorstore.as_retriever(search_kwargs={"k": 10})

        similarity_retriever = self.chunk_vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        mmr_retriever = self.chunk_vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 15}
        )

        ensemble_retriever = EnsembleRetriever(
            retrievers=[similarity_retriever, mmr_retriever],
            weights=[0.6, 0.4]
        )

        return ensemble_retriever, doc_retriever

    def create_compressed_retriever(self, base_retriever):
        """Add compression"""
        try:
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", streaming=True, temperature=0)
            compressor = LLMChainExtractor.from_llm(llm)
            return ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=base_retriever
            )
        except:
            return base_retriever


# -------------------------------
# Hash-based Document Processing with Caching
# -------------------------------
def compute_documents_hash(documents_dir):
    """Compute hash of all documents for caching"""
    hash_object = hashlib.sha256()
    try:
        pdf_files = sorted([f for f in os.listdir(documents_dir) if f.lower().endswith('.pdf')])
        for filename in pdf_files:
            filepath = os.path.join(documents_dir, filename)
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_object.update(chunk)
        return hash_object.hexdigest()
    except:
        return None


def process_documents_with_caching():
    """Process documents with hash-based caching (avoid reprocessing unless files changed)"""
    processor = EnhancedDocumentProcessor(DOCUMENTS_DIR, VECTORSTORE_DIR)

    current_hash = compute_documents_hash(DOCUMENTS_DIR)
    if not current_hash:
        st.error("Error computing document hash")
        return None, None

    doc_vectorstore_path = os.path.join(VECTORSTORE_DIR, "doc_vectorstore")
    chunk_vectorstore_path = os.path.join(VECTORSTORE_DIR, "chunk_vectorstore")
    hash_file = os.path.join(VECTORSTORE_DIR, "documents_hash.txt")

    # Try to load from cache
    if (os.path.exists(doc_vectorstore_path) and
            os.path.exists(chunk_vectorstore_path) and
            os.path.exists(hash_file)):

        with open(hash_file, "r") as f:
            saved_hash = f.read().strip()

        if saved_hash == current_hash:
            st.info("Loading cached vectorstore (documents unchanged)...")
            try:
                embeddings = OpenAIEmbeddings()
                doc_vectorstore = FAISS.load_local(
                    doc_vectorstore_path,
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                chunk_vectorstore = FAISS.load_local(
                    chunk_vectorstore_path,
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                st.success("Loaded cached vectorstore successfully")
                return doc_vectorstore, chunk_vectorstore
            except Exception as e:
                st.warning(f"Cache load failed: {e}. Rebuilding...")

    # Need to rebuild vectorstore
    pdf_files = [f for f in os.listdir(DOCUMENTS_DIR) if f.lower().endswith('.pdf')]
    total_files = len(pdf_files)

    if total_files == 0:
        st.error("No PDF files found in documents directory")
        return None, None

    st.info(f"Processing {total_files} documents (this will take a few minutes)...")

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        doc_vectorstore, chunk_vectorstore = processor.create_hierarchical_vectorstore_with_progress(
            progress_bar, status_text
        )

        doc_vectorstore.save_local(doc_vectorstore_path)
        chunk_vectorstore.save_local(chunk_vectorstore_path)

        with open(hash_file, "w") as f:
            f.write(current_hash)

        status_text.text("Processing complete!")
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()

        return doc_vectorstore, chunk_vectorstore

    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")
        progress_bar.empty()
        status_text.empty()
        return None, None


def create_context_aware_conversation_chain(doc_vectorstore, chunk_vectorstore, user_context="", qa_summary=""):
    """
    OPTIMIZED: Create conversation chain with COMBINED enhanced prompt
    This eliminates the need for a separate enhancement LLM call
    """
    advanced_retriever = AdvancedRetriever(doc_vectorstore, chunk_vectorstore)
    ensemble_retriever, doc_retriever = advanced_retriever.create_ensemble_retriever()
    compressed_retriever = advanced_retriever.create_compressed_retriever(ensemble_retriever)

    full_context = ""
    if user_context:
        full_context += f"\n{user_context}\n"
    if qa_summary:
        full_context += f"\n{qa_summary}\n"

    # ============================================================================
    # OPTIMIZED PROMPT: Combines RAG accuracy + Enhancement formatting in ONE call
    # ============================================================================
    enhanced_prompt = PromptTemplate(
        template=f"""You are Taxmen AI - a professional, engaging, and empathetic tax advisor with comprehensive knowledge and user context.

{full_context}

CRITICAL INSTRUCTIONS - ANSWER ACCURACY:
1. NOT PURELY QUESTION-BASED: Analyze datasets and documents to provide calculated answers
2. MAINTAIN CONTINUITY: Use user profile and previous Q&A history - NEVER ask for information already provided
3. SHOW CALCULATIONS: Display step-by-step methodology with clear formulas
4. DATASET-DRIVEN: Reference specific documents and regulations, not generic advice
5. APPLY METHODOLOGIES: Use calculation rules from user context automatically

KEY PRINCIPLES:
- Real Estate: Individual investment income NOT taxable locally
- Small Business Relief: Gross receipts = Revenue + Asset sales + Exempt/out-of-scope + Dividends
- Transitional Rules: Compare market value vs time apportionment, consider control for off-plan
- Connected Parties: Calculate excess amounts, state implications and requirements

CRITICAL INSTRUCTIONS - RESPONSE FORMATTING & TONE:
Your response MUST be elegant, friendly, and beautifully articulated with these characteristics:
1. **Warm and Reassuring**: Use a friendly, empathetic tone while maintaining professional accuracy
2. **Well-Structured**: Use appropriate formatting with:
   - Clear sections and headers where relevant
   - Bullet points (•) or numbered lists for multi-step processes
   - **Bold text** for key points, figures, and important terms
   - *Italics* for emphasis on critical concepts
3. **Empathy and Support**: Make the user feel supported and understood
4. **Clear Language**: Explain complex tax concepts in accessible, easy-to-understand terms
5. **Context and Explanations**: Provide helpful context that aids understanding
6. **Professional Signature**: ALWAYS end with:

---
**- Taxmen AI**
*Your Tax Intelligence Partner*

RETRIEVED DOCUMENTS:
{{context}}

USER QUESTION: {{question}}

COMPREHENSIVE ANSWER (accurate calculations + beautiful formatting with markdown):""",
        input_variables=["context", "question"]
    )

    # Use GPT-4 for high-quality single-pass response
    llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0.3)  # Slightly higher temp for better flow

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=compressed_retriever,
        memory=ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True,
            output_key='answer'
        ),
        combine_docs_chain_kwargs={"prompt": enhanced_prompt},
        return_source_documents=True,
        verbose=False
    )

    return conversation_chain


def analyze_collection_stats(doc_vectorstore, chunk_vectorstore):
    """Get collection statistics"""
    doc_count = doc_vectorstore.index.ntotal if hasattr(doc_vectorstore, 'index') else 0
    chunk_count = chunk_vectorstore.index.ntotal if hasattr(chunk_vectorstore, 'index') else 0

    return {
        "total_documents": doc_count,
        "total_chunks": chunk_count,
        "avg_chunks_per_doc": chunk_count / doc_count if doc_count > 0 else 0
    }


# -------------------------------
# Custom PDF Processing for Upload
# -------------------------------
def extract_text_from_pdf(pdf_file):
    """Extract text from PDF"""
    pdf_reader = PdfReader(pdf_file)
    return [page.extract_text() for page in pdf_reader.pages if page.extract_text()]


def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    """Chunk text for processing"""
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator="\n")
    return splitter.split_text(text)


def create_vectorstore(docs, metadatas):
    """Create vectorstore from documents"""
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(texts=docs, embedding=embeddings, metadatas=metadatas)


def process_uploaded_pdf(uploaded_file):
    """Process uploaded PDF file"""
    file_path = os.path.join(TEMP_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with open(file_path, "rb") as f:
        pages_text = extract_text_from_pdf(f)

    docs, metadatas = [], []
    for page_number, page_text in enumerate(pages_text):
        chunks = chunk_text(page_text)
        docs.extend(chunks)
        metadatas.extend([{"source": uploaded_file.name, "page": page_number + 1}] * len(chunks))

    if docs:
        vectorstore = create_vectorstore(docs, metadatas)
        return vectorstore, f"Successfully processed {uploaded_file.name} ({len(docs)} chunks created)"
    else:
        return None, "Could not extract any text from the uploaded PDF."


def check_file_size(uploaded_file):
    """Check if file size is within limits"""
    file_size_mb = uploaded_file.size / (1024 * 1024)
    return file_size_mb <= 5


# -------------------------------
# Certificate Extraction and Management
# -------------------------------
def extract_certificate_data(pdf_file, filename):
    """Extract comprehensive information from certificates using advanced LLM analysis"""
    try:
        pdf_reader = PdfReader(pdf_file)
        full_text = ""

        for page in pdf_reader.pages:
            full_text += page.extract_text() + "\n"

        openai_api_key = os.environ.get("OPENAI_API_KEY")

        if not openai_api_key:
            st.error("OpenAI API key is required for certificate data extraction.")
            return None

        extraction_prompt = f"""You are an expert document analyst specializing in Federal Tax Authority certificates and Business License certificates. Analyze this document and extract ALL relevant information with perfect accuracy.

DOCUMENT TYPE IDENTIFICATION:
First, identify what type of document this is:
1. "Tax Registration Certificate - Corporate Tax"
2. "Tax Registration Certificate - VAT" 
3. "Business License Certificate"
4. "Other Document"

COMPREHENSIVE DATA EXTRACTION:
Extract ALL available information and return as a valid JSON object. Use empty string "" for missing fields.

REQUIRED JSON STRUCTURE (extract ALL applicable fields):
{{
    "filename": "{filename}",
    "upload_date": "{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    "document_type": "",
    "certificate_type": "",
    "tax_registration_number": "",
    "legal_name_english": "",
    "legal_name_arabic": "",
    "registered_address": "",
    "contact_number": "",
    "effective_registration_date": "",
    "license_number": "",
    "licensing_authority": "",
    "issue_date": "",
    "expiry_date": "",
    "version_number": "",
    "first_tax_period_start": "",
    "first_tax_period_end": "",
    "first_return_due_date": "",
    "vat_return_period": "",
    "vat_return_due_date": "",
    "tax_periods_schedule": "",
    "company_type": "",
    "formation_number": "",
    "managers": "",
    "business_activities": "",
    "activity_codes": "",
    "issuing_authority": "",
    "document_reference": "",
    "additional_notes": ""
}}

DOCUMENT TEXT:
{full_text}

Return ONLY the complete JSON object with all applicable fields filled:"""

        llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0)
        response = llm.invoke(extraction_prompt).content.strip()

        try:
            if response.startswith("```"):
                lines = response.split("\n")
                json_lines = []
                in_json = False
                for line in lines:
                    if line.strip().startswith("```"):
                        in_json = not in_json
                        continue
                    if in_json:
                        json_lines.append(line)
                response = "\n".join(json_lines)

            cert_data = json.loads(response)

            expected_fields = [
                "filename", "upload_date", "document_type", "certificate_type",
                "tax_registration_number", "legal_name_english", "legal_name_arabic",
                "registered_address", "contact_number", "effective_registration_date",
                "license_number", "licensing_authority", "issue_date", "expiry_date",
                "version_number", "first_tax_period_start", "first_tax_period_end",
                "first_return_due_date", "vat_return_period", "vat_return_due_date",
                "tax_periods_schedule", "company_type", "formation_number",
                "managers", "business_activities", "activity_codes",
                "issuing_authority", "document_reference", "additional_notes"
            ]

            for field in expected_fields:
                if field not in cert_data:
                    cert_data[field] = ""

            return cert_data

        except json.JSONDecodeError as e:
            st.error(f"Failed to parse AI response as JSON. Error: {str(e)}")
            return None

    except Exception as e:
        st.error(f"Error extracting certificate data: {str(e)}")
        return None


def save_certificate_data_robust(cert_data):
    """Save certificate data with enhanced CSV handling and validation"""
    try:
        file_path = os.path.join(CERTIFICATES_DIR, CERTIFICATES_DATA_FILE)
        file_exists = os.path.isfile(file_path)

        fieldnames = [
            "filename", "upload_date", "document_type", "certificate_type",
            "tax_registration_number", "legal_name_english", "legal_name_arabic",
            "registered_address", "contact_number", "effective_registration_date",
            "license_number", "licensing_authority", "issue_date", "expiry_date",
            "version_number", "first_tax_period_start", "first_tax_period_end",
            "first_return_due_date", "vat_return_period", "vat_return_due_date",
            "tax_periods_schedule", "company_type", "formation_number",
            "managers", "business_activities", "activity_codes",
            "issuing_authority", "document_reference", "additional_notes"
        ]

        cleaned_data = {}
        for field in fieldnames:
            value = cert_data.get(field, "")
            if value:
                value = str(value).replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
                value = re.sub(r',+', ', ', value)
                value = value.replace('"', "'")
                value = value.strip()
            cleaned_data[field] = value

        with open(file_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(
                csvfile,
                fieldnames=fieldnames,
                quoting=csv.QUOTE_ALL,
                escapechar='\\'
            )

            if not file_exists:
                writer.writeheader()

            writer.writerow(cleaned_data)

        return True

    except Exception as e:
        st.error(f"Error saving certificate data: {str(e)}")
        return False


def load_certificates_data_robust():
    """Load certificate data with enhanced error handling"""
    try:
        file_path = os.path.join(CERTIFICATES_DIR, CERTIFICATES_DATA_FILE)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, on_bad_lines='skip', encoding='utf-8')

            expected_columns = [
                "filename", "upload_date", "document_type", "certificate_type",
                "tax_registration_number", "legal_name_english", "legal_name_arabic",
                "registered_address", "contact_number", "effective_registration_date",
                "license_number", "licensing_authority", "issue_date", "expiry_date",
                "version_number", "first_tax_period_start", "first_tax_period_end",
                "first_return_due_date", "vat_return_period", "vat_return_due_date",
                "tax_periods_schedule", "company_type", "formation_number",
                "managers", "business_activities", "activity_codes",
                "issuing_authority", "document_reference", "additional_notes"
            ]

            for col in expected_columns:
                if col not in df.columns:
                    df[col] = ""

            df = df.reindex(columns=expected_columns, fill_value="")
            return df
        else:
            return pd.DataFrame()

    except Exception as e:
        st.error(f"Error loading certificate data: {str(e)}")
        return pd.DataFrame()


# -------------------------------
# LLM Helper Functions
# -------------------------------
def is_tax_related(question):
    """Check if the query is tax-related."""
    tax_guardrail_prompt = f"""
Determine if this question is related to taxes, accounting, finance, or business compliance.
Return ONLY 'YES' if tax/finance/accounting/business related, or 'NO' if about any other topic.

Question: {question}

Answer (ONLY 'YES' or 'NO'):
"""
    try:
        classification_llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0)
        response = classification_llm.invoke(tax_guardrail_prompt).content.strip().upper()
        return response == "YES"
    except:
        return True


# -------------------------------
# Authentication
# -------------------------------
def check_password():
    """Authentication system"""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.markdown("""
        <style>
        .login-container {
            max-width: 500px;
            margin: 0 auto;
            padding: 30px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(128, 128, 128, 0.2);
            color: #f8f9fa;
        }
        .login-title {
            font-size: 2em;
            text-align: center;
            margin-bottom: 20px;
            color: #6b5b95;
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        st.markdown('<h2 class="login-title">Welcome to Taxmen AI v6 Complete</h2>', unsafe_allow_html=True)

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if username == "Taxmen" and password == "Taxmenv6":
                st.session_state.authenticated = True
                clear_temp_directory()
                st.rerun()
            else:
                st.error("Invalid username or password")

        st.markdown('</div>', unsafe_allow_html=True)
        st.stop()

    return True


def load_lottieurl(url):
    """Load Lottie animation from URL"""
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None


# -------------------------------
# Main Application
# -------------------------------
def main():
    check_password()

    # Initialize ALL session state variables
    session_defaults = {
        'doc_vectorstore': None,
        'chunk_vectorstore': None,
        'custom_vectorstore': None,
        'kb_selection': "Existing Knowledge Base",
        'conversation_chain': None,
        'chat_history': [],
        'user_id': "default_user",
        'context_loaded': False,
        'collection_stats': None,
        'uploaded_file_info': None,
        'extracted_cert_data': None
    }

    for key, default_value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

    # Auto-load user context on startup
    if not st.session_state.context_loaded:
        persistent_context = load_persistent_context(st.session_state.user_id)
        completed_categories = get_all_completed_categories(st.session_state.user_id)

        if completed_categories:
            st.sidebar.success(f"Context loaded: {len(completed_categories)} profile(s)")

        st.session_state.context_loaded = True

    # Sidebar
    with st.sidebar:
        st.header("Configuration")

        if st.button("Logout", type="secondary"):
            st.session_state.authenticated = False
            st.rerun()

        st.markdown("---")

        openai_api_key = st.text_input(
            "OpenAI API Key:",
            type="password",
            value=os.environ.get("OPENAI_API_KEY", "")
        )
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        else:
            st.warning("Please enter your OpenAI API key")

        st.markdown("---")
        st.header("Your Profile Status")

        completed = get_all_completed_categories(st.session_state.user_id)
        if completed:
            st.success(f"{len(completed)} active profile(s)")
            for cat in completed:
                st.info(f"✓ {cat}")
        else:
            st.warning("Complete questionnaires to enable context-aware responses")

        st.markdown("---")
        st.header("Recent Activity")
        recent_qa = load_qa_history(st.session_state.user_id, limit=3)
        if recent_qa:
            for i, qa in enumerate(recent_qa, 1):
                with st.expander(f"{i}. {qa['category']}"):
                    st.write(f"**Q:** {qa['question'][:100]}...")

    # Enhanced Title with Styling
    st.markdown("""
    <style>
    .main-title {
        font-size: 3.2em;
        text-align: center;
        font-weight: 700;
        background: linear-gradient(90deg, #ff6f61, #6b5b95, #88b04b);
        background-size: 200% auto;
        color: transparent;
        -webkit-background-clip: text;
        background-clip: text;
        margin-bottom: 10px;
    }
    .enhanced-badge {
        text-align: center;
        color: #6b5b95;
        font-size: 1.2em;
        margin-bottom: 30px;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="main-title">Taxmen AI v6 Complete Edition</h1>', unsafe_allow_html=True)
    st.markdown('<p class="enhanced-badge">⚡ OPTIMIZED: Single-Pass Enhanced Responses</p>',
                unsafe_allow_html=True)

    # Main Navigation Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Profile Setup",
        "💬 Ask Questions",
        "📄 Certificate Manager",
        "📊 Certificate Database",
        "📜 Q&A History"
    ])

    with tab1:
        st.header("Profile Setup (One-Time Questionnaire)")
        st.info(
            "Complete relevant questionnaires once. Your responses are saved permanently and auto-loaded for all future sessions.")

        questionnaire = load_questionnaire()
        category = st.selectbox(
            "Select Assessment Category:",
            list(questionnaire.keys()),
            key="questionnaire_category"
        )

        if category:
            is_completed = check_questionnaire_completed(st.session_state.user_id, category)

            # Load saved responses if available
            saved_responses = {}
            if is_completed:
                context = load_persistent_context(st.session_state.user_id)
                saved_responses = context[category]['responses']

                st.success(f"✓ You've already completed '{category}' on {context[category]['timestamp']}")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("View Saved Responses", width="stretch"):
                        st.session_state[f'view_{category}'] = True
                with col2:
                    if st.button("Update Responses", width="stretch"):
                        st.session_state[f'update_{category}'] = True
                        st.rerun()

            # Show saved responses if requested
            if st.session_state.get(f'view_{category}', False):
                st.markdown("### Your Saved Responses")
                for q_id, ans in saved_responses.items():
                    st.write(f"**{q_id}:** {ans}")
                if st.button("Close"):
                    st.session_state[f'view_{category}'] = False
                    st.rerun()
                st.markdown("---")

            # Show questionnaire (new or update mode)
            if not is_completed or st.session_state.get(f'update_{category}', False):
                st.markdown(f"### {category}")
                st.info(questionnaire[category]['description'])

                if 'methodology' in questionnaire[category]:
                    with st.expander("View Calculation Methodology"):
                        st.code(questionnaire[category]['methodology'])

                # Display questions with pre-populated values
                responses = {}
                questions = questionnaire[category]['questions']

                for q in questions:
                    st.markdown(f"**{q['question']}**")

                    # Get saved value if exists
                    saved_value = saved_responses.get(q['id'], None)

                    if q['type'] == 'select':
                        default_index = 0
                        if saved_value and saved_value in q['options']:
                            default_index = q['options'].index(saved_value) + 1

                        resp = st.selectbox(
                            "Select your answer:",
                            [""] + q['options'],
                            index=default_index,
                            key=f"q_{q['id']}",
                            label_visibility="collapsed"
                        )

                    elif q['type'] == 'multiselect':
                        default_values = []
                        if saved_value:
                            # Handle saved multiselect (stored as string)
                            if isinstance(saved_value, str):
                                try:
                                    default_values = eval(saved_value) if saved_value.startswith('[') else [saved_value]
                                except:
                                    default_values = []
                            else:
                                default_values = saved_value

                        resp = st.multiselect(
                            "Select all that apply:",
                            q['options'],
                            default=default_values,
                            key=f"q_{q['id']}",
                            label_visibility="collapsed"
                        )

                    elif q['type'] == 'number':
                        default_num = float(saved_value) if saved_value else 0.0
                        resp = st.number_input(
                            "Enter amount:",
                            min_value=0.0,
                            value=default_num,
                            key=f"q_{q['id']}",
                            label_visibility="collapsed"
                        )

                    elif q['type'] == 'date':
                        default_date = None
                        if saved_value:
                            try:
                                default_date = datetime.strptime(str(saved_value), '%Y-%m-%d').date()
                            except:
                                pass

                        resp = st.date_input(
                            "Select date:",
                            value=default_date,
                            key=f"q_{q['id']}",
                            label_visibility="collapsed"
                        )

                    else:  # text
                        default_text = saved_value if saved_value else ""
                        resp = st.text_area(
                            "Your answer:",
                            value=default_text,
                            key=f"q_{q['id']}",
                            height=100,
                            label_visibility="collapsed"
                        )

                    if resp:
                        responses[q['id']] = str(resp)

                    st.markdown("---")

                # Save button
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Save Profile", type="primary", width="stretch"):
                        if responses:
                            if save_persistent_context(st.session_state.user_id, category, responses):
                                st.success(f"✅ Profile saved successfully!")
                                st.session_state.conversation_chain = None  # Reset to rebuild with new context
                                if f'update_{category}' in st.session_state:
                                    del st.session_state[f'update_{category}']
                                time.sleep(1)
                                st.rerun()
                        else:
                            st.warning("Please answer at least one question before saving")

                with col2:
                    if st.button("Cancel", width="stretch"):
                        if f'update_{category}' in st.session_state:
                            del st.session_state[f'update_{category}']
                        st.rerun()

    with tab2:
        st.header("Ask Tax Questions")

        # Check and display active context
        completed = get_all_completed_categories(st.session_state.user_id)
        qa_history = load_qa_history(st.session_state.user_id, limit=5)

        if completed or qa_history:
            with st.expander("📋 Active Context (Click to view)", expanded=False):
                if completed:
                    st.success(f"**{len(completed)} Saved Profile(s):**")
                    for cat in completed:
                        st.write(f"  ✓ {cat}")

                if qa_history:
                    st.info(f"**{len(qa_history)} Previous Conversations** loaded as context")
                    for i, qa in enumerate(qa_history[-3:], 1):  # Show last 3
                        st.caption(f"{i}. [{qa['category']}] {qa['question'][:60]}...")
        else:
            st.warning("⚠️ **No context yet.** Complete questionnaires in 'Profile Setup' for personalized answers!")

        st.markdown("---")

        # Knowledge Base Selection
        kb_options = ["Existing Knowledge Base", "Customized Knowledge Base"]
        st.session_state.kb_selection = st.radio(
            "Select Knowledge Base:",
            options=kb_options,
            index=kb_options.index(st.session_state.kb_selection),
            horizontal=True
        )

        # Custom Knowledge Base Upload
        if st.session_state.kb_selection == "Customized Knowledge Base":
            st.subheader("Upload Your Document")
            uploaded_file = st.file_uploader("Choose a PDF file (max 5 MB):", type="pdf")

            if uploaded_file:
                if check_file_size(uploaded_file):
                    if st.button("Process Document"):
                        with st.spinner("Processing your document..."):
                            st.session_state.custom_vectorstore, st.session_state.uploaded_file_info = process_uploaded_pdf(
                                uploaded_file)
                            if st.session_state.custom_vectorstore:
                                st.success(st.session_state.uploaded_file_info)
                                st.session_state.conversation_chain = None
                            else:
                                st.error(st.session_state.uploaded_file_info)
                else:
                    st.error("File size exceeds 5 MB limit")

        if not openai_api_key:
            st.error("⚠️ Please enter your OpenAI API key in the sidebar to continue")
            st.stop()

        # Process documents with caching
        if st.session_state.kb_selection == "Existing Knowledge Base":
            if st.session_state.doc_vectorstore is None or st.session_state.chunk_vectorstore is None:
                with st.spinner("Initializing knowledge base..."):
                    st.session_state.doc_vectorstore, st.session_state.chunk_vectorstore = process_documents_with_caching()

                    if st.session_state.doc_vectorstore and st.session_state.chunk_vectorstore:
                        stats = analyze_collection_stats(
                            st.session_state.doc_vectorstore,
                            st.session_state.chunk_vectorstore
                        )
                        st.session_state.collection_stats = stats
                        st.success(
                            f"✅ Knowledge base ready: {stats['total_documents']} docs, {stats['total_chunks']:,} chunks")

            # Display stats
            if st.session_state.collection_stats:
                with st.expander("📊 Collection Statistics"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Documents", st.session_state.collection_stats['total_documents'])
                    with col2:
                        st.metric("Text Chunks", f"{st.session_state.collection_stats['total_chunks']:,}")
                    with col3:
                        st.metric("Avg Chunks/Doc",
                                  f"{st.session_state.collection_stats['avg_chunks_per_doc']:.1f}")

            # Create conversation chain
            if (st.session_state.doc_vectorstore and st.session_state.chunk_vectorstore):
                user_context = build_comprehensive_user_context(st.session_state.user_id)
                qa_summary = build_qa_summary(st.session_state.user_id)

                if st.session_state.conversation_chain is None:
                    st.session_state.conversation_chain = create_context_aware_conversation_chain(
                        st.session_state.doc_vectorstore,
                        st.session_state.chunk_vectorstore,
                        user_context,
                        qa_summary
                    )

        # Query interface
        question = st.text_area(
            "Your Tax Question:",
            height=120,
            placeholder="e.g., Calculate my Small Business Relief eligibility based on my profile data"
        )

        if completed:
            st.info("💡 Your saved profile data will be automatically used for calculations")

        ask_button = st.button("Get Enhanced Answer", type="primary", width="stretch")

        # ============================================================================
        # OPTIMIZED QUERY HANDLING - Single LLM call with timing
        # ============================================================================
        if ask_button and question.strip():
            if not openai_api_key:
                st.error("⚠️ Please enter your OpenAI API key in the sidebar.")
                st.stop()
            elif not st.session_state.conversation_chain:
                st.error("⚠️ Knowledge base not ready")
                st.stop()

            # START TIMING INSTRUMENTATION
            reset_timing()

            # Step 1: Validate question (only for existing KB)
            if st.session_state.kb_selection == "Existing Knowledge Base":
                with time_step("1. Question Validation", "Check if question is tax-related using LLM"):
                    is_valid = is_tax_related(question)

                if not is_valid:
                    st.error("❌ Please ask tax, accounting, finance, or business compliance related questions")
                    st.stop()

            with st.spinner("🤔 Generating comprehensive answer with beautiful formatting..."):
                try:
                    # Step 2: Build user context
                    with time_step("2. Build User Context", "Load user profile from storage and Q&A history"):
                        user_context = build_comprehensive_user_context(st.session_state.user_id)
                        qa_summary = build_qa_summary(st.session_state.user_id)

                    # Step 3: SINGLE LLM CALL - RAG + Enhancement combined
                    with time_step("3. RAG Chain Invoke (OPTIMIZED)", "Single call: Retrieve + Generate enhanced response"):
                        response = st.session_state.conversation_chain.invoke({'question': question})

                    # Step 4: Extract answer (already enhanced!)
                    with time_step("4. Extract Answer", "Parse answer from response object"):
                        final_answer = response.get('answer', '').strip()

                    # NOTE: Step 5 "Enhance Response" is ELIMINATED - already done in Step 3!

                    # Check if we got a valid answer
                    if not final_answer or final_answer == "Information not available.":
                        st.error("❌ No answer generated. Please rephrase your question or provide more details.")

                        # Still show timing even if no answer
                        if QUERY_TIMING:
                            display_timing_table("Query Time (No Answer Found)")

                    else:
                        # Step 6: Display answer
                        with time_step("6. Display Answer", "Render markdown and show source documents"):
                            st.markdown("### ✨ Your Comprehensive Answer:")
                            st.markdown(final_answer)

                            # Show source documents if available
                            if 'source_documents' in response and response['source_documents']:
                                with st.expander("📚 View Source Documents"):
                                    for i, doc in enumerate(response['source_documents'][:5]):
                                        source = doc.metadata.get('filename', 'Unknown source')
                                        page = doc.metadata.get('page', 'Unknown page')
                                        st.write(f"**Source {i + 1}:** {source} (Page {page})")
                                        st.write(f"*Preview:* {doc.page_content[:200]}...")
                                        st.write("---")

                        # Step 7: Save to history
                        with time_step("7. Save to History", "Write Q&A to CSV history file"):
                            save_qa_to_history(question, final_answer, "General", st.session_state.user_id)
                            st.session_state.chat_history.append((question, final_answer))

                        # ============================================================================
                        # DISPLAY TIMING TABLE
                        # ============================================================================
                        display_timing_table("Total Query Time (OPTIMIZED)")

                        # Show optimization benefit
                        st.success("⚡ **Optimization Active:** Response generated in single LLM call (no separate enhancement step)")

                except Exception as e:
                    st.error(f"❌ Error generating response: {str(e)}")

                    # Show timing even if error occurred
                    if QUERY_TIMING:
                        display_timing_table("Query Time (Error Occurred)")

    with tab3:
        st.header("Certificate Manager")

        st.markdown("""
        Upload tax certificates (Corporate Tax Registration, VAT Registration, Business Licenses) 
        for automatic data extraction and storage.
        """)

        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if openai_api_key:
            st.success("✅ AI-powered extraction enabled")
        else:
            st.warning("⚠️ Enter OpenAI API key in sidebar to enable AI-powered extraction")
            st.info("You can still upload and store certificates manually")

        cert_file = st.file_uploader(
            "Choose a PDF certificate file:",
            type="pdf",
            key="cert_uploader",
            help="Upload Tax Registration or Business License certificates"
        )

        if cert_file:
            st.success(f"✓ Selected: {cert_file.name}")

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("Preview Certificate", width="stretch"):
                    st.info("Certificate preview - PDF viewer would go here")

            with col2:
                extract_button = st.button("Extract Data", width="stretch")
                if extract_button:
                    if not openai_api_key:
                        st.error("❌ OpenAI API key required for extraction")
                    else:
                        with st.spinner("🤖 Processing certificate with AI..."):
                            cert_data = extract_certificate_data(cert_file, cert_file.name)
                            if cert_data:
                                st.session_state['extracted_cert_data'] = cert_data
                                st.success("✅ Certificate data extracted successfully!")

                                st.subheader("Extracted Information:")
                                for key, value in cert_data.items():
                                    if value and key not in ['filename', 'upload_date']:
                                        st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                            else:
                                st.error("❌ Failed to extract certificate data")

            with col3:
                save_button = st.button("Save to Database", width="stretch")
                if save_button:
                    # Use extracted data if available, otherwise extract now
                    if 'extracted_cert_data' in st.session_state:
                        cert_data = st.session_state['extracted_cert_data']
                    elif openai_api_key:
                        with st.spinner("Extracting data..."):
                            cert_data = extract_certificate_data(cert_file, cert_file.name)
                    else:
                        st.error("❌ Please extract data first or enter OpenAI API key")
                        cert_data = None

                    if cert_data:
                        if save_certificate_data_robust(cert_data):
                            # Save physical file
                            cert_path = os.path.join(CERTIFICATES_DIR, cert_file.name)
                            with open(cert_path, "wb") as f:
                                f.write(cert_file.getbuffer())

                            st.success("✅ Certificate saved successfully!")
                            if 'extracted_cert_data' in st.session_state:
                                del st.session_state['extracted_cert_data']
                            time.sleep(2)
                            st.rerun()
                        else:
                            st.error("❌ Failed to save certificate to database")
        else:
            st.info("💡 Upload a certificate PDF file to get started")

            # Show example of what can be extracted
            with st.expander("What information can be extracted?"):
                st.markdown("""
                **From Tax Registration Certificates:**
                - Tax Registration Number (TRN)
                - Legal Name (English & Arabic)
                - Registered Address
                - Effective Registration Date
                - Tax Period Schedule
                - Return Due Dates

                **From Business Licenses:**
                - License Number
                - Company Name
                - Business Activities
                - Issue & Expiry Dates
                - Licensing Authority
                """)

    with tab4:
        st.header("Certificate Database")

        certificates_df = load_certificates_data_robust()

        if not certificates_df.empty:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Certificates", len(certificates_df))
            with col2:
                corp_tax = len(certificates_df[
                                   certificates_df.get('certificate_type', pd.Series()).str.contains('Corporate',
                                                                                                     case=False,
                                                                                                     na=False)])
                st.metric("Corporate Tax", corp_tax)
            with col3:
                vat = len(certificates_df[
                              certificates_df.get('certificate_type', pd.Series()).str.contains('VAT', case=False,
                                                                                                na=False)])
                st.metric("VAT Certificates", vat)
            with col4:
                unique = certificates_df.get('legal_name_english', pd.Series()).nunique()
                st.metric("Unique Companies", unique)

            # Search functionality
            search_term = st.text_input("🔍 Search by Company Name:")
            if search_term and 'legal_name_english' in certificates_df.columns:
                certificates_df = certificates_df[
                    certificates_df['legal_name_english'].str.contains(search_term, case=False, na=False) |
                    certificates_df.get('legal_name_arabic', pd.Series()).str.contains(search_term, case=False,
                                                                                       na=False)
                    ]

            st.dataframe(certificates_df, width="stretch")

            # Export functionality
            csv_data = certificates_df.to_csv(index=False)
            st.download_button(
                "📥 Download as CSV",
                data=csv_data,
                file_name=f"certificates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                width="stretch"
            )
        else:
            st.info("🔭 No certificate data found. Upload and save certificates in the Certificate Manager tab.")

    with tab5:
        st.header("Q&A History")

        history = load_qa_history(st.session_state.user_id, limit=100)

        if history:
            st.success(f"📚 Found {len(history)} previous conversations")

            col1, col2 = st.columns(2)
            with col1:
                categories = list(set([h['category'] for h in history]))
                filter_category = st.selectbox("Filter by category:", ["All"] + categories)

            # Display history
            for i, qa in enumerate(reversed(history), 1):
                if filter_category == "All" or qa['category'] == filter_category:
                    with st.expander(f"{i}. [{qa['category']}] {qa['timestamp']} - {qa['question'][:80]}..."):
                        st.markdown(f"**Question:** {qa['question']}")
                        st.markdown(f"**Answer:** {qa['answer']}")

            # Export history
            history_df = pd.DataFrame(history)
            csv_data = history_df.to_csv(index=False)
            st.download_button(
                "📥 Download Q&A History",
                data=csv_data,
                file_name=f"qa_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                width="stretch"
            )
        else:
            st.info("🔭 No conversation history found. Start asking questions in the 'Ask Questions' tab!")


if __name__ == "__main__":
    clear_temp_directory()
    main()