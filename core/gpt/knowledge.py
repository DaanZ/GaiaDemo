import glob
import json
import os

import rootpath as rootpath
from langchain_community.document_loaders import TextLoader

conversation_files = os.path.join(rootpath.detect(), "data", "conversations.json")
companies_file = os.path.join(rootpath.detect(), "data", "companies.json")
industries_file = os.path.join(rootpath.detect(), "data", "industries.json")
encoding = "utf-8"


def load_existing_conversations():
    """Load existing notes from a JSON file if it exists."""
    if os.path.exists(conversation_files):
        with open(conversation_files, "r", encoding=encoding) as file:
            return json.load(file)
    return {}


def save_conversations(conversations_dict):
    conversations_json = json.dumps(conversations_dict, indent=4)
    with open(conversation_files, "w", encoding=encoding) as file:
        file.write(conversations_json)


def load_existing_industries():
    """Load existing notes from a JSON file if it exists."""
    if os.path.exists(industries_file):
        with open(industries_file, "r", encoding=encoding) as file:
            return json.load(file)
    return {}


def save_industries(industries_dict):
    industries_json = json.dumps(industries_dict, indent=4)
    with open(industries_file, "w", encoding=encoding) as file:
        file.write(industries_json)


def load_existing_companies():
    """Load existing notes from a JSON file if it exists."""
    if os.path.exists(companies_file):
        with open(companies_file, "r", encoding=encoding) as file:
            return json.load(file)
    return {}


def save_companies(companies_dict):
    companies_json = json.dumps(companies_dict, indent=4)
    with open(companies_file, "w", encoding=encoding) as file:
        file.write(companies_json)


def read_company_meetings(company_id):
    companies = load_existing_companies()
    meeting_info = []
    for company_key in companies:
        company = companies[company_key]
        if company["company_id"] != company_id:
            continue

        if "notes" not in company:
            continue

        for note in company["notes"]:
            if "summary" in note:
                text = note['summary']
            elif "message" in note:
                text = note['message']
            else:
                print("unknown note format", note)
                continue
            meeting_info.append(f"Meeting with {company['name']}: {text}")

        # given that this is the company_id we don't have to find any other anymore.
        break

    return meeting_info


def webpage_pages(company_folder: str = "grounded_world"):
    paths = os.path.join(rootpath.detect(), "data", "website", company_folder, "*.txt")
    pages = []
    for path in glob.glob(paths):
        loader = TextLoader(path, encoding="utf-8")
        pages.extend(loader.load_and_split())
    return pages
