import requests
import json
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from typing import Dict
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser

# Define the expected output model for the email
class EmailOutput(BaseModel):
    email_subject: str = Field(description="Subject of the email")
    email_content: str = Field(description="Content/body of the email")

# Create the Pydantic output parser using the model
email_parser = PydanticOutputParser(pydantic_object=EmailOutput)

# Load environment variables from .env file for local development
load_dotenv()

# Function to fetch the latest offer and signature from NocoDB
def fetch_latest_offer_and_signature(nocodb_api_url, table_id, api_token):
    headers = {
        'xc-token': api_token,
        'Content-Type': 'application/json'
    }
    params = {
        'limit': 1,
        'offset': 0,
        'sort': '-Id'  # Adjust sorting if necessary
    }

    # Making the request to get the latest record
    response = requests.get(
        f"https://{nocodb_api_url}/api/v2/tables/{table_id}/records",
        headers=headers,
        params=params
    )
    
    # Raise an exception if the request was unsuccessful
    if response.status_code != 200:
        raise Exception(f"Failed to fetch records: {response.text}")

    # Parse the response JSON
    data = response.json()
    records = data.get('list', [])
    
    # Raise an exception if no records are found
    if not records:
        raise Exception("No records found in the table for offer and signature.")

    # Access the first (latest) record
    latest_record = records[0]
    
    # Fetch the 'Offer' and 'Signature' values from the record (adjust based on your actual field names)
    offer = latest_record.get('Offer', '')  # Adjust 'Offer' to match your field name
    signature = latest_record.get('Email Signature', '')  # Adjust 'Signature' to match your field name

    return offer, signature

# Function to set up and call LLM for generating emails
def generate_personalized_emails(agents_records, openai_api_key, offer, signature):
  
    # Initialize OpenAI LLM
    llm = ChatOpenAI(api_key=openai_api_key, temperature=0.7, model_name="gpt-4")

    # Create the prompt template with format instructions
    email_prompt_template = """
    You are a professional email assistant. Generate a personalized email subject and content based on the agent's details provided below.

    Agent Details:
    {agent_details}

    Offer:
    {offer}

    Email Signature:
    {signature}

    Email Guidelines:
    - The email should be professional and courteous.
    - Address the agent by their name.
    - Mention any relevant details that make the email personalized.

    {format_instructions}
    """

    # Create the prompt template object
    email_prompt = PromptTemplate(
        input_variables=["agent_details", "offer", "signature"],
        template=email_prompt_template,
        partial_variables={
            "format_instructions": email_parser.get_format_instructions()
        }
    )

    # Process each agent record and generate personalized emails
    personalized_emails = []
    for agent_data in agents_records:  
        agent_details = json.dumps(agent_data)  

        try:
            # Invoke the LLM with the prompt
            prompt_with_data = email_prompt.format(
                agent_details=agent_details, 
                offer=offer, 
                signature=signature
            )
            result = llm(prompt_with_data)
            
            # Parse the result using the email parser
            parsed_result = email_parser.parse(result.content)
            
            # Extract email subject and content from the parsed result
            email_subject = parsed_result.email_subject.strip()
            email_content = parsed_result.email_content.strip()

            # Print the result for debugging
            print("RESULT", parsed_result)

            personalized_emails.append({
                'Email': agent_data.get('Agent Email', ''),
                'Email-Subject': email_subject,
                'Email-Content': email_content,
                'Send Button': False,  # Assuming default value
                'Status': 'Pending'
            })
        except Exception as e:
            print(f"Failed to generate email for agent {agent_data.get('Agent Email', 'unknown')}: {e}")

    return personalized_emails

# Function to insert emails into NocoDB
def insert_emails_into_nocodb(nocodb_api_url, table_id, api_token, emails_data):
    headers = {
        'xc-token': api_token,
        'Content-Type': 'application/json'
    }

    # Modify payload to remove 'fields' key and pass data directly
    payload = [email_data for email_data in emails_data]

    # Print payload for debugging
    print("Payload being sent:", json.dumps(payload, indent=4))

    response = requests.post(
        f"https://{nocodb_api_url}/api/v2/tables/{table_id}/records",
        headers=headers,
        data=json.dumps(payload)
    )

    print("Response:", response.text)

    if response.status_code != 200:
        raise Exception(f"Failed to insert records: {response.text}")

    print("Successfully inserted records into NocoDB.")

# Function to fetch agent details from NocoDB
def fetch_agents_details(nocodb_api_url, table_id, api_token, batch_size=100):
    headers = {
        'xc-token': api_token,
        'Content-Type': 'application/json'
    }
    offset = 0
    all_records = []

    while True:
        params = {
            'limit': batch_size,
            'offset': offset
        }
        response = requests.get(
            f"https://{nocodb_api_url}/api/v2/tables/{table_id}/records",
            headers=headers,
            params=params
        )
        if response.status_code != 200:
            raise Exception(f"Failed to fetch records: {response.text}")

        data = response.json()
        records = data.get('list', [])
        if not records:
            break

        all_records.extend(records)
        offset += batch_size

        if data.get('pageInfo', {}).get('isLastPage', True):
            break

    return all_records

# Mock Pipedream handler function for local development
def handler():

    # Set up configuration variables from environment or mock data
    NOCO_DB_API_URL = os.getenv("NOCO_DB_API_URL")
    NOCO_DB_API_TOKEN = os.getenv("NOCO_DB_API_TOKEN")
    AGENT_TABLE_ID = os.getenv("AGENT_TABLE_ID")
    LEADS_ACTION_TABLE = os.getenv("LEADS_ACTION_TABLE")
    OFFER_SIGNATURE_TABLE_ID = os.getenv("OFFER_SIGNATURE_TABLE_ID")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    BATCH_SIZE = 100

    # Step 1: Fetch agent details
    agents_records = fetch_agents_details(
        NOCO_DB_API_URL, AGENT_TABLE_ID, NOCO_DB_API_TOKEN, BATCH_SIZE)
    print("Agents records:", agents_records)

    # Step 2: Fetch the latest offer and signature
    offer, signature = fetch_latest_offer_and_signature(
        NOCO_DB_API_URL, OFFER_SIGNATURE_TABLE_ID, NOCO_DB_API_TOKEN)
    print("Offer:", offer)
    print("Signature:", signature)

    # Step 3: Generate personalized emails
    personalized_emails = generate_personalized_emails(
        agents_records, OPENAI_API_KEY, offer, signature)
    print("Generated Emails:", personalized_emails)

    # Step 4: Insert emails into NocoDB
    insert_emails_into_nocodb(
        NOCO_DB_API_URL, LEADS_ACTION_TABLE, NOCO_DB_API_TOKEN, personalized_emails)
    print("Inserted emails into NocoDB.")

    # Return data that can be used in future steps
    return {"success": True, "processed_emails": len(personalized_emails)}

# Call handler function for local debugging
if __name__ == "__main__":
    handler()
