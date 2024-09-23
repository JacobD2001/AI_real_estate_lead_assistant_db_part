import requests
import json
import os
import logging
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from typing import Dict
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from flask import Flask, request, jsonify

# Configuration
app = Flask(__name__)
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout) 
    ]
)

class EmailOutput(BaseModel):
    email_subject: str = Field(description="Subject of the email")
    email_content: str = Field(description="Content/body of the email")

email_parser = PydanticOutputParser(pydantic_object=EmailOutput)

# Function to fetch the latest offer and signature from NocoDB
def fetch_latest_offer_and_signature(nocodb_api_url, table_id, api_token):
    headers = {
        'xc-token': api_token,
        'Content-Type': 'application/json'
    }
    params = {
        'limit': 1,
        'offset': 0,
        'sort': '-Id'
    }

    response = requests.get(
        f"https://{nocodb_api_url}/api/v2/tables/{table_id}/records",
        headers=headers,
        params=params
    )
    
    if response.status_code != 200:
        logging.error(f"Failed to fetch records: {response.text}")
        raise Exception(f"Failed to fetch records: {response.text}")

    data = response.json()
    records = data.get('list', [])
    
    if not records:
        logging.error("No records found in the table for offer and signature.")
        raise Exception("No records found in the table for offer and signature.")

    latest_record = records[0]
    offer = latest_record.get('Offer', '')
    signature = latest_record.get('Email Signature', '')

    return offer, signature

# Function to set up and call LLM for generating emails
def generate_personalized_emails(agents_records, openai_api_key, offer, signature):
    llm = ChatOpenAI(api_key=openai_api_key, temperature=0.7, model_name="gpt-4o-mini")

    email_prompt_template = """
    # Context
    You are the best email copywriter in the world, specializing in crafting emails that drive real results. You will be given:
    - Agent Details
    - A Specific Offer
    - An Email Signature

    Your task is to create highly personalized emails to send to realtors or real estate agents. The primary goal of the email is to engage the agent and prompt them to respond. Be sure to craft an email that is both friendly and professional while avoiding any pushy sales tactics. Keep the tone personable, yet concise.

    # Instructions
    1. **Personalization**: Focus on the agent's details to create a tailored subject and message. Use details like property location or nearby landmarks to make the email feel unique to them. Compliment the agent on something specific to their information.
    
    2. **Subject Line**: Create a concise, engaging, and personalized subject line that grabs the agent's attention without sounding too formal or pushy.

    3. **Body of the Email**:
    - Start by acknowledging something specific about the agent's property or business, you can get a lot of information from the agent details in below section, especially in {{agent_description}} and {{agent_marketing_area_cities}}
    - Introduce the offer in a subtle, friendly way that makes it feel like a helpful solution to a problem they might have. The offer you will be given comes from an input from user, your task is to craft the offer in a way that it sounds natural and not forced.
    - End with {{signature}}. The signature will be an input from user, introduce it in a way that it sounds natural and not forced, but don't change the name of the company or name of the person.
    - Maintain a conversational tone that encourages the agent to respond or schedule a call. Avoid long paragraphs or overly complex language.
    - Be professional, but not too formal.
    - Pay attention to {{agent_description}}, as it says a lot about the agent, and you can use it to your advantage.

    4. **Call to Action**: Include a clear, friendly invitation for the agent to schedule a call with you. This should feel more like a helpful suggestion rather than a demand.

    5. **Proofreading**: Ensure that the email is free from any grammar or spelling mistakes. It should read smoothly and be easy to follow.

    # Example of an email structure:
    Subject: "I noticed your listings in {{agent_marketing_area_cities}} and wanted to reach out"

    Hi {{agent_name}},

    I noticed your listing at {{agent_marketing_area_cities}}—what a great spot! Being just {{agent_city}} miles from {{try to find a landmark or city close to agent_city}}, it's sure to grab attention.

    At {{find company name from offer}}, we help agents like you by {{find offer value from offer}}, and I'd love to discuss how we can make your workload lighter and more effective.

    Feel free to book a call at your convenience!

    Talk soon,  
    {{signature}}

    # Note
    The above email is just an example, and you can modify it to fulfill the criteria.
    Always avoid being generic and try to be personal and engaging.
    All the details about the agent are in the below section, and you can use it to your advantage.
    
    # Important Notes
    Your email should always be ready to send, meaning you always put the signature and offer and all the details. YOU ARE NOT TO LEAVE ANY PLACEHOLDERS.
    Always end with the provided signature and offer, don't create your own signature or offer.

    # Agent Details
    Agent Name: {agent_name}
    Agent City: {agent_city}
    Agent Address Line: {agent_address_line}
    Agent State: {agent_state}
    Agent Description: {agent_description}
    Agent First Year Active: {agent_first_year_active}
    Agent Marketing Area Cities: {agent_marketing_area_cities}
    Recently Sold Properties Count: {recently_sold_properties_count}
    Recently Sold Last Sold Date: {recently_sold_last_sold_date}
    Agent Slogan: {agent_slogan}
    Agent Specializations: {agent_specializations}
    Agent Title: {agent_title}

    # Offer
    {offer}

    # Email Signature
    {signature}

    # Format Instructions
    {format_instructions}
    """

    # Create the prompt template object
    email_prompt = PromptTemplate(
        input_variables=[
            "agent_name", "agent_city", "agent_address_line", "agent_state", 
            "agent_description", "agent_first_year_active", 
            "agent_marketing_area_cities", "recently_sold_properties_count", 
            "recently_sold_last_sold_date",
            "agent_slogan", "agent_specializations", "agent_title", "offer", "signature"
        ],
        template=email_prompt_template,
        partial_variables={
            "format_instructions": email_parser.get_format_instructions()
        }
    )

    # Process each agent record and generate personalized emails
    personalized_emails = []
    for agent_data in agents_records:  
        try:
            # Extract necessary properties from agent_data
            agent_name = agent_data.get("Agent Full Name") or agent_data.get("Agent Person Name") or ""
            agent_city = agent_data.get("Agent City", "")
            agent_address_line = agent_data.get("Agent Address Line", "")
            agent_state = agent_data.get("Agent State", "")
            agent_description = agent_data.get("Agent Description", "")
            agent_first_year_active = agent_data.get("Agent First Year (Active)", "")
            agent_marketing_area_cities = agent_data.get("Agent Marketing Area Cities", "")
            recently_sold_properties_count = agent_data.get("Recently Sold Properties Count", "")
            recently_sold_last_sold_date = agent_data.get("Recently Sold Last Sold Date", "")
            agent_slogan = agent_data.get("Agent Slogan", "")
            agent_specializations = agent_data.get("Agent Specializations", "")
            agent_title = agent_data.get("Agent Title", "")

            # Invoke the LLM with the prompt
            prompt_with_data = email_prompt.format(
                agent_name=agent_name,
                agent_city=agent_city,
                agent_address_line=agent_address_line,
                agent_state=agent_state,
                agent_description=agent_description,
                agent_first_year_active=agent_first_year_active,
                agent_marketing_area_cities=agent_marketing_area_cities,
                recently_sold_properties_count=recently_sold_properties_count,
                recently_sold_last_sold_date=recently_sold_last_sold_date,
                agent_slogan=agent_slogan,
                agent_specializations=agent_specializations,
                agent_title=agent_title,
                offer=offer,
                signature=signature
            )
            result = llm(prompt_with_data)
            parsed_result = email_parser.parse(result.content)
            email_subject = parsed_result.email_subject.strip()
            email_content = parsed_result.email_content.strip()
            personalized_emails.append({
                'Email': agent_data.get('Agent Email', ''),
                'Email-Subject': email_subject,
                'Email-Content': email_content,
                'Send Button': False,  #TODO: TEST - Assuming default value
                'Status': 'Pending'
            })
        except Exception as e:
            logging.error(f"Failed to generate email for agent {agent_data.get('Agent Email', 'unknown')}: {e}")

    return personalized_emails


# Function to insert emails into NocoDB
def insert_emails_into_nocodb(nocodb_api_url, table_id, api_token, emails_data):
    headers = {
        'xc-token': api_token,
        'Content-Type': 'application/json'
    }

    payload = [email_data for email_data in emails_data]

    response = requests.post(
        f"https://{nocodb_api_url}/api/v2/tables/{table_id}/records",
        headers=headers,
        data=json.dumps(payload)
    )

    if response.status_code != 200:
        logging.error(f"Failed to insert records: {response.text}")
        raise Exception(f"Failed to insert records: {response.text}")

    logging.info("Successfully inserted records into NocoDB.")

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
            logging.error(f"Failed to fetch records: {response.text}")
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

@app.route('/process-agents', methods=['POST'])
def process_agents():
    data = request.json
    AGENT_TABLE_ID = data.get('agent_table_id')
    NOCO_DB_API_URL = os.getenv("NOCO_DB_API_URL")
    NOCO_DB_API_TOKEN = os.getenv("NOCO_DB_API_TOKEN")
    LEADS_ACTION_TABLE = os.getenv("LEADS_ACTION_TABLE")
    OFFER_SIGNATURE_TABLE_ID = os.getenv("OFFER_SIGNATURE_TABLE_ID")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    BATCH_SIZE = 100

    if not AGENT_TABLE_ID:
        return jsonify({'error': 'agent_table_id is required'}), 400  

    agents_records = fetch_agents_details(
        NOCO_DB_API_URL, AGENT_TABLE_ID, NOCO_DB_API_TOKEN, BATCH_SIZE)
    logging.info("Fetched agent records.")

    offer, signature = fetch_latest_offer_and_signature(
        NOCO_DB_API_URL, OFFER_SIGNATURE_TABLE_ID, NOCO_DB_API_TOKEN)
    logging.info("Fetched latest offer and signature.")

    personalized_emails = generate_personalized_emails(
        agents_records, OPENAI_API_KEY, offer, signature)
    logging.info("Generated personalized emails.")

    insert_emails_into_nocodb(
        NOCO_DB_API_URL, LEADS_ACTION_TABLE, NOCO_DB_API_TOKEN, personalized_emails)
    logging.info("Inserted emails into NocoDB.")

    return jsonify({'processed_emails': personalized_emails}), 200

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)))