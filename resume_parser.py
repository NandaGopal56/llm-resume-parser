from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os, json, re
from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional, Dict
from datetime import date
import traceback
from text_extraction import extract_text

# Suppress logging warnings
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

# # load environment variables
load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    raise Exception("GOOGLE_API_KEY not found in the environment")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=3
)
file_path = 'sample resumes/mid-level-software-engineer.pdf'
extracted_text = extract_text(file_path)


class Company(BaseModel):
    '''Details about the candidate's experience at a specific company.'''

    company: Optional[str] = Field(None, description='Company name where candidate worked.')
    duration_from: Optional[str] = Field(None, description='Start date (YYYY-MM-DD) of employment.')
    duration_to: Optional[str] = Field(None, description='End date (YYYY-MM-DD) of employment.')
    role: Optional[str] = Field(None, description='Job role/title in the company.')
    industry_of_the_company: Optional[str] = Field(None, description='Industry or sector of the company.')
    summary_of_work: Optional[List[str]] = Field(None, description='Brief 2-3 sentence summary of key responsibilities, no more than 120 words.')

class UserResumeProfile(BaseModel):
    '''Candidate information, including personal details, skills, and work experience.'''

    name: Optional[str] = Field(None, description='Full name of the candidate.')
    phone: Optional[str] = Field(None, description='Phone number.')
    email: Optional[str] = Field(None, description='Email address.')
    address: Optional[str] = Field(None, description='Current address.')
    total_experience: Optional[float] = Field(None, description='Total work experience in years.')
    skills: Optional[Dict[str, List[str]]] = Field(None, description='Dictionary of skills, grouped by domain.')
    companies: Optional[List[Company]] = Field(None, description='List of past companies with job details.')


# Custom parser
def extract_json(message: AIMessage) -> UserResumeProfile:
    """
    Extracts JSON content from a string where JSON is embedded between json and  tags.

    """
    try:
        text = message.content

        match = re.search(r'```json\n(.*?)```', text, re.DOTALL)

        if match:
            json_str = match.group(1).strip()
            data = json.loads(json_str)
            UserResumeProfile(**data)

            return data
        
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        return None
    
    return None

# Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
           "You are an AI model designed to extract and summarize resume data into a structured JSON format. The input text contains information about a candidate's name, contact details, total experience, skills, and work history. "
            "The goal is to produce a brief, compact summary that captures key details accurately."
            "\n\n**Guidelines:**"
            "- Parse dates in `YYYY-MM-DD` format."
            "- If any values are missing or not provided, use `null`."
            "- For each company, limit the `summary_of_work` to 2-3 sentences, keeping it concise (50-120 words). Focus on key responsibilities and avoid excessive detail."
            "- Ensure that `industry_of_the_company` is included for each job experience strictly."
            "- Ensure that `role` is included for each job experience strictly."
            "- If the input does not resemble a resume (e.g., lacks name, contact info, or job details), the output should be completely empty."
           
            "Output your answer in JSON format such that the schema of output json will match with the below given json schema: \n{schema}\n."
            "Do not include the sample JSON schema in the output and only respond with the summarized content in the JSON format"
        ),
        ("human", "{extracted_text}"),
    ]
).partial(schema=UserResumeProfile.model_json_schema())

prompt_output = prompt.format_prompt(extracted_text=extracted_text).to_string()

output_content = llm.invoke(prompt_output)

extracted_resume = extract_json(output_content)

print(json.dumps(extracted_resume))