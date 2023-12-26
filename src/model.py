import os
from dotenv import load_dotenv
from genai.extensions.langchain import LangChainInterface
from genai.schemas import GenerateParams
from genai.credentials import Credentials

load_dotenv()
api_key = os.getenv("GENAI_KEY", None)
api_url = os.getenv("GENAI_API", None)

creds = Credentials(api_key, api_endpoint=api_url)

model_id = "meta-llama/llama-2-70b-chat"

params_1 = GenerateParams(
    decoding_method="greedy",
    max_new_tokens=100,
    min_new_tokens=1,
    temperature=0.6,
    stop_sequences=["\n\n"],
    repetition_penalty=1.15,
)

# params_2 = GenerateParams(
#     decoding_method="greedy",
#     max_new_tokens=1024,
#     min_new_tokens=1,
#     temperature=0.4,
#     stop_sequences=["</ANSWER>"],
#     repetition_penalty=1.0,
# )

llm_1 = LangChainInterface(model=model_id, params=params_1, credentials=creds)

# llm_2 = LangChainInterface(model=model_id, params=params_2, credentials=creds)
