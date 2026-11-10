"""Exact original synthetic MockTransport helpers, no real-corpus tests or credentials."""
import json
from openalex_agent.config import Settings

def settings():
    return Settings(llm_base_url='https://api.openai.com/v1',llm_api_key='SYNTHETIC_OFFLINE_SECRET',llm_model='gpt-6.1-terra',llm_reasoning_effort='low',llm_temperature=None,llm_max_tokens=4096,llm_timeout=120,llm_input_price=1,llm_output_price=5)

class OfflinePermit:
    """Only used with httpx.MockTransport; never a user approval document."""
    def check(self):pass

def envelope(output,usage=None):
    return {'id':'synthetic-response','model':'gpt-6.1-terra','service_tier':'default','choices':[{'message':{'content':json.dumps(output)},'finish_reason':'stop'}],
        'usage':usage if usage is not None else {'prompt_tokens':100,'completion_tokens':20,'total_tokens':120,'completion_tokens_details':{'reasoning_tokens':3}}}
