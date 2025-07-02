import os
import json
import re
import time
import random
import argparse
from tqdm import tqdm
from loguru import logger
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
import openai # Import OpenAI
from google import genai # Import Google GenAI
from google.genai import types

# Assuming utils.py is in the same directory or accessible via PYTHONPATH
from utils import read_jsonl, write_line, read_file

logger = logger.bind(name=__name__)

# Common API provider configurations
# These can be moved to a separate config file if preferred for better separation of concerns
API_KEY = {
    'google':    'AIzaSyAZGQT_A3yazNt_IqVWuHyWpZCoqz6vt0E',
    'v3':        'sk-NgMsgNsCtJTemze1EfD16cEaB8Cd44B4B3DcE3B2Dc32C78f',
    'zhipu':     '16c5bb5687484c6a9d99d29680eaa688.HT3i9vqtdRKI5NpQ',
    'silicon':   'sk-tbcibchgpckkflhjprvpoeunsqjjiztezbnehachaisegxkc',
    'together':  'd7c8dc7a66dfefa1803e3a794feb24af065774cab4ec21f346960492db6e89f9',
    'infinite':  'sk-ensxjjowxajbpmjl'
}
BASE_URL = {
    'google':    'https://generativelanguage.googleapis.com/v1beta/openai/',
    'v3':        'https://api.gpt.ge/v1/',
    'zhipu':     'https://open.bigmodel.cn/api/paas/v4/',
    'silicon':   'https://api.siliconflow.cn/v1/',
    'together':  'https://api.together.xyz/v1',
    'infinite':  'https://cloud.infini-ai.com/maas/v1/'
}

class BaseModel(ABC):
    def __init__(self, model_name: str, input_file: str, output_file: str, sleep_time: float, debug: bool, **kwargs):
        self.model_name = model_name
        self.input_file = input_file
        self.output_file = output_file
        self.sleep_time = sleep_time
        self.debug = debug
        self.kwargs = kwargs

        self.client = self._initialize_client()
        self.data = self._load_data()
        self.processed_uids = self._load_processed_uids()

        random.seed(42) # for reproducibility, common across all

    def _initialize_client(self):
        """Initializes the LLM client based on model_name."""
        if self.model_name.startswith('gemini'):
            # Google GenAI client
            api_key = os.getenv("GENAI_API_KEY", API_KEY['google'])
            return genai.Client(api_key=api_key)
        elif self.model_name.startswith(('gpt','claude', 'glm', 'qwen', 'deepseek', 'meta', 'Llama')):
            # OpenAI-compatible client (including Zhipu, Siliconflow, Together, Infinite)
            if self.model_name.startswith('glm'):
                provider = 'zhipu'
            elif self.model_name.startswith(('qwen', 'deepseek')):
                provider = 'silicon'
            elif self.model_name.startswith('meta') or self.model_name.startswith('Llama'):
                provider = 'together'
            elif self.model_name.startswith(('gpt','claude')):
                provider = 'v3'
            else:
                provider = 'infinite' # Default for others

            return openai.OpenAI(
                api_key=API_KEY[provider],
                base_url=BASE_URL[provider],
            )
        else:
            raise ValueError(f"Unsupported model name prefix: {self.model_name}")

    @abstractmethod
    def _load_data(self, type: str = 'jsonl') -> List[Dict]:
        # if type == 'json':
        #     return read_file(self.input_file)
        # elif type == 'jsonl':
        #     return read_jsonl(self.input_file)
        # else:
        #     raise ValueError(f"Unsupported data type: {type}. Use 'json' or 'jsonl'.")
        pass

    def _load_processed_uids(self) -> set:
        """Loads UIDs of already processed records from the output file."""
        processed = set()
        if os.path.exists(self.output_file):
            for rec in read_jsonl(self.output_file):
                # Handle different ID fields (uuid for mcq, topic_id for timeline)
                uid = rec.get('uuid') or rec.get('topic_id')
                if uid:
                    processed.add(uid)
        return processed

    @abstractmethod
    def _get_prompt(self, record: Dict) -> str:
        """Generates the prompt string for the LLM based on the record."""
        pass

    def _get_response(self, prompt: str) -> Optional[str]:
        """Sends the prompt to the LLM and returns the raw response text."""
        try:
            if isinstance(self.client, openai.OpenAI):
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                return resp.choices[0].message.content.strip()
            elif isinstance(self.client, genai.Client):
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.0
                    )
                )
                return response.text
            else:
                raise TypeError("Unsupported LLM client type.")
        except Exception as e:
            logger.error(f"API error: {e}")
            return None

    @abstractmethod
    def _parse_response(self, response_text: Optional[str]) -> Any:
        """Parses the raw LLM response into a structured format."""
        pass

    @abstractmethod
    def _format_output(self, original_record: Dict, parsed_result: Any) -> Dict:
        """Formats the final output record to be written to the output file."""
        pass

    def _process_record(self, record: Dict) -> Dict:
        """Performs any pre-processing on a record before generating the prompt.
        Can be overridden by subclasses."""
        return record

    def run(self):
        """Main method to run the LLM processing pipeline."""
        data_to_process = []
        for d in self.data:
            uid = d.get('uuid') or d.get('topic_id')
            if uid and uid not in self.processed_uids:
                data_to_process.append(d)
            elif not uid:
                logger.warning(f"Record missing UUID/Topic ID: {d}")
                data_to_process.append(d) # Process if no UID to track

        # random.shuffle(data_to_process) # Optional: shuffle for distributed processing

        logger.info(f"Starting processing: {len(data_to_process)} records to process.")
        for rec in tqdm(data_to_process, desc="Processing data"):
            processed_rec = self._process_record(rec)
            prompt = self._get_prompt(processed_rec)
            if self.debug:
                import sys
                logger.debug(prompt)
                sys.exit()
            answer_text = self._get_response(prompt)
            parsed_result = self._parse_response(answer_text)
            
            output_data = self._format_output(rec, parsed_result)
            write_line(output_data, self.output_file)
            time.sleep(random.uniform(self.sleep_time - 1, self.sleep_time + 1)) # Common rate limiting
