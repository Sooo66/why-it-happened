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
import openai  # Import OpenAI
from google import genai  # Import Google GenAI
from google.genai import types

# Assuming utils.py is in the same directory or accessible via PYTHONPATH
from utils import read_jsonl, write_line, read_file, write_file, write_jsonl

logger = logger.bind(name=__name__)

# Common API provider configurations
# These can be moved to a separate config file if preferred for better separation of concerns
API_KEY = {
    "google": "AIzaSyAZGQT_A3yazNt_IqVWuHyWpZCoqz6vt0E",
    "v3": "sk-NgMsgNsCtJTemze1EfD16cEaB8Cd44B4B3DcE3B2Dc32C78f",
    "zhipu": "16c5bb5687484c6a9d99d29680eaa688.HT3i9vqtdRKI5NpQ",
    "silicon": "sk-tbcibchgpckkflhjprvpoeunsqjjiztezbnehachaisegxkc",
    "together": "d7c8dc7a66dfefa1803e3a794feb24af065774cab4ec21f346960492db6e89f9",
    "infinite": "sk-ensxjjowxajbpmjl",
}
BASE_URL = {
    "google": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "v3": "https://api.gpt.ge/v1/",
    "zhipu": "https://open.bigmodel.cn/api/paas/v4/",
    "silicon": "https://api.siliconflow.cn/v1/",
    "together": "https://api.together.xyz/v1",
    "infinite": "https://cloud.infini-ai.com/maas/v1/",
}


class BaseModel(ABC):
    def __init__(
        self,
        model_name: str,
        input_file: str,
        output_file: str,
        sleep_time: float,
        debug: bool,
        **kwargs,
    ):
        self.model_name = model_name
        self.input_file = input_file
        self.output_file = output_file
        self.sleep_time = sleep_time
        self.debug = debug
        self.kwargs = kwargs

        self.data = None
        self.processed_uids = set()
        self.none_uuid = set()

        self.client = self._initialize_client()
        self._load_processed_uids()

        # random.seed(42)  # for reproducibility, common across all

    def _initialize_client(self):
        """Initializes the LLM client based on model_name."""
        if self.model_name.startswith("gemini") and self.model_name.endswith("2.0"):
            api_key = os.getenv("GENAI_API_KEY", API_KEY["google"])
            return genai.Client(api_key=api_key)
        elif self.model_name.startswith(
            ("gpt", "claude", "glm", "qwen", "deepseek", "meta", "Llama", 'gemini')
        ):
            # OpenAI-compatible client (including Zhipu, Siliconflow, Together, Infinite)
            if self.model_name.startswith("glm"):
                provider = "zhipu"
            elif self.model_name.startswith(("qwen", "deepseek")):
                provider = "silicon"
            elif self.model_name.startswith("meta") or self.model_name.startswith(
                "Llama"
            ):
                provider = "together"
            elif self.model_name.startswith(("gpt", "claude", "gemini")):
                provider = "v3"
            else:
                provider = "infinite"  # Default for others

            return openai.OpenAI(
                api_key=API_KEY[provider],
                base_url=BASE_URL[provider],
            )
        else:
            raise ValueError(f"Unsupported model name prefix: {self.model_name}")

    @abstractmethod
    def _load_data(self) -> List[Dict]:
        pass

    def _load_processed_uids(self) -> set:
        """Loads UIDs of already processed records from the output file."""
        logger.debug(f"here: {self.output_file}")
        if os.path.exists(self.output_file):
            logger.debug(f"here2")
            for rec in read_jsonl(self.output_file):
                # Handle different ID fields (uuid for mcq, topic_id for timeline)
                uid = rec.get("uuid") or rec.get("topic_id")
                # ans = rec.get("answer")
                ans = list(rec.items())[1][1]
                # logger.debug(f"Processing record: {uid}, answer: {ans}")
                # import sys; sys.exit()
                if uid and ans:
                    self.processed_uids.add(uid)
                elif uid and not ans:
                    self.none_uuid.add(uid)
            logger.info(f"Loaded {len(self.processed_uids)} processed UIDs from {self.output_file}.")
    
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
                    temperature=0.7,
                )
                return resp.choices[0].message.content.strip()
            elif isinstance(self.client, genai.Client):
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=0.7),
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
    
    @abstractmethod
    def _process_record(self, record: Dict) -> Dict:
        """Performs any pre-processing on a record before generating the prompt.
        Can be overridden by subclasses."""
        pass
    
    def _update_none_uuid(self, output_data: Dict):
        data = read_jsonl(self.output_file)
        for d in data:
            if d['uuid'] == output_data['uuid']:
                d['answer'] = output_data['answer']
                break
        write_jsonl(data, self.output_file)

    def run(self):
        """Main method to run the LLM processing pipeline."""
        data_to_process = []
        for d in self.data:
            uid = d.get("uuid") or d.get("topic_id")
            if uid and uid not in self.processed_uids:
                data_to_process.append(d)

        # random.shuffle(data_to_process) # Optional: shuffle for distributed processing

        import random
        # random.shuffle(data_to_process)
        logger.info(f"Starting processing: {len(data_to_process)} records to process.")
        for rec in tqdm(data_to_process, desc="Processing data"):
            # processed_rec = self._process_record(rec)
            rec = self._process_record(rec)
            prompt = self._get_prompt(rec)
            if self.debug:
                import sys

                logger.debug(prompt)
                sys.exit()
            answer_text = self._get_response(prompt)
            parsed_result = self._parse_response(answer_text)

            output_data = self._format_output(rec, parsed_result)
            uid = rec.get("uuid") or rec.get("topic_id")
            if uid in self.none_uuid:
                # self._update_none_uuid(output_data)
                pass
            else:
                write_line(output_data, self.output_file)
            time.sleep(
                random.uniform(self.sleep_time - 0.5, self.sleep_time + 0.5)
            )  # Common rate limiting
