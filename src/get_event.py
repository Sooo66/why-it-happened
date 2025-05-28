import requests
import json
import ast
import regex as re
import loguru
from typing import Optional, Dict, Any, List
from utils import read_file, write_file, write_line, convert_to_json_list, read_jsonl
from tqdm import tqdm
import argparse
# import google generative AI client correctly
from google import genai
import random
import os
import time

prompt_template = '''
You are an expert in event extraction. Extract up to 3 core events from the provided summarized text, defined as specific actions, occurrences, or state changes involving entities. Follow these requirements:
1. List each event concisely, including:
    - Main action or occurrence.
    - Key entities involved.
    - Relevant temporal or contextual details.
2. Exclude non-event details (e.g., opinions, background) unless critical for context.
3. Capture the most significant core events, avoiding redundancy.
4. Limit the output to a maximum of 3 events.

Input Text: {text}

Output Format: ["event1", "event2", ...]'''

class EventExtractor:
    def __init__(self, input_file: str, output_file: str, model_name: str):
        self.model_name = model_name
        self.input_file = input_file
        self.output_file = output_file
        self.data = self._load_data()
        self.api_key = os.getenv("GENAI_API_KEY", "AIzaSyAZGQT_A3yazNt_IqVWuHyWpZCoqz6vt0E")
        self.client = genai.Client(api_key=self.api_key)

    def _load_data(self) -> List[Dict]:
        return read_jsonl(self.input_file)

    def _get_prompt(self, text: str) -> str:
        return prompt_template.format(text=text)

    def _get_response(self, text: str) -> Optional[str]:
        content = self._get_prompt(text)
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=content
            )
            return response.text
        except Exception as e:
            loguru.logger.error(f"API call failed: {e}")
            return None

    def parse(self, response_text: str) -> Optional[List[str]]:
        if not response_text:
            return None
        # extract the first python list in the response
        match = re.search(r"\[.*\]", response_text, re.DOTALL)
        if not match:
            loguru.logger.error(f"No list found in response: {response_text}")
            return None
        list_text = match.group(0)
        try:
            events = ast.literal_eval(list_text)
            if isinstance(events, list):
                # ensure strings
                return [str(e).strip() for e in events][:3]
        except Exception as e:
            loguru.logger.error(f"Failed to parse events list: {e}")
            loguru.logger.error(f"List text: {list_text}")
        return None

    def run(self):
        loguru.logger.info(f"Starting extraction: {self.input_file} -> {self.output_file}")
        for d in tqdm(self.data, desc='Processing data', unit='d'):
            uuid = d['uuid']
            text = d['summary']
            if not text:
                loguru.logger.warning(f"Empty summary for uuid {uuid}")
                continue
            rps_text = self._get_response(text)
            events = self.parse(rps_text)
            if events is not None:
                loguru.logger.info(f"uuid: {uuid}, events cnt: {len(events)}")
            else:
                loguru.logger.error(f"Failed to get events for uuid {uuid}")
            new_d = {
                'uuid': uuid,
                'events': events
            }
            write_line(new_d, self.output_file)
            time.sleep(random.uniform(1, 5))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract core events from documents.")
    parser.add_argument(
        "--input_file", type=str, default="../data/doc_smy_pairs.jsonl", help="Input JSONL file with documents."
    )
    parser.add_argument(
        "--output_file", type=str, default="../data/doc_event_pairs.jsonl", help="Output JSONL file for extracted events."
    )
    parser.add_argument(
        "--model_name", type=str, default="gemini-2.0-flash", help="Model name to use."
    )
    args = parser.parse_args()

    event_extractor = EventExtractor(
        input_file=args.input_file,
        output_file=args.output_file,
        model_name=args.model_name
    )
    event_extractor.run()
