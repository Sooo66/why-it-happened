import requests
import json
import ast
import regex as re
import loguru
from typing import Optional, Dict, List
from utils import read_file, write_file, write_line, convert_to_json_list, read_jsonl
from tqdm import tqdm
import argparse
from google import genai
import random
import os
import time
from loguru import logger

logger = logger.bind(name=__name__)

prompt_template = '''
You are an expert in timeline summarization. Your task is to create a clear, coherent, and concise timeline of key events based on the provided topic and its associated documents. Adhere strictly to the following guidelines:

1. Event Selection: Each document contains an "events" field. Include only the most significant events that represent critical developments or turning points in the topic. Exclude background information, minor details, or redundant entries.
2. Event Deduplication: Identify and merge multiple descriptions of the same real-world event into a single timeline entry, ensuring no duplication.
3. Event Positioning: For each unique event, record the values of the "position" field from documents where the event appears. DO NOT use the index of the document in the list — use the actual "position" field inside each document.
4. Timeline Scope: Construct a timeline with 10–20 events, depending on the topic’s complexity and the number of documents provided.
5. Chronological Order: Arrange events in strict chronological order to reflect the progression of the topic.
6. Output Structure: For each event, provide the following fields:
    - "event": The exact event description from the document’s "event" field.
    - "position": A list of document position where the event is mentioned.

Output Format Example:
[
    {{"event": "The initial protest broke out in the city center.", "pos ition": [0, 2]}},
    {{"event": "Police forces responded with tear gas.", "position": [1]}},
    ...
]

Input: {text}

Output:
'''

class TimelineSummarizer:
    def __init__(self, input_file: str, output_file: str, model_name: str):
        self.model_name = model_name
        self.input_file = input_file
        self.output_file = output_file
        self.data = [self._load_data()[3]]
        self.api_key = os.getenv("GENAI_API_KEY", "AIzaSyAZGQT_A3yazNt_IqVWuHyWpZCoqz6vt0E")
        self.client = genai.Client(api_key=self.api_key)

    def _load_data(self) -> List[Dict]:
        return read_file(self.input_file)

    def _get_prompt(self, text: str) -> str:
        return prompt_template.format(text=text)

    def _get_response(self, prompt: str) -> Optional[str]:
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            return response.text
        except Exception as e:
            logger.error(f"API call failed: {e}")
            return None

    def _parse(self, response_text: str) -> Optional[List[Dict]]:
        if not response_text:
            return None
        match = re.search(r"\[.*\]", response_text, re.DOTALL)
        if not match:
            logger.error(f"No list found in response: {response_text}")
            return None
        list_text = match.group(0)
        try:
            parsed = ast.literal_eval(list_text)
            if not isinstance(parsed, list):
                return None
            timeline = []
            seen = set()
            for idx, item in enumerate(parsed):
                if not isinstance(item, dict):
                    continue
                event = item.get("event", "").strip()
                pos = item.get("position", [])
                if not event or event in seen:
                    continue
                seen.add(event)
                timeline.append({
                    "event": event,
                    "event_order": len(timeline) + 1,
                    "position": sorted(set(pos))
                })
            return timeline
        except Exception as e:
            logger.error(f"Failed to parse timeline: {e}")
            logger.error(f"List text: {list_text}")
        return None

    def run(self):
        logger.info(f"Starting extraction: {self.input_file} -> {self.output_file}")
        for tpc in tqdm(self.data):
            docs = tpc['docs']
            for doc in docs:
                doc.pop('imageUrl', None)
                doc.pop('content', None)
                doc.pop('link', None)
                doc.pop('source', None)
                doc.pop('date', None)

            input_dict = {
                'topic_id': tpc['topic_id'],
                'topic': tpc['topic'],
                'docs': docs
            }

            prompt = self._get_prompt(str(input_dict))
            # logger.warning(prompt)

            # import sys; sys.exit()
            rps_text = self._get_response(prompt)
            timeline = self._parse(rps_text)

            result = {
                'topic_id': tpc['topic_id'],
                'topic': tpc['topic'],
                'timeline': timeline if timeline else []
            }

            write_line(result, self.output_file)
            time.sleep(random.uniform(1, 5))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract timeline from docs.")
    parser.add_argument('--input_file', type=str, default='../data/raw_docs_events.json')
    parser.add_argument('--output_file', type=str, default='../data/timeline.jsonl')
    parser.add_argument('--model_name', type=str, default='gemini-2.5-flash-preview-05-20')
    args = parser.parse_args()
    TimelineSummarizer(args.input_file, args.output_file, args.model_name).run()

