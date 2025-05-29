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
import openai
import time
from loguru import logger

logger = logger.bind(name=__name__)

prompt_template = '''
You are given two events within the same topic, each with its surrounding context. 
Your task is to assign a causal strength score from 0 to 100, indicating how likely Event 1 is the cause of Event 2.

Scoring guide:
- 0–20   : No causality
- 21–50  : Low causality
- 51–80  : Moderate causality
- 81–100 : High causality

Return **only** the integer score (no extra text).

Topic: 
{topic}

Event 1:
{event1}
Context:
{event1_context}

Event 2:
{event2}
Context:
{event2_context}
'''

class Annotator:
    def __init__(self, model_name: str, input_file: str, output_file: str, type: str):
        self.model_name = model_name
        self.input_file = input_file
        self.output_file = output_file
        self.type = type

        self.provider = ''

        if self.model_name.startswith('gemini'):
            self.provider = 'google'
        elif self.model_name.startswith('gpt') or self.model_name.startswith('claude'):
            self.provider = 'v3'
        elif self.model_name.startswith('glm'):
            self.provider = 'zhipu'
        elif self.model_name.startswith('deepseek') or self.model_name.startswith('qwen'):
            self.provider = 'silicon'
        else:
            self.provider = 'infinite'
            

        API_KEY = {
            'google': 'AIzaSyAZGQT_A3yazNt_IqVWuHyWpZCoqz6vt0E',
            'v3': 'sk-NgMsgNsCtJTemze1EfD16cEaB8Cd44B4B3DcE3B2Dc32C78f',
            'zhipu': '16c5bb5687484c6a9d99d29680eaa688.HT3i9vqtdRKI5NpQ',
            'silicon': 'sk-tbcibchgpckkflhjprvpoeunsqjjiztezbnehachaisegxkc',
            'infinite': 'sk-ensxjjowxajbpmjl'
        }

        BASE_URL = {
            'google': 'https://generativelanguage.googleapis.com/v1beta/openai/',
            'v3': 'https://api.gpt.ge/v1/',
            'zhipu': 'https://open.bigmodel.cn/api/paas/v4/',
            'silicon': 'https://api.siliconflow.cn/v1/',
            'infinite': 'https://cloud.infini-ai.com/maas/v1/'
        }

        self.api_key = API_KEY[self.provider]
        self.base_url = BASE_URL[self.provider]
        self.data = self._load_data()

        openai.base_url = self.base_url
        openai.api_key = self.api_key
        openai.default_headers = {"x-foo": "true"}

        self.output_file = f'../data/{model_name}_causal_score.jsonl'
        self.processed_uuids = set()
        self.none_uuids = set()
        if os.path.exists(self.output_file):
            for rec in read_jsonl(self.output_file):
                if 'uuid' in rec:
                    self.processed_uuids.add(rec['uuid'])
                if rec['score'] is None:
                    self.none_uuids.add(rec['uuid'])

    def _load_data(self):
        if self.type == 'normal':
            return read_jsonl(self.input_file)
        else:
            data = read_jsonl(self.input_file)
            return list(filter(data, lambda x: x['uuid'] in self.none_uuids))
    
    def _get_prompt(self, record: Dict) -> str:
        def format_context(context_list: List[str]) -> str:
            return "\n".join(f"- {c}" for c in context_list)

        return prompt_template.format(
            topic          = record.get('topic', 'N/A'),
            event1         = record['event1'],
            event1_context = format_context(record['event1_context']),
            event2         = record['event2'],
            event2_context = format_context(record['event2_context']),
        )

    def _get_response(self, prompt: str) -> Optional[str]:
        try:
            completion = openai.chat.completions.create(
                model     = self.model_name,
                messages  = [{"role": "user", "content": prompt}],
                temperature = 0.0,
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Failed to get model response: {e}")
            # logger.error(completion.choices[0].message.content.strip())
            return None
        
    def _parse(self, text: Optional[str]) -> Optional[int]:
        """
        Extract the first integer between 0 and 100 from the model's reply.
        Returns None if parsing fails.
        """
        if not text:
            return None

        # find any 1–3 digit number
        match = re.search(r'\b(\d{1,3})\b', text)
        if not match:
            logger.error(f"No numeric score found in response: {text!r}")
            return None

        score = int(match.group(1))
        # clamp to [0, 100]
        score = max(0, min(100, score))
        return score

    def run(self):
        for d in tqdm(self.data, desc='Processing d'):
            if d['uuid'] in self.processed_uuids:
                continue
            prompt = self._get_prompt(d)
            if self.model_name.startswith('qwen'):
                prompt += '\n /no_think'
            # logger.warning(prompt)
            # import sys; sys.exit()
            rps_text = self._get_response(prompt)
            score = self._parse(rps_text)

            new_d = {
                'uuid': d['uuid'],
                'score': score
            }

            write_line(new_d, f'../data/{self.model_name}_causal_score.jsonl')
            self.processed_uuids.add(d['uuid'])

            if self.provider == 'infinite':
                time.sleep(random.uniform(5, 10))
            elif self.provider == 'silicon' or self.provider == 'google':
                time.sleep(random.uniform(3, 5))
            else:
                time.sleep(random.uniform(0, 0.5))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract timeline from docs.")
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--input_file', type=str, default='../data/event_pairs.jsonl')
    parser.add_argument('--output_file', type=str, default='../data/a.jsonl')
    parser.add_argument('--type', type=str, default='normal') # none
    args = parser.parse_args()
    Annotator(args.model_name, args.input_file, args.output_file, args.type).run()

