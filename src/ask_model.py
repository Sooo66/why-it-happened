import os
import json
import re
import time
import random
import argparse
from tqdm import tqdm
from loguru import logger
from typing import Dict, List, Optional
import openai
from utils import read_jsonl, write_line

logger = logger.bind(name=__name__)

# Prompt template for MCQ solving in English with context
prompt_template = '''
Context:
{context}

Question: {question}
Options:
A. {option_A}
B. {option_B}
C. {option_C}
D. {option_D}

Please select all correct answers based on the context above. Respond with comma-separated letters only (e.g., "A,B").'''

# API provider configurations
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

class McqSolver:
    def __init__(
        self,
        model_name: str,
        input_file: str,
        context_file: str,
        # output_file: str,
        retry_missing: bool = False,
    ):
        self.model_name = model_name
        self.input_file = input_file
        self.context_file = context_file
        self.output_file = f'../data/{model_name.split('/')[-1] if model_name.find('/') else model_name}_mcq_fix.jsonl'
        self.retry_missing = retry_missing

        # determine provider based on model_name prefix
        if model_name.startswith('gemini'):
            self.provider = 'google'
        elif model_name.startswith(('gpt','claude')):
            self.provider = 'v3'
        elif model_name.startswith('glm'):
            self.provider = 'zhipu'
        elif model_name.startswith(('deepseek','qwen')):
            self.provider = 'silicon'
        elif model_name.startswith('meta'):
            self.provider = 'together'
        else:
            self.provider = 'infinite'

        # configure OpenAI client
        openai.api_key = API_KEY[self.provider]
        openai.base_url = BASE_URL[self.provider]
        openai.default_headers = {"x-foo": "true"}

        self.client = openai.OpenAI(
            api_key=API_KEY[self.provider],
            base_url=BASE_URL[self.provider],
        )

        # load previous to avoid duplicates
        self.processed = set()
        if os.path.exists(self.output_file):
            for rec in read_jsonl(self.output_file):
                uid = rec.get('uuid')
                if uid:
                    self.processed.add(uid)

        # load MCQ records
        raw = read_jsonl(self.input_file)
        if self.retry_missing:
            self.data = [r for r in raw if r['uuid'] in self._missing_uuids()]
        else:
            self.data = [r for r in raw if r['uuid'] not in self.processed]

        # load context data (JSON list of {topic_id, docs: [{summary}]})
        self.context_data: Dict[str, List[str]] = {}
        with open(self.context_file, 'r', encoding='utf-8') as f:
            topics = json.load(f)
        for topic in topics:
            tid = topic.get('topic_id')
            summaries = [doc.get('summary','') for doc in topic.get('docs', [])]
            self.context_data[tid] = sorted(set(summaries))

    def _missing_uuids(self) -> set:
        missing = set()
        if not os.path.exists(self.output_file):
            return missing
        for rec in read_jsonl(self.output_file):
            if not rec.get('answer'):
                missing.add(rec['uuid'])
        return missing

    def _get_prompt(self, record: Dict) -> str:
        # fetch context summaries for this topic
        ctx_list = self.context_data.get(record.get('topic_id',''), [])
        context_str = '\n'.join(f"- {s}" for s in ctx_list)
        return prompt_template.format(
            context=context_str,
            question=record['question'],
            option_A=record['option_A'],
            option_B=record['option_B'],
            option_C=record['option_C'],
            option_D=record['option_D'],
        )

    def _get_response(self, prompt: str) -> Optional[str]:
        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"API error: {e}")
            return None

    def _parse_answer(self, text: Optional[str]) -> Optional[str]:
        if not text:
            return None
        letters = re.findall(r"[A-D]", text.upper())
        if not letters:
            logger.error(f"No valid option found in response: {text}")
            return None
        unique = sorted(set(letters), key=lambda x: ord(x))
        return ','.join(unique)

    def run(self):
        for rec in tqdm(self.data, desc="Solving MCQs"):
            prompt = self._get_prompt(rec)
            # logger.warning(prompt)
            # import sys; sys.exit()
            answer_text = self._get_response(prompt)
            answer = self._parse_answer(answer_text)
            out = {'uuid': rec['uuid'], 'answer': answer}
            write_line(out, self.output_file)
            time.sleep(random.uniform(2, 2.5))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Solve MCQs via LLM with context.")
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--input_file', default='../data/mcq_fix.jsonl')
    parser.add_argument('--context_file', default='../data/raw_docs_events.json')
    # parser.add_argument('--output_file', default='../data/mcq_answers.jsonl')
    parser.add_argument('--retry_missing', action='store_true')
    args = parser.parse_args()
    solver = McqSolver(
        model_name=args.model_name,
        input_file=args.input_file,
        context_file=args.context_file,
        # output_file=args.output_file,
        retry_missing=args.retry_missing
    )
    solver.run()
