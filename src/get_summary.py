import requests
import json
import loguru
from typing import Optional, Dict, Any, List
from utils import read_file, write_file, write_line, convert_to_json_list
from tqdm import tqdm
import argparse
from google import genai
import random
import time

prompt = '''
You are an expert in text summarization. Please summarize the provided text according to the following requirements::
1. Clearly identify and fully retain all core events described within the text.
2. Remove unnecessary descriptive details and supplementary information.
3. Ensure the summary is concise yet information-rich to facilitate subsequent event extraction and causal relation annotation.

Text: {text}
'''

class Summarizer:
    def __init__(self, input_file: str, output_file: str, model_name: str):
        self.model_name = model_name
        self.input_file = input_file
        self.output_file = output_file
        self.data = self._load_data()
        self.api_key = "AIzaSyAZGQT_A3yazNt_IqVWuHyWpZCoqz6vt0E"
        self.client = genai.Client(api_key=self.api_key)

    def _load_data(self) -> List[Dict]:
        data = read_file(self.input_file)
        return data
    
    def _get_prompt(self, text: str) -> str:
        return prompt.format(text=text)

    def _summarize(self, text: str) -> Optional[str]:
        content = self._get_prompt(text)

        response = self.client.models.generate_content(
            model="gemini-2.0-flash", contents=content
        )
        return response.text

    def run(self):
        for tpc in tqdm(self.data, desc="Processing topics", unit="topic"):
            try:
                docs = tpc['docs']
                for doc in tqdm(docs, desc="Processing documents", unit="doc"):
                    uuid = doc.get('uuid')
                    text = doc['content']
                    prompt = self._get_prompt(text)
                    summary = self._summarize(prompt)
                    if summary is not None:
                        len_summary = len(summary.split())
                        loguru.logger.info(f"uuid: {uuid}, summary length: {len_summary}")
                    else:
                        loguru.logger.error(f"Failed to summarize document with uuid {uuid}")
                    new_d = {
                        'uuid': uuid,
                        'summary': summary
                    }
                    write_line(new_d, self.output_file)
                    time.sleep(random.uniform(5, 10))
            except KeyError as e:
                loguru.logger.error(f"Missing content key for uuid {uuid}: {e}")
                continue
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize documents.")
    parser.add_argument(
        "--input_file", type=str, default="../data/raw_docs.json", help="Input JSON file with documents."
    )
    parser.add_argument(
        "--output_file", type=str, default="../data/doc_smy_pairs.jsonl", help="Output JSON file for summarized documents."
    )
    parser.add_argument(
        "--model_name", type=str, default="gemini-2.0-flash", help="Model name to be used."
    )
    args = parser.parse_args()
    ipt, opt = args.input_file, args.output_file
    model_name = args.model_name
    loguru.logger.info(f"Input file: {ipt}, Output file: {opt}")

    summarizer = Summarizer(ipt, opt, model_name)
    summarizer.run()