import json
import ast
import regex as re
import loguru
from typing import Optional, Dict, List, Any
from utils import read_jsonl, write_line, read_file # Keep read_file for now if it's used elsewhere in utils
from tqdm import tqdm
import argparse
import os
import time
from loguru import logger
from llm_base_model import BaseModel # Import BaseModel only

logger = logger.bind(name=__name__)

DEFAULT_PROMPT_TEMPLATE = '''
You are an AI expert specializing in timeline summarization and event disambiguation. Your task is to analyze a set of documents, each containing event descriptions, and generate a single, deduplicated, and chronologically accurate timeline of key events.
Adhere strictly to the following workflow and core principles to complete this task.

## Workflow
Step 1: Event Identification & Chronological Sorting
- Extract All Events: Iterate through all documents and extract every event description from the "events" field.
- Parse Temporal Information: From each event description, carefully parse explicit dates, times, and relative order cues (e.g., "afterward," "the next day," "prior to this"). This is the foundation for the timeline.
- Initial Sort: Based on the parsed temporal information, perform a strict preliminary sort of all events. If an event lacks a specific timestamp, infer its position in the sequence from the context. This is the most critical step to ensure the final output's order is accurate.

Step 2: Event Normalization & Deduplication
- Define Core Events: Treat each event as an "atomic event" that contains a clear, distinct action. A good event description should be "A did B," not "Because A did B, C happened."
- Semantic Grouping: Group together event descriptions that refer to the exact same real-world occurrence, even if they are phrased differently. For example, "The parent company announced the acquisition plan" and "The firm was acquired by its parent company" should be treated as the same event.
- Select a Representative Event: For each event group, you must select the clearest, most complete, and most representative event description from the source documents to be the canonical version.
- DO NOT Synthesize New Descriptions: You are forbidden from creating or rephrasing event descriptions. You must use the original text.
- Correct Flawed Descriptions: If an original description violates the "atomic event" principle (e.g., it describes a cause-and-effect relationship), prioritize selecting another description from the same group that conforms to the principle. If none exists, minimally rephrase it to align with the atomic action rule.

Step 3: Key Event Selection & Construction
- Filter for Significant Events: From the deduplicated and sorted list, filter for the 10-20 most critical turning points or milestone events. Trivial, overly detailed, or purely background events should be excluded.
- Aggregate Position Data: For each selected key event, collect the "position" values from all original documents that mentioned that event (including its semantic variations) and compile them into a list.
- Generate Final Output: Construct the final timeline according to the specified JSON format.

## Core Principles
- Chronological Fidelity: The accuracy of the timeline's order is the highest priority. No decision should compromise the natural sequence of events.
- Source Text Integrity: Never invent new phrasing. The final "event" field must originate from the source documents, with the only exception being minimal corrections to align with the "atomic event" definition.
- Atomic Events: Each event must represent a single, specific action or occurrence. Avoid including causation, explanations, or multiple actions within a single event.

## Output Format
[
    {{"event": "The initial protest broke out in the city center.", "pos ition": [0, 2]}},
    {{"event": "Police forces responded with tear gas.", "position": [1]}},
    ...
]

## Input:
{text}

## Output:
'''

class TimelineSummarizer(BaseModel): # Inherit from BaseModel
    def __init__(self, input_file: str, output_file: str, model_name: str, sleep_time: float, debug: bool):
        super().__init__(model_name, input_file, output_file, sleep_time, debug) # Call parent constructor
        self.prompt_template = DEFAULT_PROMPT_TEMPLATE
        self.data = self._load_data()

    def _load_data(self) -> List[Dict]:
        return read_file(self.input_file)

    def _get_prompt(self, record: Dict) -> str:
        return self.prompt_template.format(text=str(record))

    # Removed _get_response as it's now handled by BaseModel

    def _parse_response(self, response_text: str) -> Optional[List[Dict]]:
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

    def _process_record(self, record: Dict) -> Dict:
        """Pre-processes the record by popping unnecessary keys from docs."""
        docs = record['docs']
        for doc in docs:
            doc.pop('imageUrl', None)
            doc.pop('snippet', None)
            doc.pop('content', None)
            doc.pop('link', None)
            doc.pop('source', None)
            doc.pop('date', None)
            doc.pop('ori_content', None)
        
        return {
            'topic_id': record['topic_id'],
            'topic': record['topic'],
            'docs': docs
        }

    def _format_output(self, original_record: Dict, parsed_result: Any) -> Dict:
        return {
            'topic_id': original_record['topic_id'],
            'topic': original_record['topic'],
            'timeline': parsed_result if parsed_result else []
        }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract timeline from docs.")
    parser.add_argument('--input_file', type=str, default='../sample_data/raw_data.json')
    parser.add_argument('--output_file', type=str, default='../sample_data/timeline.jsonl')
    parser.add_argument('--model_name', type=str, default='gpt-4.5-preview-2025-02-27')
    parser.add_argument('--sleep_time', type=float, default=2.0, help="Sleep time between requests.")
    parser.add_argument('--debug', action='store_true', help="Enable debug mode for more")
    args = parser.parse_args()
    TimelineSummarizer(args.input_file, args.output_file, args.model_name, args.sleep_time, args.debug).run()
