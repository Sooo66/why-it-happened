import os
import json
import re
import time
import random
import argparse
from tqdm import tqdm
from loguru import logger
from typing import Dict, List, Optional, Any
from utils import read_jsonl, write_line
from prompt_template import icl_prompt, icl_cot_prompt
from llm_base_model import BaseModel  # Import BaseModel only
from nltk.tokenize import sent_tokenize, word_tokenize

logger = logger.bind(name=__name__)

# Prompt template for MCQ solving in English with context
DEFAULT_PROMPT_TEMPLATE = """
Context:
{context}

Question: {question}
Options:
A. {option_A}
B. {option_B}
C. {option_C}
D. {option_D}

Please select all correct answers based on the context above. Respond with comma-separated letters only (e.g., "A,B")."""


class McqSolver(BaseModel):  # Inherit from BaseModel
    def __init__(
        self,
        model_name: str,
        input_file: str,
        context_file: str,
        prompt: str,
        context_type: str,
        retry_missing: bool = False,
        sleep_time: float = 1.0,
        debug: bool = False,
    ):
        output_file = f'../sample_data/pred_data/{context_type}/{model_name.split("/")[-1] if model_name.find("/") else model_name}_{prompt}.jsonl'
        super().__init__(model_name, input_file, output_file, sleep_time, debug)

        self.context_file = context_file
        self.retry_missing = retry_missing
        self.prompt_type = prompt
        self.context_type = context_type
        self.prompt_template = DEFAULT_PROMPT_TEMPLATE
        self.processed_uids = self._load_processed_uids()  # Initialize processed_uids

        if self.prompt_type.startswith("icl"):
            self.prompt_template = (
                icl_prompt if self.prompt_type == "icl" else icl_cot_prompt
            )
            ice_uuid = [
                "dc232094-0dac-4c8e-8de3-95488ac8f8e0",
                "2ab218c1-6513-4261-89b9-9795b0805d28",
                "c7e7d73e-6d12-410f-852d-ed5e52c12d45",
            ]
            for _ in ice_uuid:
                self.processed_uids.add(_)  # Use processed_uids from BaseModel
        

        self.data = self._load_data()  # Load data using BaseModel's method
        self._load_context_data()  # Load context data specific to McqSolver

    def _load_data(self) -> List[Dict]:
        # BaseModel already loaded self.data from self.input_file
        # if self.retry_missing:
        #     # Re-filter data based on missing answers
        #     missing_uuids = self._missing_uuids()
        #     data = [r for r in self.data if r["uuid"] in missing_uuids]
        # else:
        #     # Filter out already processed UIDs
        #     data = [r for r in self.data if r["uuid"] not in self.processed_uids]
        # return data # Return the filtered data
        data = read_jsonl(self.input_file)
        if self.retry_missing:
            missing_uuids = self._missing_uuids()
            data = [r for r in data if r["uuid"] in missing_uuids]
        else:
            data = [r for r in data if r["uuid"] not in self.processed_uids]
        logger.info(f"Loaded {len(data)} records from {self.input_file}")
        return data

    def _load_context_data(self):
        """Loads context data specific to MCQ solving."""
        self.context_data = {}
        with open(self.context_file, "r", encoding="utf-8") as f:
            topics = json.load(f)
        for topic in topics:
            tid = topic.get("topic_id")
            summaries = []
            if self.context_type == "smy":
                summaries = [doc.get("summary", "") for doc in topic.get("docs", [])]
            elif self.context_type == "ori":
                summaries = [
                    doc.get("ori_content", "") for doc in topic.get("docs", [])
                ]
            elif self.context_type == "dis":
                summaries = []

                for doc in topic.get("dis_T", []):
                    content = doc.get("content", "")
                    sentences = sent_tokenize(content)
                    tokens_so_far = 0
                    selected_sentences = []

                    for sent in sentences:
                        tokens_in_sent = len(word_tokenize(sent))
                        if tokens_so_far + tokens_in_sent > 5000:
                            break
                        selected_sentences.append(sent)
                        tokens_so_far += tokens_in_sent

                    truncated_content = " ".join(selected_sentences)
                    summaries.append(truncated_content)

                # 原 docs 部分保持不变
                summaries += [
                    doc.get("ori_content", "") for doc in topic.get("docs", [])
                ]
            self.context_data[tid] = sorted(set(summaries))
            random.shuffle(self.context_data[tid])
            logger.info(f"Loaded {len(self.context_data[tid])} for topic {tid}")

    def _missing_uuids(self) -> set:
        missing = set()
        if not os.path.exists(self.output_file):
            return missing
        for rec in read_jsonl(self.output_file):
            if not rec.get("answer"):
                missing.add(rec["uuid"])
        return missing
    
    def _load_processed_uids(self):
        """Loads UIDs of already processed records from the output file."""
        processed = set()
        if os.path.exists(self.output_file):
            for rec in read_jsonl(self.output_file):
                uid = rec.get("uuid")
                answer = rec.get("answer")
                if answer:
                    processed.add(uid)
        return processed

    def _get_prompt(self, record: Dict) -> str:
        # fetch context summaries for this topic
        ctx_list = self.context_data.get(record.get("topic_id", ""), [])
        context_str = "\n".join(f"- {s}" for s in ctx_list)
        return self.prompt_template.format(
            context=context_str,
            question=record["question"],
            option_A=record["option_A"],
            option_B=record["option_B"],
            option_C=record["option_C"],
            option_D=record["option_D"],
        )

    # Removed _get_response as it's now handled by BaseModel

    def _parse_response(self, text: Optional[str]) -> Optional[str]:
        if not text:
            return None

        if self.prompt_type == "icl_cot":
            match = re.search(r"Final Answer:\s*\"?([A-D](?:,[A-D])*)", text)
            if match:
                answers = match.group(1).split(",")
                unique = sorted(set(ans.strip().upper() for ans in answers))
                if not unique:
                    logger.error(f"No valid option found in response: {text}")
                    return None
                return ",".join(unique)

        letters = re.findall(r"[A-D]", text.upper())
        if not letters:
            logger.error(f"No valid option found in response: {text}")
            return None
        unique = sorted(set(letters), key=lambda x: ord(x))
        return ",".join(unique)

    def _format_output(self, original_record: Dict, parsed_result: Any) -> Dict:
        return {"uuid": original_record["uuid"], "answer": parsed_result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve MCQs via LLM with context.")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--input_file", default="../sample_data/mcq_0710.jsonl")
    parser.add_argument("--context_file", default="../sample_data/raw_data.json")
    parser.add_argument("--prompt", default="DIO")
    parser.add_argument(
        "--context_type",
        default="smy",
        choices=["smy", "ori", "dis"],
        help="Type of context to use.",
    )
    parser.add_argument("--retry_missing", action="store_true")
    parser.add_argument("--sleep_time", type=float, default=0.5, help="Sleep time between API calls.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode to print prompt and exit.")
    args = parser.parse_args()
    solver = McqSolver(
        model_name=args.model_name,
        input_file=args.input_file,
        context_file=args.context_file,
        prompt=args.prompt,
        context_type=args.context_type,
        retry_missing=args.retry_missing,
        sleep_time=args.sleep_time,
        debug=args.debug,
    )
    solver.run()
