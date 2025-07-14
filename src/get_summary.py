import loguru
from typing import Optional, Dict, Any, List
from utils import read_file, write_file, write_line, convert_to_json_list
from tqdm import tqdm
import argparse
from llm_base_model import BaseModel # Import BaseModel

loguru.logger = loguru.logger.bind(name=__name__) # Bind logger for consistency

# You are an expert in text summarization. Please summarize the provided text according to the following requirements::
# 1. Clearly identify and fully retain all core events described within the text.
# 2. Remove unnecessary descriptive details and supplementary information.
# 3. Ensure the summary is concise yet information-rich to facilitate subsequent event timeline summarization and causal relation annotation.

class Summarizer(BaseModel): # Inherit from BaseModel
    _PROMPT_TEMPLATE = """
You are an expert in text summarization. Please summarize the provided text with the following requirements:
1. Identify and explicitly retain all core events described in the text. A core event is defined as an action or occurrence involving specific actors and outcomes, which is relevant to the main narrative.
2. Remove unimportant descriptive details, background information, and subjective or supplementary commentary that do not constitute discrete events.
3. Keep the summary concise yet comprehensive enough to preserve all events needed for timeline construction and causal relation annotation.
Output only the summary content itself, without any explanations, headings, or additional text.

Text: {text}
"""

    def __init__(
        self,
        model_name: str,
        input_file: str,
        output_file: str,
        sleep_time: float = 5.0, # Default sleep time
        debug: bool = False,
        **kwargs,
    ):
        super().__init__(model_name, input_file, output_file, sleep_time, debug, **kwargs)
        # BaseModel's __init__ will handle client initialization and data loading
        # self.data will be loaded by BaseModel.run, but we need to override _load_data to flatten it.

    def _load_data(self, type: str = "json") -> List[Dict]:
        """Loads data and flattens the nested 'docs' structure."""
        # Use BaseModel's _load_data to read the initial JSON structure
        # topics_data = super()._load_data(type=type)
        topics_data = read_file(self.input_file)
        
        flattened_docs = []
        for tpc in topics_data:
            if "docs" in tpc and isinstance(tpc["docs"], list):
                for doc in tpc["docs"]:
                    flattened_docs.append(doc)
            else:
                loguru.logger.warning(f"Topic missing 'docs' key or 'docs' is not a list: {tpc['topic_id']}")

            if "dis_T" in tpc and isinstance(tpc["dis_T"], list):
                for doc in tpc["dis_T"]:
                    flattened_docs.append(doc)
            else:
                loguru.logger.warning(f"Topic missing 'dis_T' key or 'dis_T' is not a list: {tpc['topic_id']}")
        return flattened_docs

    def _get_prompt(self, record: Dict) -> str:
        """Generates the prompt string for the LLM based on the record."""
        text = record.get("content", "")
        if not text:
            loguru.logger.error(f"Record missing 'content' key: {record}")
            return ""
        return self._PROMPT_TEMPLATE.format(text=text)

    def _parse_response(self, response_text: Optional[str]) -> Any:
        """Parses the raw LLM response into a structured format."""
        return response_text # Summary is just text

    def _format_output(self, original_record: Dict, parsed_result: Any) -> Dict:
        """Formats the final output record to be written to the output file."""
        uuid = original_record.get("uuid")
        summary = parsed_result
        if summary is not None:
            len_summary = len(summary.split())
            loguru.logger.info(f"uuid: {uuid}, summary length: {len_summary}")
        else:
            loguru.logger.error(f"Failed to summarize document with uuid {uuid}")
        return {"uuid": uuid, "answer": summary}
    
    def _process_record(self, record):
        return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize documents.")
    parser.add_argument(
        "--input_file",
        type=str,
        default="../data/raw_docs.json",
        help="Input JSON file with documents.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="../data/doc_smy_pairs.jsonl",
        help="Output JSON file for summarized documents.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gemini-2.5-flash",
        help="Model name to be used.",
    )
    parser.add_argument(
        "--sleep_time",
        type=float,
        default=0.5,
        help="Time to sleep between API calls to avoid rate limiting.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (e.g., print prompts and exit).",
    )
    args = parser.parse_args()
    
    loguru.logger.info(f"Input file: {args.input_file}, Output file: {args.output_file}")

    summarizer = Summarizer(
        model_name=args.model_name,
        input_file=args.input_file,
        output_file=args.output_file,
        sleep_time=args.sleep_time,
        debug=args.debug,
    )
    summarizer.data = summarizer._load_data() # Manually load data after init
    summarizer.run()
