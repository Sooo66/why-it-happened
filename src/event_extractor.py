import json
import ast
import regex as re
import loguru
from typing import Optional, Dict, Any, List
from utils import read_jsonl, write_line, read_file
from tqdm import tqdm
import argparse
import os
import time
from llm_base_model import BaseModel  # Import BaseModel only

loguru.logger = loguru.logger.bind(name=__name__)  # Use loguru.logger directly

DEFAULT_PROMPT_TEMPLATE = """
You are an expert in event extraction. Extract up to 5 core, atomic events from the provided text.
An atomic event is defined as a single, objective action or state change involving identifiable entities, without combining multiple actions or including speculation.

Extraction Guidelines:
1. Be concise and objective  
   - Describe only what factually happened or is a definite, confirmed action or state change.  
   - Exclude any opinions, predictions, expectations, or speculative language (e.g., "may," "expected to," "likely",  "think").

2. Capture atomic structure  
   - Each event must contain only one main action or change.  
   - Avoid combining multiple events into a single sentence.

3. Include key entities  
   - Clearly state who/what is involved in the event.  
   - Include relevant temporal or contextual information if mentioned.

4. Avoid redundancy  
   - Ensure each event is distinct and does not repeat the content of another.
   - Different statements about the same event should be combined into one.

5. Limit to 5 events  
   - Extract only no more the 5 most significant events, based on factual impact and clarity.

Output Format:  
["event1", "event2", "event3", ...]

Example 1:
input: "Intelligence exchanges primarily occur between individual nation states in Europe, outside of EU jurisdiction. The UK possesses unique intelligence capabilities due to its partnerships with the U.S. and the "Five Eyes" alliance. The UK drafted the EU Counter Terrorism Strategy in 2005 and influenced EU initiatives like the Europol Counter Terrorism Centre (CTC) and the PNRD. The UK's intelligence-led counter terrorism policing concept integrates intelligence agencies with the police. The EAW has increased extradition speed between EU states, and Europol/Eurojust facilitate police/judicial cooperation. The UK joined the Schengen Information System for information sharing. Post-Brexit, EU states are expected to continue counter-terrorism cooperation with the UK, and the UK will likely seek access to EU law enforcement databases. The UK will presumably remain in the non-EU Counter Terrorist Group (CTG). The UK will need to negotiate new arrangements, potentially through bilateral agreements. Officials acknowledge potential limits compared to the pre-Brexit situation due to precedents. The negotiation process may hinder counter-terrorism efforts. The UK government will aim to retain close counter-terrorism cooperation. Brexit could exacerbate tensions in Northern Ireland and may encourage violent republican dissidents if they can build grassroots support.\n"
output: ['The UK drafted the EU Counter Terrorism Strategy in 2005.', 'The UK influenced EU initiatives, including the Europol Counter Terrorism Centre (CTC) and the PNRD', 'The UK joined the Schengen Information System.']

Example 2:
input: "Trump supporters protested in Washington, D.C., to reject the election results. Trump addressed the crowd, urging them to protest before they marched to the Capitol. Protesters clashed with police and breached security barriers, prompting the House and Senate to recess and the Capitol to be locked down. D.C. Mayor Bowser ordered a citywide curfew. Proud Boys marched towards the Capitol building, with some chanting "storm the Capitol" and "1776!". Police fired gas cannisters.\n"
output: ['Trump supporters protested in Washington, D.C.', 'Trump addressed supporters in Washington, D.C.', 'Protesters clashed with police and breached security barriers at the Capitol.
', 'The House and Senate recessed, and the Capitol was locked down.']

Example 3:
input: "Former Japanese Prime Minister Shinzo Abe was assassinated at a campaign rally. Security arrested the suspected gunman, Tetsuya Yamagami. Yamagami fired two shots at Abe; the second shot hit Abe's chest and neck. Abe died several hours later. Yamagami attacked Abe due to Abe's association with a group Yamagami hated. Tributes were given by President Biden, Secretary of State Antony Blinken, Indian Prime Minister Narendra Modi, Russian President Vladimir Putin, and former Australian Prime Minister Tony Abbott. Abe worked to build up Japan's military and reform the economy. Abe failed to revise Japan's constitution but passed legislation allowing Japan's military to expand operations overseas. Abe was a critic of China and called on the US to give Taiwan assurances of help in the event of an attack by China.\n"
output: ['Former Japanese Prime Minister Shinzo Abe was assassinated at a campaign rally.', 'Security arrested the suspected gunman, Tetsuya Yamagami.', 'Leaders of many countries expressed their condolences to Abe']

input: "{text}"
output:
"""


class EventExtractor(BaseModel):  # Inherit from BaseModel
    def __init__(
        self,
        input_file: str,
        output_file: str,
        model_name: str,
        sleep_time: float,
        debug: bool,
    ):
        super().__init__(
            model_name, input_file, output_file, sleep_time, debug
        )  # Call parent constructor
        self.prompt_template = DEFAULT_PROMPT_TEMPLATE

    def _load_data(self):
        raw_data = read_file(self.input_file)
        data = [
            {"uuid": record["uuid"], "summary": record["summary"]}
            for d in raw_data
            for record in d["docs"]
        ]
        return data

    def _get_prompt(self, record: Dict) -> str:
        text = record["summary"]
        return self.prompt_template.format(text=text)

    def _parse_response(self, response_text: str) -> Optional[List[str]]:
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
                return [str(e).strip() for e in events][:5]
        except Exception as e:
            loguru.logger.error(f"Failed to parse events list: {e}")
            loguru.logger.error(f"List text: {list_text}")
        return None

    def _format_output(self, original_record: Dict, parsed_result: Any) -> Dict:
        return {"uuid": original_record["uuid"], "events": parsed_result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract core events from documents.")
    parser.add_argument(
        "--input_file",
        type=str,
        default="../sample_data/raw_data.json",
        help="Input JSONL file with documents.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="../sample_data/events.jsonl",
        help="Output JSONL file for extracted events.",
    )
    parser.add_argument(
        "--model_name", type=str, default="gemini-2.5-flash", help="Model name to use."
    )
    parser.add_argument("--sleep_time", type=float, default=2, help="Sleep time.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for more verbose logging.",
    )
    args = parser.parse_args()

    event_extractor = EventExtractor(
        input_file=args.input_file,
        output_file=args.output_file,
        model_name=args.model_name,
        sleep_time=args.sleep_time,
        debug=args.debug,
    )
    event_extractor.run()
