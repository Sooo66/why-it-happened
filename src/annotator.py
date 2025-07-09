import regex as re
from typing import Optional, Dict, List, Any
import argparse
import os
from loguru import logger

import json
from utils import read_jsonl
from llm_base_model import BaseModel

logger = logger.bind(name=__name__)

prompt_template = '''
You are an expert in causal reasoning.  
Your task is to determine whether **Event 1 caused Event 2 through a direct, single-step causal relationship**, based only on the provided contexts. Evaluate the causal relationship by considering the information from both provided contexts together. Focus exclusively on **single-step causality**, which can be explicit or implicit, but must be reasonable, plausible, and directly linked in one step. Do not infer multi-step chains of causation, and do not rely on any background knowledge beyond what is explicitly stated or reasonably implied in the contexts.

### What qualifies as single-step causality
- Explicitly stated or reasonably implied direct cause, linked to the effect in one step.  
- For implicit causality, ensure the link is plausible and directly follows from the context without requiring additional assumptions.  
- Not multi-step, coincidental, or speculative beyond the context.

### Scoring criteria
Assign a single integer score from **0 to 100** that reflects the strength of the **single-step causal relationship** between Event 1 and Event 2, based solely on the provided contexts. Use the full 0–100 range as precisely as possible. Avoid round number bias (such as 50, 60, or 80) unless they truly reflect the strength of the causal link.

Score ranges:
- **0–40: No causality:** The events are unrelated, coincidental, or only minimally connected. The context provides no or negligible evidence of a single-step causal link.
- **41–59: Uncertain causality:** The context suggests a possible causal link, but ambiguity or conflicting factors prevent a clear judgment. Reflect uncertainty if the context lacks sufficient detail or presents conflicting evidence.
- **60–100: Clear causality:** The context strongly supports a direct, plausible, single-step causal link with minimal ambiguity.

### Reasoning requirement
Before outputting the final score, provide a brief reasoning (2-3 sentences) that:
- Pinpoints the specific evidence from the contexts that establishes the causal connection (or lack thereof).
- Explains how this evidence demonstrates the causality, specifying whether the link is explicit (directly stated using causal language like "because of," "leading to," "prompting") or implicit (implied by sentence structure or sequence).
- Justifies the assigned score by relating the strength and clarity of this evidence to the 0-100 scale, noting any ambiguities.

### Output format
Your output must be in **JSON format**, containing two fields:
- `"reasoning"`: your reasoning paragraph.
- `"score"`: the integer score you assigned.

## Examples:
### Example 1:
Input:
Topic: UK holds Brexit referendum, decides to leave EU
Event 1: Prime Minister David Cameron pledged a referendum on EU membership in 2013.
Context 1: On June 23, U.K. citizens will vote on whether the United Kingdom should remain in or leave the European Union (EU). In 2013, Prime Minister David Cameron pledged a referendum on EU membership if his Conservative Party won the next election. The party won, and the referendum date was set in February. Cameron's attempt to renegotiate the U.K.'s EU membership terms failed, leading to the "Brexit" campaign. The nation divided into "remain" and "leave" camps. The EU has 28 member states and represents a large economic output. Member nations contribute to the EU budget. EU initiatives have funded projects in the U.K. The EU ensures the free movement of people, goods, services, and money within its single market. Polls show a close split in public opinion. Registration for the referendum caused the government website to crash and spurred an extension of the initial deadline. A transition out of the EU could take years if U.K. citizens vote "leave".
Event 2: The Conservative Party won the election.
Context 2: Same as Context 1.
Output:
{{
  "reasoning": "The context states Cameron 'pledged a referendum on EU membership if his Conservative Party won the next election,' which was followed by the party's victory. This sequence suggests an implicit causal link, but the context does not explicitly state the pledge was the reason for the win. The causality is uncertain because the pledge could have been one of many factors, making the connection plausible but not definite.",
  "score": 52
}}

### Example 2:
Input:
Topic: Former Japanese PM Shinzo Abe assassinated
Event 1: Videos of the assassination circulated on social media.
Context 1: Shinzo Abe, Japan's former prime minister, was assassinated while giving a campaign speech in Nara. The suspected gunman was arrested at the scene with a handmade gun. Abe, 67, was shot, collapsed, and was bleeding from the chest. Videos of the assassination circulated on social media, prompting social media companies to remove harmful content. Abe was a polarizing figure known for revitalizing Japan’s economy and his revisionist views of World War II.
Event 2: Social media companies removed harmful content.
Context 2: Same as Context 1.
Output:
{{
  "reasoning": "The context explicitly states that 'Videos of the assassination circulated on social media, prompting social media companies to remove harmful content.' The causal keyword 'prompting' creates a direct single-step link between the circulation of the videos and the companies' action, supporting a very high score.",
  "score": 95
}}

### Example 3:
Input:
Topic: Former Japanese PM Shinzo Abe assassinated
Event 1: Benjamin Netanyahu visited Tokyo in 2014.
Context 1: Shinzo Abe, former Japanese prime minister, was assassinated at a campaign rally. Abe was shot multiple times from behind while speaking at a rally in Nara. Abe, a nationalist, aimed to change Japan’s pacifist character. He increased diplomacy with Israel, starting with a 2014 visit to Tokyo by Benjamin Netanyahu, which increased trade. Abe offered to host a peace summit in 2017 and visited Jerusalem in 2018.
Event 2: Shinzo Abe was pronounced dead at 5:03 p.m.
Context 2: Former Japanese Prime Minister Shinzo Abe was shot while delivering a campaign speech in Nara. The attack occurred at approximately 11:30 a.m. before upper house elections. Abe collapsed, bleeding from the neck and chest. He was taken to a hospital in cardiac arrest and pronounced dead at 5:03 p.m. due to excessive bleeding; a bullet reached his heart. Tetsuya Yamagami, 41, was arrested for attempted murder; police recovered a gun, possibly homemade. Witnesses reported hearing two gunshots. World leaders, including Donald Trump, Emmanuel Macron, and Narendra Modi, condemned the attack.
Output:
{{
  "reasoning": "Context 1 describes a diplomatic visit by Netanyahu in 2014, while Context 2 details Abe's assassination in 2022. The contexts provide no information connecting these two distinct events, indicating a complete lack of a causal relationship.",
  "score": 3
}}

Input:
Topic:
{topic}
Event 1:
{event1}
Context 1:
{event1_context}
Event 2:
{event2}
Context 2:
{event2_context}
Output:
'''


class Annotator(BaseModel):
    def __init__(
        self,
        model_name: str,
        input_file: str,
        output_file: str,
        type: str,
        sleep_time: float = 0.5,
        debug: bool = False,
    ):
        self.type = type

        output_file = (
            f"../sample_data/annotation_data/{model_name}_causal_score.jsonl"
        )

        # 先调用 BaseModel 的构造函数
        super().__init__(model_name, input_file, output_file, sleep_time, debug)

        # 初始化 Annotator 专属字段
        self.none_uids = set()
        self.processed_uids = set()
        if os.path.exists(self.output_file):
            for rec in read_jsonl(self.output_file):
                uuid = rec.get("uuid")
                if uuid:
                    if rec.get("score") is None:
                        self.none_uids.add(uuid)
                    else:
                        self.processed_uids.add(uuid)

        self.data = self._load_data()
        # self.processed_uids = self._load_processed_uids()

    def _load_data(self) -> List[Dict]:
        data = read_jsonl(self.input_file)
        if self.type == "normal":
            return data
        else:  # type == 'none' - process records with None score or unannotated
            return [
                x
                for x in data
                if x.get("uuid") in self.none_uids
                or x.get("uuid") not in self.processed_uids
            ]

    def _get_prompt(self, record: Dict) -> str:
        def format_context(context_list: List[str]) -> str:
            return "\n".join(f"- {c}" for c in context_list)

        prompt = prompt_template.format(
            topic=record.get("topic", "N/A"),
            event1=record["event1"],
            event1_context=format_context(record["event1_context"]),
            event2=record["event2"],
            event2_context=format_context(record["event2_context"]),
        )
        if self.model_name.startswith("qwen"):
            prompt += "\n /no_think"
        return prompt

    def _parse_response(self, text: Optional[str]):
        if not text:
            return None
        
        try:
            json_match = re.search(r'^```json\s*(.*?)\s*```$', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                json_str = text.strip()
            
            data = json.loads(json_str)
            data["score"] = max(0, min(100, data["score"]))
            return data
        except json.JSONDecodeError as je:
            logger.error(f"Failed to parse JSON from response: {text}. Error: {je}")
            return None
        except KeyError as ke:
            logger.error(f"Failed to get 'score' field: {text}. Missing key: {ke}")
            return None

    def _format_output(
        self, original_record: Dict, parsed_result: Optional[int]
    ) -> Dict:
        return {"uuid": original_record.get("uuid"), "reasoning": parsed_result.get("reasoning"), "score": parsed_result.get("score")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract timeline from docs.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="gemini-2.5-flash",
    )
    parser.add_argument(
        "--input_file", type=str, default="../sample_data/event_pairs.jsonl"
    )
    parser.add_argument(
        "--output_file", type=str, default="../data/a.jsonl"
    )  # This will be overridden
    parser.add_argument("--type", type=str, default="normal")  # none
    parser.add_argument(
        "--sleep_time",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
    )
    args = parser.parse_args()
    Annotator(
        args.model_name,
        args.input_file,
        args.output_file,
        args.type,
        args.sleep_time,
        args.debug,
    ).run()
