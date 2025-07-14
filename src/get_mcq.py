import json
import re
import time
import random
import argparse
import uuid
from itertools import combinations
from tqdm import tqdm
from loguru import logger
from typing import Dict, List, Tuple, Set
from utils import read_jsonl, write_line

# --- Main Class ---

logger = logger.bind(name=__name__)

class McqGenerator:
    # --- Constants for easy configuration ---
    # NEW: Score thresholds are now configurable class variables
    POS_THRESHOLD = 60
    AMBIGUOUS_LOWER_BOUND = 40
    HARD_NEGATIVE_LOWER_BOUND = 10 # For picking harder negatives from the low-score pool
    
    # CHANGED: A more logical negative option for "no answer" questions
    NEG_OPTION_NO_ANSWER = "None of the others are correct causes."
    
    MAX_PER_EVENT = 5  # max questions per target event

    def __init__(self, event_path: str, score_path: str, annotation_path: str, output_path: str):
        self.event_data = read_jsonl(event_path)
        self.score_data = read_jsonl(score_path)
        self.output_path = output_path
        
        # Clear output file before writing
        with open(self.output_path, 'w') as f:
            pass

        self.score_map = {item["uuid"]: item["avg"] for item in self.score_data}
        
        # NEW: Pre-processing step to categorize all potential causes for each event
        # This is more efficient and centralizes the logic for candidate selection.
        self.events_by_topic: Dict[str, Dict[int, str]] = {}
        self.causal_candidates: Dict[str, Dict[str, Dict[str, List[str]]]] = {}
        if annotation_path:
            self._prepare_score(annotation_path)
        self._prepare_causal_candidates()

    def _prepare_score(self, annotation_path: str):
        annotation_data = read_jsonl(annotation_path)
        uuid2score = {rec['uuid']: rec['score'] for rec in annotation_data}
        for uuid, value in self.score_map.items():
            if uuid in uuid2score:
                logger.debug(f"UUID {uuid} found in annotation data with score {uuid2score[uuid]}")
                self.score_map[uuid] = 50 if uuid2score[uuid] == 0 else 100
            else:
                continue

    def _prepare_causal_candidates(self):
        """
        NEW: Pre-processes all event pairs to categorize potential causes for each target event.
        This avoids re-calculating pools for every event in the run loop.
        """
        logger.info("Preprocessing and categorizing causal candidates...")
        # Temporary structure to hold all relationships
        raw_records_by_topic: Dict[str, List[Dict]] = {}
        for rec in self.event_data:
            topic = rec["topic_id"]
            raw_records_by_topic.setdefault(topic, []).append(rec)
            self.events_by_topic.setdefault(topic, {})[rec["event1_order"]] = rec["event1"]
            self.events_by_topic.setdefault(topic, {})[rec["event2_order"]] = rec["event2"]

        for topic, records in raw_records_by_topic.items():
            self.causal_candidates.setdefault(topic, {})
            for rec in records:
                target_event = rec["event2"]
                cause_candidate = rec["event1"]
                score = self.score_map.get(rec["uuid"], 0)
                is_temporally_before = rec["event1_order"] < rec["event2_order"]

                # Ensure the data structure for the target event exists
                self.causal_candidates[topic].setdefault(target_event, {
                    "positives": [],
                    "later_events": [], # High-score but temporally after -> strong negative
                    "ambiguous": [],    # 40-60 score -> strong negative
                    "hard_negatives": [], # 30-40 score -> medium negative
                    "easy_negatives": []  # < 30 score -> weak negative
                })
                
                # Categorize the cause candidate based on score and temporality
                if is_temporally_before:
                    if score > self.POS_THRESHOLD:
                        self.causal_candidates[topic][target_event]["positives"].append(cause_candidate)
                    elif self.AMBIGUOUS_LOWER_BOUND <= score <= self.POS_THRESHOLD:
                        self.causal_candidates[topic][target_event]["ambiguous"].append(cause_candidate)
                    elif self.HARD_NEGATIVE_LOWER_BOUND <= score < self.AMBIGUOUS_LOWER_BOUND:
                        self.causal_candidates[topic][target_event]["hard_negatives"].append(cause_candidate)
                    else:
                        self.causal_candidates[topic][target_event]["easy_negatives"].append(cause_candidate)
                else:  # Event happens after the target event
                    if score > self.POS_THRESHOLD:
                        # This is a great distractor: high causal score but wrong order
                        self.causal_candidates[topic][target_event]["later_events"].append(cause_candidate)

        # Deduplicate all lists
        for topic in self.causal_candidates:
            for event in self.causal_candidates[topic]:
                for key in self.causal_candidates[topic][event]:
                    self.causal_candidates[topic][event][key] = list(dict.fromkeys(self.causal_candidates[topic][event][key]))

    def _generate_standard_mcq(self, topic: str, target: str, positives: List[str], neg_pool: List[str], k: int) -> Dict:
        """Generates a standard MCQ with k positive answers."""
        if len(positives) < k or len(neg_pool) < (4 - k):
            return None

        pos_subset = random.sample(positives, k)
        neg_subset = random.sample(neg_pool, 4 - k)
        
        options = pos_subset + neg_subset
        random.shuffle(options)
        
        return self._format_question(topic, target, options, set(pos_subset))

    def _generate_no_answer_mcq(self, topic: str, target: str, neg_pool: List[str]) -> Dict:
        """Generates an MCQ where the correct answer is 'None of the above'."""
        if len(neg_pool) < 3:
            return None # Not enough distractors to make this question type

        # Add the special option and select 3 other distractors
        options = [self.NEG_OPTION_NO_ANSWER] + random.sample(neg_pool, 3)
        random.shuffle(options)
        
        return self._format_question(topic, target, options, {self.NEG_OPTION_NO_ANSWER})

    def _format_question(self, topic: str, target: str, options: List[str], golden_set: Set[str]) -> Dict:
        """Helper to format the final record."""
        labels = ["A", "B", "C", "D"]
        paired = list(zip(labels, options))
        
        golden_labels = sorted([lbl for lbl, opt in paired if opt in golden_set])
        if not golden_labels:
            return None # Should not happen with correct logic, but a safeguard

        record = {
            "topic_id": topic,
            "uuid": str(uuid.uuid4()),
            "target_event": f"Which of the following options are the causes of the event '{target}'?",
            "option_A": paired[0][1],
            "option_B": paired[1][1],
            "option_C": paired[2][1],
            "option_D": paired[3][1],
            "golden_answer": ",".join(golden_labels),
        }
        return record

    def run(self):
        stats: Dict[str, Dict[str, int]] = {}
        all_topics = self.events_by_topic.keys()

        for topic in tqdm(all_topics, desc="Topics"):
            stats[topic] = {"events": len(self.events_by_topic[topic]), "mcqs": 0}
            sorted_orders = sorted(self.events_by_topic[topic].keys())

            for order in sorted_orders:
                target = self.events_by_topic[topic][order]
                
                # Skip if this event never appeared as a target event in pairs
                if target not in self.causal_candidates.get(topic, {}):
                    continue
                
                candidates = self.causal_candidates[topic][target]
                positives = candidates["positives"]
                
                # Create a prioritized negative pool for higher difficulty
                # CHANGED: Prioritized pool based on your suggestions
                neg_pool = (
                    candidates["later_events"]
                    + candidates["ambiguous"]
                    + candidates["hard_negatives"]
                    + candidates["easy_negatives"]
                )
                # Remove any accidental overlap with positives
                neg_pool = [n for n in neg_pool if n not in positives]
                
                generated_count = 0

                # --- Generation Strategy ---
                
                # 1. Attempt to generate a "No Correct Answer" question first (high difficulty)
                if generated_count < self.MAX_PER_EVENT:
                    record = self._generate_no_answer_mcq(topic, target, neg_pool)
                    if record:
                        write_line(record, self.output_path)
                        stats[topic]["mcqs"] += 1
                        generated_count += 1
                
                # 2. Generate standard questions with 1, 2, or 3 positives
                possible_k = [k for k in [1, 2, 3] if len(positives) >= k]
                random.shuffle(possible_k) # Try k in random order

                for k in possible_k:
                    if generated_count >= self.MAX_PER_EVENT:
                        break
                    # Generate one question for each k
                    record = self._generate_standard_mcq(topic, target, positives, neg_pool, k)
                    if record:
                        write_line(record, self.output_path)
                        stats[topic]["mcqs"] += 1
                        generated_count += 1

        # log stats
        logger.info("Done. Stats per topic:")
        for t, s in stats.items():
            logger.info(f"Topic {t}: events={s['events']} mcqs={s['mcqs']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate balanced and difficult MCQ data.")
    parser.add_argument("--event_path", default="/Users/ymx66/Workspace/why-it-happened/sample_data/annotation_data/pairs_with_score_0710.jsonl", help="Path to event pairs file.")
    parser.add_argument(
        "--score_path", default="/Users/ymx66/Workspace/why-it-happened/sample_data/annotation_data/pairs_with_score_0710.jsonl", help="Path to scored pairs file."
    )
    parser.add_argument(
        "--annotation_path", default="/Users/ymx66/Workspace/why-it-happened/sample_data/submission.jsonl", help="Path to annotation file."
    )
    parser.add_argument("--output_path", default="../submit_data/sample_data/batch1.jsonl", help="Path for the generated MCQ file.")
    args = parser.parse_args()

    # To run this, you would need to create the sample_data directory and the files
    # or change the default paths to your actual file locations.
    
    # Example of creating dummy files for testing:
    # os.makedirs("./sample_data/annotation_data", exist_ok=True)
    # with open("./sample_data/event_pairs.jsonl", "w") as f: f.write('{"topic_id": "T1", "event1": "E1", "event2": "E3", "event1_order": 1, "event2_order": 3, "uuid": "u1"}\n')
    # with open("./sample_data/annotation_data/pairs_with_score.jsonl", "w") as f: f.write('{"uuid": "u1", "score": 85}\n')

    generator = McqGenerator(
        event_path=args.event_path,
        score_path=args.score_path,
        annotation_path=args.annotation_path,
        output_path=args.output_path,
    )
    generator.run()