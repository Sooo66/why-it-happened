import json
import re
import time
import random
import argparse
import uuid
from itertools import combinations
from tqdm import tqdm
from loguru import logger
from typing import Dict, List
from utils import read_jsonl, write_line

logger = logger.bind(name=__name__)

THRESHOLD = 50
NEG_OPTION = "The information provided is insufficient to determine the cause."
MAX_PER_EVENT = 5  # max questions per target event

class McqGenerator:
    def __init__(self, event_path: str, score_path: str, output_path: str):
        self.event_data = read_jsonl(event_path)
        self.score_data = read_jsonl(score_path)
        self.score_map = {item['uuid']: item['score'] for item in self.score_data}

        # build topic -> order->event mapping
        self.events_by_topic: Dict[str, Dict[int, str]] = {}
        for rec in self.event_data:
            t = rec['topic_id']
            self.events_by_topic.setdefault(t, {})[rec['event1_order']] = rec['event1']
            self.events_by_topic.setdefault(t, {})[rec['event2_order']] = rec['event2']

        # store raw pair records
        self.records_by_topic: Dict[str, List[Dict]] = {}
        for rec in self.event_data:
            self.records_by_topic.setdefault(rec['topic_id'], []).append(rec)

        self.output_path = output_path

    def run(self):
        stats: Dict[str, Dict[str, int]] = {}
        for topic, events in tqdm(self.events_by_topic.items(), desc="Topics"):
            sorted_orders = sorted(events.keys())
            stats[topic] = {'events': len(sorted_orders), 'mcqs': 0}

            for order in sorted_orders:
                target = events[order]
                # positive cause candidates
                positives = [r['event1'] for r in self.records_by_topic[topic]
                             if r['event2'] == target
                             and r['event1_order'] < r['event2_order']
                             and self.score_map.get(r['uuid'], 0) > THRESHOLD]
                positives = list(dict.fromkeys(positives))
                if not positives:
                    continue

                # negative pools
                later = list({r['event1'] for r in self.records_by_topic[topic]
                              if r['event2'] == target
                              and r['event1_order'] > r['event2_order']
                              and self.score_map.get(r['uuid'], 0) > THRESHOLD})
                low = list({r['event1'] for r in self.records_by_topic[topic]
                            if r['event2'] == target
                            and r['event1_order'] < r['event2_order']
                            and self.score_map.get(r['uuid'], 0) <= THRESHOLD})
                neg_pool = later + low + [NEG_OPTION]

                # determine subset sizes to sample: always include 1, and up to 2 if available
                ks = [1]
                if len(positives) >= 2:
                    ks.append(2)
                if len(positives) >= 3:
                    ks.append(3)
                # limit to at most two sizes

                generated = 0
                for k in ks:
                    # sample up to MAX_PER_EVENT//len(ks) subsets of size k
                    trials = min(len(list(combinations(positives, k))),
                                 max(1, MAX_PER_EVENT // len(ks)))
                    sampled = set()
                    tries = 0
                    while len(sampled) < trials and tries < trials * 3:
                        subset = tuple(sorted(random.sample(positives, k)))
                        if subset not in sampled:
                            sampled.add(subset)
                        tries += 1

                    for pos_subset in sampled:
                        pos_list = list(pos_subset)
                        # select negatives randomly from pool
                        n_neg = 4 - len(pos_list)
                        neg_choices = set(neg_pool) - set(pos_list)
                        if len(neg_choices) < n_neg:
                            continue
                        neg_list = random.sample(list(neg_choices), n_neg)

                        # assemble and shuffle
                        options = pos_list + neg_list
                        random.shuffle(options)
                        labels = ['A', 'B', 'C', 'D']
                        paired = list(zip(labels, options))
                        golden = [lbl for lbl, opt in paired if opt in pos_list]
                        golden_str = ','.join(sorted(golden))

                        # write record
                        record = {
                            'topic_id': topic,
                            'uuid': str(uuid.uuid4()),
                            'question': f"Which of the following options are the causes of the event '{target}'?",
                            'option_A': paired[0][1],
                            'option_B': paired[1][1],
                            'option_C': paired[2][1],
                            'option_D': paired[3][1],
                            'golden_answer': golden_str
                        }
                        write_line(record, self.output_path)
                        stats[topic]['mcqs'] += 1
                        generated += 1
                        if generated >= MAX_PER_EVENT:
                            break
                    if generated >= MAX_PER_EVENT:
                        break

        # log stats
        logger.info("Done. Stats per topic:")
        for t, s in stats.items():
            logger.info(f"Topic {t}: events={s['events']} mcqs={s['mcqs']}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate balanced MCQ data.")
    parser.add_argument('--event_path', default='../data/event_pairs.jsonl')
    parser.add_argument('--score_path', default='../data/pairs_with_score.jsonl')
    parser.add_argument('--output_path', default='../data/mcq_fix.jsonl')
    args = parser.parse_args()

    McqGenerator(
        event_path=args.event_path,
        score_path=args.score_path,
        output_path=args.output_path
    ).run()
