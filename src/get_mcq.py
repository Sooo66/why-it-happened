import json
from utils import read_jsonl, write_line, read_file
from tqdm import tqdm
from loguru import logger
from typing import List, Dict, Optional
import argparse

class McqGenerator:
    def __init__(self, event_path, score_path):
        self.event_path = event_path
        self.score_path = score_path
        self.event_data, self.score_data = self._load_data()
        self.uuid2score = {
            d['uuid']: d['score']
            for d in self.score_data
        }

    def _load_data(self):
        return read_jsonl(self.event_path), read_jsonl(self.score_path)

    def run():
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Get mcq data.")
    McqGenerator('../data/event_pairs.jsonl', '../data/pairs_with_scores.jsonl').run()
