import json
from typing import List, Dict
from loguru import logger

logger = logger.bind(name=__name__)

def read_file(file_path: str) -> List[Dict]:
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
            if not isinstance(data, list):
                logger.error("The input json object is not a list.")
                raise ValueError
            return data
    except FileNotFoundError:
        logger.error(f"The input file_path: {file_path} is not exist.")
        raise
    except json.JSONDecodeError:
        logger.error(f"Failed to decode json obj in file_path: {file_path}")
        raise


def write_file(data: List[Dict], file_path: str) -> None:
    try:
        if not isinstance(data, list):
            logger.error(f"The data obj is not a list.")
            raise ValueError
        with open(file_path, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except IOError as e:
        logger.error(
            f"Failed to write data to file_path: {file_path}, caused by {str(e)}"
        )


def write_line(data: Dict, file_path: str) -> None:
    try:
        if not isinstance(data, dict):
            logger.error(f"The data obj is not a dict.")
            raise ValueError
        with open(file_path, "a") as f:
            json.dump(data, f, ensure_ascii=False)
            f.write("\n")
    except IOError as e:
        logger.error(
            f"Failed to write data to file_path: {file_path}, caused by {str(e)}"
        )


def convert_to_json_list(input_file: str, output_file: str) -> None:
    try:
        data_list = []
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data_list.append(json.loads(line))

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data_list, f, ensure_ascii=False, indent=2)
    except (IOError, json.JSONDecodeError) as e:
        logger.error(f"Failed to convert file {input_file} to JSON list: {str(e)}")
        raise

def read_jsonl(path: str) -> List:
    with open(path, 'r') as f:
        data = [json.loads(l) for l in f.readlines()]
    return data

import uuid
def set_doc_uuid(data: List[Dict]) -> List[Dict]:
    """
    为每个文档设置唯一的 UUID。
    """
    try:
        for d in data:
            docs = d.get('docs')
            for doc in docs:
                if 'uuid' not in doc:
                    doc['uuid'] = str(uuid.uuid4())

        return data
    except KeyError:
        logger.error("The input data does not contain 'docs' key.")
        raise 

import statistics
import nltk
import os
os.environ['NLTK_DATA'] = '/home/yangmingxuan/nltk_data'
def get_length_distri(data: List[Dict], key: str="docs") -> List[int]:
    try:
        length = []
        for d in data:
            docs = d[key]
            for doc in docs:
                if 'content' in doc:
                    token_len = len(nltk.word_tokenize(doc['content']))
                    length.append(token_len)
        
        return length
    except KeyError:
        logger.error(f"The input data does not contain '{key}' key.")
        raise

def cut_length(data: List[Dict], key: str="docs", max_length: int = 1000) -> List[Dict]:
    """
    截断文档内容到指定长度。
    """
    try:
        for d in data:
            docs = d["key"]
            for doc in docs:
                if 'content' in doc:
                    tokens = nltk.word_tokenize(doc['content'])
                    if len(tokens) > max_length:
                        doc['content'] = ' '.join(tokens[:max_length])
        return data
    except KeyError:
        logger.error("The input data does not contain 'docs' key.")
        raise

def set_pairs(raw_data_path: str, rps_data_path: str, x_key: str) -> None:
    raw_data = read_file(raw_data_path)
    rps_data = read_jsonl(rps_data_path)
    
    uuid2x = {}
    for rps in rps_data:
        uuid_str = rps['uuid']
        x = rps[x_key]
        uuid2x[uuid_str] = x
    
    for d in raw_data:
        docs = d['docs']
        for doc in docs:
            try:
                uuid = doc['uuid']
                x = uuid2x[uuid]
                doc[x_key] = x
            except KeyError:
                logger.error("raw_data do not have uuid: {uuid}.")
    
        o_path = raw_data_path.split('.')[0] + "_" + x_key + '.jsonl'
        write_line(d, o_path)
            
def count_timeline_nodes(path: str) -> List[int]:
    data = read_jsonl(path)
    cnt = []
    for d in data:
        tl = d['timeline']
        cnt.append(len(tl))
    
    return cnt

import uuid

def make_pairs_from_timeline(path: str, raw_data_path: str) -> None:
    raw_data = read_file(raw_data_path)
    data = read_jsonl(path)

    for d in data:
        topic_id = d['topic_id']
        topic = d['topic']
        timeline = d['timeline']
        ori_data = raw_data[topic_id - 1]

        for i in range(1, len(timeline)):
            event2 = timeline[i]['event']
            pos2 = timeline[i]['position']
            event2_context = [doc['summary'] for doc in ori_data['docs'] if doc['position'] in pos2]

            for j in range(i):
                event1 = timeline[j]['event']
                pos1 = timeline[j]['position']
                event1_context = [doc['summary'] for doc in ori_data['docs'] if doc['position'] in pos1]

                new_d = {
                    'topic_id': topic_id,
                    'topic': topic,
                    'uuid': str(uuid.uuid4()),
                    'event1': event1,
                    'event1_context': event1_context,
                    'event2': event2,
                    'event2_context': event2_context,
                }

                write_line(new_d, '../data/event_pairs.jsonl')


from collections import defaultdict

def get_annotation_res(paths: List[str]) -> None:
    data_list = [read_jsonl(p) for p in paths]
    
    score_sums = defaultdict(list)
    
    for data in data_list:
        for item in data:
            uuid = item['uuid']
            score = item['score']
            score_sums[uuid].append(score)

    averages = [
        {
            'uuid': uuid, 
            'score': int(sum(scores) / len(scores))
        }
        for uuid, scores in score_sums.items()
    ]
    
    for d in averages:
        write_line(d, '../data/pairs_with_score.jsonl')