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


def write_jsonl(data: List[Dict], file_path: str) -> None:
    try:
        if not isinstance(data, list):
            logger.error(f"The data obj is not a list.")
            raise ValueError
        with open(file_path, "w") as f:
            for item in data:
                json.dump(item, f, ensure_ascii=False)
                f.write("\n")
    except IOError as e:
        logger.error(
            f"Failed to write data to file_path: {file_path}, caused by {str(e)}"
        )
        logger.error(f"Failed to decode json obj in file_path: {file_path}")


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
    with open(path, "r") as f:
        data = [json.loads(l) for l in f.readlines()]
    return data


import uuid


def set_doc_uuid(data: List[Dict]) -> List[Dict]:
    """
    为每个文档设置唯一的 UUID。
    """
    try:
        for d in data:
            docs = d.get("docs")
            for doc in docs:
                if "uuid" not in doc:
                    doc["uuid"] = str(uuid.uuid4())

        return data
    except KeyError:
        logger.error("The input data does not contain 'docs' key.")
        raise


import statistics
import nltk
import os

os.environ["NLTK_DATA"] = "/home/yangmingxuan/nltk_data"


def get_length_distri(data: List[Dict], key: str = "docs") -> List[int]:
    try:
        length = []
        for d in data:
            docs = d[key]
            for doc in docs:
                if "content" in doc:
                    token_len = len(nltk.word_tokenize(doc["content"]))
                    length.append(token_len)

        return length
    except KeyError:
        logger.error(f"The input data does not contain '{key}' key.")
        raise


def cut_length(data: List[Dict], max_length: int = 1000) -> List[Dict]:
    for tpc in data:
        for d in tpc["docs"]:
            tokens = nltk.word_tokenize(d["content"])
            if len(tokens) > max_length:
                d["ori_content"] = " ".join(tokens[:max_length])
            else:
                d["ori_content"] = d["content"]
    return data


def set_pairs(raw_data_path: str, rps_data_path: str, x_key: str) -> None:
    raw_data = read_file(raw_data_path)
    rps_data = read_jsonl(rps_data_path)

    uuid2x = {}
    for rps in rps_data:
        uuid_str = rps["uuid"]
        x = rps[x_key]
        uuid2x[uuid_str] = x

    for d in raw_data:
        docs = d["docs"]
        for doc in docs:
            try:
                uuid = doc["uuid"]
                x = uuid2x[uuid]
                doc[x_key] = x
            except KeyError:
                logger.error("raw_data do not have uuid: {uuid}.")

        # o_path = raw_data_path.split('.')[0] + "_" + x_key + '.jsonl'
        # write_line(d, o_path)
    write_file(raw_data, raw_data_path)


def count_timeline_nodes(path: str) -> List[int]:
    data = read_jsonl(path)
    cnt = []
    for d in data:
        tl = d["timeline"]
        cnt.append(len(tl))

    return cnt


import uuid


def make_pairs_from_timeline(path: str, raw_data_path: str) -> None:
    raw_data = read_file(raw_data_path)
    data = read_jsonl(path)

    for d in data:
        topic_id = d["topic_id"]
        topic = d["topic"]
        timeline = d["timeline"]
        # ori_data = raw_data[topic_id - 1]
        ori_data = next(
            (item for item in raw_data if item["topic_id"] == topic_id), None
        )

        for i in range(1, len(timeline)):
            event2 = timeline[i]["event"]
            pos2 = timeline[i]["position"]
            event2_context = [
                doc["summary"] for doc in ori_data["docs"] if doc["position"] in pos2
            ]

            for j in range(i):
                event1 = timeline[j]["event"]
                pos1 = timeline[j]["position"]
                event1_context = [
                    doc["summary"]
                    for doc in ori_data["docs"]
                    if doc["position"] in pos1
                ]

                new_d = {
                    "topic_id": topic_id,
                    "topic": topic,
                    "uuid": str(uuid.uuid4()),
                    "event1": event1,
                    "event1_context": event1_context,
                    "event1_order": timeline[j]["event_order"],
                    "event2": event2,
                    "event2_context": event2_context,
                    "event2_order": timeline[i]["event_order"],
                }

                write_line(new_d, "../sample_data/event_pairs.jsonl")


def repair_event_pairs(timeline_path, event_pair_path):
    timelines = read_jsonl(timeline_path)
    event_pairs = read_jsonl(event_pair_path)

    for tpc in timelines:
        topic_id = tpc["topic_id"]
        timeline = tpc["timeline"]
        event_order_map = {}
        event_pos_map = {}
        for tl in timeline:
            event_order_map[tl["event"]] = tl["event_order"]
            event_pos_map[tl["event"]] = tl["position"]

        ent_data = [d for d in event_pairs if d["topic_id"] == topic_id]

        for ent in ent_data:
            # Create a new dictionary with desired field order
            new_ent = {
                "topic_id": ent["topic_id"],
                "topic": ent["topic"],
                "uuid": ent["uuid"],
                "event1": ent["event1"],
                "event1_order": event_order_map[ent["event1"]],
                "event1_context": ent["event1_context"],
                "event1_pos": event_pos_map[ent["event1"]],
                "event2": ent["event2"],
                "event2_order": event_order_map[ent["event2"]],
                "event2_context": ent["event2_context"],
                "event2_pos": event_pos_map[
                    ent["event2"]
                ],  # Fixed typo: 'event2_map' -> 'event2_pos'
            }
            write_line(new_ent, "../data/event_pairs.jsonl")


from collections import defaultdict


def get_annotation_res(paths: List[str]) -> None:
    data_list = [read_jsonl(p) for p in paths]

    score_sums = defaultdict(list)

    for data in data_list:
        for item in data:
            uuid = item["uuid"]
            score = item["score"]
            score_sums[uuid].append(score)

    averages = [
        {"uuid": uuid, "score": int(sum(scores) / len(scores))}
        for uuid, scores in score_sums.items()
    ]

    for d in averages:
        write_line(d, "../sample_data/annotation_data/pairs_with_score.jsonl")


def get_mcq_score(pred_file: str, gt_file: str):
    pred = read_jsonl(pred_file)
    gt = read_jsonl(gt_file)

    # 建立 uuid 到 golder 的映射
    gt_dict = {d["uuid"]: d["golden_answer"] for d in gt}

    total = 0
    score = 0

    for p in pred:
        uuid = p["uuid"]
        if p["answer"] is None:
            continue
        pred_ans = set(p["answer"].split(","))
        # gold_ans = set(gt_dict[uuid].split(','))
        gold_ans = set(gt_dict.get(uuid, "").split(","))

        total += 1
        if pred_ans == gold_ans:
            score += 1
        elif pred_ans.issubset(gold_ans):
            score += 0.5
        else:
            score += 0

    print(f"Total: {total}, Score: {score}, Accuracy: {score / total:.4f}")
    return score / total


# 0-20: 0, 20-50: 1, 50-80: 2, 80-100: 3
def get_score_dist(data: str) -> Dict:
    data = read_jsonl(data)
    score_dist = {"0-20": 0, "20-50": 0, "50-80": 0, "80-100": 0}
    for d in data:
        score = d["score"]
        if score < 20:
            score_dist["0-20"] += 1
        elif score < 50:
            score_dist["20-50"] += 1
        elif score < 80:
            score_dist["50-80"] += 1
        else:
            score_dist["80-100"] += 1
    return score_dist


import random


def get_verify_data(ori_data: str, score_data: str) -> List[Dict]:
    ori = read_jsonl(ori_data)
    score = read_jsonl(score_data)

    uuid2score = {d["uuid"]: d["score"] for d in score}

    # 构建 uuid -> ori 映射
    uuid2ori = {d["uuid"]: d for d in ori if d["uuid"] in uuid2score}

    # 初始化分层容器
    buckets: Dict[str, List[Dict]] = {
        "0-20": [],
        "20-50": [],
        "50-80": [],
        "80-100": [],
    }

    # 分配数据到各分层
    for uuid, sc in uuid2score.items():
        if uuid not in uuid2ori:
            continue
        item = uuid2ori[uuid]
        item["ori_score"] = sc  # 添加分数到原始数据中
        if sc < 20:
            buckets["0-20"].append(item)
        elif sc < 50:
            buckets["20-50"].append(item)
        elif sc < 80:
            buckets["50-80"].append(item)
        else:
            buckets["80-100"].append(item)

    # 从每层中随机抽样20个
    sampled_data = []
    for label, data_list in buckets.items():
        sample_size = min(20, len(data_list))
        if label == "0-20" or label == "80-100":
            sample_size = min(10, sample_size)
        sampled = random.sample(data_list, sample_size)
        sampled_data.extend(sampled)
        # print(sampled[0])
        print(f"{label}: sampled {sample_size} items")

    return sampled_data


def score_to_category(score: int) -> str:
    if 0 <= score <= 20:
        return 0
    elif 20 < score <= 50:
        return 1
    elif 50 < score <= 80:
        return 2
    elif 80 < score <= 100:
        return 3


def score_to_category2(score: int) -> int:
    if 0 <= score <= 50:
        return 0
    elif 50 < score <= 100:
        return 1


import krippendorff
import numpy as np


def calc_krippendorff_alpha(paths):
    all_data = {}
    for path in paths:
        data = read_jsonl(path)
        for d in data:
            uuid = d["uuid"]
            score = d["score"]
            score = score_to_category(score)
            # score = score_to_category2(score)
            if uuid not in all_data:
                all_data[uuid] = []
            all_data[uuid].append(score)

    num_raters = len(paths)
    sorted_uuids = sorted(all_data.keys())
    full_matrix = [[None for _ in range(num_raters)] for _ in range(len(sorted_uuids))]
    for i, uuid in enumerate(sorted_uuids):
        scores = all_data[uuid]
        for j in range(num_raters):
            full_matrix[i][j] = scores[j] if j < len(scores) else None

    def filter_matrix(matrix, valid_labels):
        new_matrix = []
        for row in matrix:
            new_row = [s if s in valid_labels else None for s in row]
            if any(v is not None for v in new_row):  # 至少有一个有效评分
                new_matrix.append(new_row)
        return np.array(new_matrix, dtype=float)

    results = {}
    for label_set in [[0, 1], [1, 2], [2, 3], [0, 3], [0, 1, 2, 3]]:
        # for label_set in [[0, 1]]:
        filtered = filter_matrix(full_matrix, label_set)
        alpha = krippendorff.alpha(
            reliability_data=filtered.T, level_of_measurement="ordinal", dtype=float
        )
        results[str(label_set)] = float(alpha)

    return results


def plt_dist(data: List, bins=30) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # 绘制分布图
    sns.histplot(data, bins=bins, color="blue", stat="count")

    # 添加标题和标签
    plt.title("Distribution of Document Lengths")
    plt.xlabel("Length")
    plt.ylabel("Density")

    # 显示图形
    plt.show()


# def rm_ir_info(in_path: str, out_path: str) -> None:
#     data = read_file(in_path)
#     for tpc in data:
#         docs = tpc['docs']
#         for doc in docs:
#             del doc['imageUrl']


def pair_dis_docs(dis_path: str, raw_path: str) -> None:
    data1 = read_file(dis_path)
    data2 = read_jsonl(raw_path)


def calculate_score_variance(paths: List[str]) -> Dict[str, float]:
    uuid_scores = defaultdict(list)

    for path in paths:
        data = read_jsonl(path)
        for item in data:
            uuid = item.get("uuid")
            score = item.get("score")
            if uuid and score is not None:
                uuid_scores[uuid].append(score)
            else:
                logger.warning(f"Skipping item with missing uuid or score in {path}: {item}")

    uuid_variance = {}
    new_d = []
    for uuid, scores in uuid_scores.items():
        if len(scores) >= 2:  # 方差计算至少需要两个数据点
            try:
                variance = statistics.variance(scores)
                average_score = int(statistics.mean(scores))
                uuid_variance[uuid] = variance
                new_d.append({"uuid": uuid, "var": variance, "avg": average_score})
            except statistics.StatisticsError as e:
                logger.error(f"Could not calculate variance for uuid {uuid}: {e}")
        else:
            logger.warning(f"Not enough scores to calculate variance for uuid {uuid}: {scores}")

    # return uuid_variance
    return new_d

from typing import List, Dict

def get_verify_data_with_var(ori_data: str, score_data: List[Dict]) -> Dict[str, List[Dict]]:
    ori = read_jsonl(ori_data)

    # 构建映射
    uuid2score = {d["uuid"]: {"avg": d["avg"], "var": d["var"]} for d in score_data}
    uuid2ori = {d["uuid"]: d for d in ori if d["uuid"] in uuid2score}

    # 初始化 var 分组 + avg 分层结构
    buckets: Dict[str, Dict[str, List[Dict]]] = {
        "<500": {
            "0-20": [],
            "20-50": [],
            "50-80": [],
            "80-100": [],
        },
        ">=500": {
            "0-20": [],
            "20-50": [],
            "50-80": [],
            "80-100": [],
        },
    }

    # 将记录分配到对应分层中
    for uuid, stats in uuid2score.items():
        if uuid not in uuid2ori:
            continue

        avg = stats["avg"]
        var = stats["var"]
        ori_item = uuid2ori[uuid].copy()
        ori_item["ori_avg"] = avg
        ori_item["ori_var"] = var

        var_group = "<500" if var < 500 else ">=500"

        if avg < 20:
            buckets[var_group]["0-20"].append(ori_item)
        elif avg < 50:
            buckets[var_group]["20-50"].append(ori_item)
        elif avg < 80:
            buckets[var_group]["50-80"].append(ori_item)
        else:
            buckets[var_group]["80-100"].append(ori_item)

    # 分别采样两个 var 分组中的各层数据，返回两个列表
    results: Dict[str, List[Dict]] = {"<500": [], ">=500": []}
    for var_group, layers in buckets.items():
        print(f"\nVar group: {var_group}")
        for label, data_list in layers.items():
            sample_size = min(10, len(data_list))
            sampled = random.sample(data_list, sample_size)
            results[var_group].extend(sampled)
            print(f"  {label}: sampled {sample_size} items")

    return results
