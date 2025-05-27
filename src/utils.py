import json
from typing import List, Dict
from loguru import logger


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
