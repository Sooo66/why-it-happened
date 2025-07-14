import time
from datetime import datetime, timedelta
import json
import random

from loguru import logger
from utils import read_file, write_line, convert_to_json_list
import regex
from tqdm import tqdm

import requests
import pytz
from charset_normalizer import detect
import trafilatura
from newspaper import Article
from bs4 import BeautifulSoup
from resiliparse.extract.html2text import extract_plain_text

logger = logger.bind(name=__name__)


def get_docs(query: str, doc_num: int = 10):
    url = "https://google.serper.dev/news"
    payload = json.dumps({
        "q": query,
        "num": doc_num,
    })
    headers = {
        "X-API-KEY": "997fdb65ef7c85428bdd12d87b9c38730ffb8af7",
        "Content-Type": "application/json",
    }
    response = requests.post(url, headers=headers, data=payload)
    return json.loads(response.text)


def filter_and_sort_docs(doc_str: str, doc_num: int, meta_date: str, do_filter: bool = True):
    try:
        data = json.loads(doc_str)
    except json.JSONDecodeError:
        logger.error("Failed to decode response.")
        return None

    news = data.get("news", [])
    if not news:
        logger.error("No news found in the response.")
        return None

    def parse_date(date_str: str) -> str:
        timezone = pytz.timezone("UTC")
        now = datetime.now(timezone)
        hours_match = regex.search(r"(\d+) hour(s)? ago", date_str)
        days_match = regex.search(r"(\d+) day(s)? ago", date_str)
        weeks_match = regex.search(r"(\d+) week(s)? ago", date_str)
        months_match = regex.search(r"(\d+) month(s)? ago", date_str)
        if hours_match:
            return now - timedelta(hours=int(hours_match.group(1)))
        elif days_match:
            return now - timedelta(days=int(days_match.group(1)))
        elif weeks_match:
            return now - timedelta(days=7 * int(weeks_match.group(1)))
        elif months_match:
            return now - timedelta(days=30 * int(months_match.group(1)))
        else:
            logger.warning(f"Unknown date format: {date_str}")
            return now

    def filter_func(x, meta_date) -> bool:
        parsed_date = datetime.strptime(x["parsed_date"], "%Y-%m").replace(tzinfo=pytz.UTC)
        left_bound = meta_date - timedelta(days=30 * 6)
        right_bound = meta_date + timedelta(days=30 * 18)
        return left_bound <= parsed_date <= right_bound

    for nw in news:
        date_str = nw.get("date", "")
        parsed_date = parse_date(date_str)
        nw["parsed_date"] = parsed_date.strftime("%Y-%m")

    if do_filter:
        try:
            meta_date = datetime.strptime(meta_date, "%Y-%m").replace(tzinfo=pytz.UTC)
        except ValueError:
            logger.error(f"Invalid meta_date format: {meta_date}. Expected format is 'YYYY-MM'.")
            raise
        news = list(filter(lambda x: filter_func(x, meta_date), news))
        if not news:
            logger.warning("No news items passed the filter criteria.")
            return None

    news.sort(key=lambda x: x["parsed_date"], reverse=True)
    return news[:doc_num]


# Global session for crawling
default_headers = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/136.0.0.0 Safari/537.36"
    )
}
session = requests.Session()
session.headers.update(default_headers)


def _fetch_url(url: str, timeout: int = 10) -> str:
    try:
        resp = session.get(url, timeout=timeout)
        resp.raise_for_status()
        enc = detect(resp.content).get("encoding") or "utf-8"
        resp.encoding = enc if enc.lower() != "ascii" else "utf-8"
        return resp.text
    except requests.RequestException as e:
        logger.warning(f"Request failed: {e}")
        return ""


def _get_archive_link(url: str) -> str:
    archive_url = f"https://archive.is/{url}"
    html = _fetch_url(archive_url)
    if not html:
        return ""
    try:
        return BeautifulSoup(html, "lxml").select_one("div.TEXT-BLOCK a")["href"] or ""
    except Exception as e:
        logger.warning(f"Failed to parse archive page: {e}")
        return ""


def fetch_and_parse_url(url: str, timeout: int = 10, output_format: str = "markdown") -> str:
    try:
        article = Article(url)
        article.download()
        article.parse()
        if len(article.text.strip()) > 20:
            return article.text.strip()
    except Exception:
        logger.warning("newspaper3k extraction failed")

    html = _fetch_url(url, timeout)
    if not html:
        archive_link = _get_archive_link(url)
        if archive_link:
            time.sleep(10)
            html = _fetch_url(archive_link, timeout)
        if not html:
            logger.error(f"Unable to fetch HTML: {url}")
            return ""

    try:
        content = trafilatura.extract(
            html,
            output_format=output_format,
            include_comments=False,
            include_tables=False,
            include_links=False,
            favor_precision=True,
        )
        if content and len(content.strip()) > 20:
            return content.strip()
    except Exception:
        logger.warning("trafilatura extraction failed")

    try:
        rp_content = extract_plain_text(html)
        if rp_content and len(rp_content.strip()) > 20:
            return rp_content.strip()
    except Exception:
        logger.warning("resiliparse extraction failed")

    soup = BeautifulSoup(html, "lxml")
    candidates = soup.find_all(["article", "div", "section"])
    best = max(candidates, key=lambda tag: len("".join(p.get_text(strip=True) for p in tag.find_all("p"))), default=None)
    container = best if best else soup
    paragraphs = [
        p.get_text(strip=True)
        for p in container.find_all("p")
        if len(p.get_text(strip=True)) > 30
    ]
    joined = "\n\n".join(paragraphs)
    if len(joined) > 20:
        return joined

    logger.error(f"All extraction methods failed: {url}")
    return ""


def load_existing_results(filepath: str) -> dict:
    try:
        lines = read_file(filepath)
        data = {}
        for obj in lines:
            # obj = json.loads(line)
            data[obj["topic_id"]] = obj
        logger.info(f"Loaded {len(data)} existing topics from {filepath}")
        return data
    except FileNotFoundError:
        logger.warning(f"No existing file found: {filepath}")
        return {}
    except Exception as e:
        logger.error(f"Failed to load existing results: {e}")
        return {}


def main():
    topics = read_file("../submit_data/topics_batch2.json")
    existing_data = load_existing_results("../submit_data/raw_data_batch2.json")

    for tpc in tqdm(topics, desc="Processing topics", unit="topic"):
        topic_id = tpc["topic_id"]
        topic_query = tpc["topic"]
        dis_words = tpc["distractor_words"]
        meta_date = tpc.get("meta_date")

        existing = existing_data.get(topic_id, {})
        need_fetch_topic = not existing.get("docs")
        need_fetch_disT = not existing.get("dis_T")

        if not need_fetch_topic and not need_fetch_disT:
            logger.info(f"topic_id {topic_id} already complete, skipping.")
            continue

        updated_tpc = existing.copy() if existing else {}
        updated_tpc.update(tpc)

        for query_type, query in [("topic", topic_query), ("distractor", dis_words)]:
            if query_type == "topic" and not need_fetch_topic:
                continue
            if query_type == "distractor" and not need_fetch_disT:
                continue

            if query_type == "topic":
                query = f"{query} {meta_date}"
            logger.info(f"[{query_type}] Input query: {query}")

            docs = get_docs(query, doc_num=15).get("news", [])
            if not docs:
                logger.error(f"No valid documents found for {query_type}: {query}")
                continue

            logger.info(f"Got {len(docs)} documents for {query_type}: {query}")
            for doc in tqdm(docs, desc=f"Parsing [{query_type}]", unit="doc"):
                url = doc.get("link")
                content = fetch_and_parse_url(url, timeout=10, output_format="markdown")
                doc["content"] = content
                time.sleep(random.uniform(10, 11))

            docs = list(filter(lambda x: len(x.get("content", "")) > 0, docs))
            logger.info(f"{query_type} has {len(docs)} valid documents for topic_id {topic_id}.")

            updated_tpc["docs" if query_type == "topic" else "dis_T"] = docs

        write_line(updated_tpc, "../submit_data/raw_data_batch2_3.jsonl")

    # convert_to_json_list("../submit_data/raw_data_batch2.jsonl", "../submit_data/raw_data_batch2.json")


if __name__ == "__main__":
    # main()
    convert_to_json_list("../submit_data/raw_data_batch2_3.jsonl", "../submit_data/raw_data_batch2_3.json")