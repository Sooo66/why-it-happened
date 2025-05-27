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

logger = logger.bind(name=__name__)

def get_docs(query: str, doc_num: int = 10):
    url = 'https://google.serper.dev/news'

    payload = json.dumps({
        'q': query,
        # 'gl': 'cn'
        # 'hl': 'zh-cn',
        'num': doc_num
    })
    headers = {
        'X-API-KEY': '997fdb65ef7c85428bdd12d87b9c38730ffb8af7',
        'Content-Type': 'application/json',
    }

    response = requests.request('POST', url, headers=headers, data=payload)
    return response.text

def filter_and_sort_docs(doc_str: str, doc_num: int, meta_date: str, do_filter: bool = True):
    try:
        data = json.loads(doc_str)
    except json.JSONDecodeError:
        logger.error("Failed to decoding response.")
        return None
    news = data.get('news', [])
    if len(news) == 0:
        logger.error("No news found in the response.")
        return None

    def parse_date(date_str: str) -> str:
        timezone = pytz.timezone('UTC')
        now = datetime.now(timezone)

        hours_match = regex.search(r"(\d+) hour(s)? ago", date_str)
        days_match = regex.search(r"(\d+) day(s)? ago", date_str)
        weeks_match = regex.search(r"(\d+) week(s)? ago", date_str)
        months_match = regex.search(r"(\d+) month(s)? ago", date_str)

        if hours_match:
            hours = int(hours_match.group(1))
            return now - timedelta(hours=hours)
        elif days_match:
            days = int(days_match.group(1))
            return now - timedelta(days=days)
        elif weeks_match:
            weeks = int(weeks_match.group(1))
            return now - timedelta(days=7 * weeks)
        elif months_match:
            months = int(months_match.group(1))
            return now - timedelta(days=30 * months)
        else:
            logger.warning(f"There is an unkonwn date format: {date_str}")
            return now
        
    def filter_func(x, meta_date) -> bool:
        parsed_date = x.get('parsed_date', None)
        if not parsed_date:
            logger.warning("No parsed date found in the news item.")
            return False
        try:
            parsed_date = datetime.strptime(parsed_date, '%Y-%m-%d').replace(tzinfo=pytz.UTC)
        except ValueError as e:
            logger.warning(f"Failed to parse meta_date: {meta_date}. Error: {e}")
            return False

        left_bound = meta_date - timedelta(days=30 * 6)
        right_bound = meta_date + timedelta(days=30 * 18)
        return left_bound <= parsed_date <= right_bound

    for nw in news:
        date_str = nw.get('date', '')
        parsed_date = parse_date(date_str)
        nw['parsed_date'] = parsed_date.strftime('%Y-%m-%d')
    if do_filter:
        try:
            meta_date = datetime.strptime(meta_date, '%Y-%m')
        except ValueError:
            logger.error(f"Invalid meta_date format: {meta_date}. Expected format is 'YYYY-MM'.")
            raise

        meta_date = meta_date.replace(tzinfo=pytz.UTC)
        news = list(filter(lambda x: filter_func(x, meta_date), news))
        if not news:
            logger.warning("No news items passed the filter criteria.")
            return None

    news.sort(key=lambda x: x['parsed_date'], reverse=True)
    
    return news[:doc_num]


# Global session and settings
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
    """
    获取 URL 内容，返回文本或空字符串。
    """
    try:
        resp = session.get(url, timeout=timeout)
        resp.raise_for_status()
        enc = detect(resp.content).get('encoding') or 'utf-8'
        resp.encoding = enc if enc.lower() != 'ascii' else 'utf-8'
        return resp.text
    except requests.RequestException as e:
        logger.warning(f"请求失败：{e}")
        return ""


def _get_archive_link(url: str) -> str:
    """
    从 archive.is 获取存档链接。
    """
    archive_url = f"https://archive.is/{url}"
    html = _fetch_url(archive_url)
    if not html:
        return ""
    try:
        return BeautifulSoup(html, 'lxml').select_one('div.TEXT-BLOCK a')['href'] or ""
    except Exception as e:
        logger.warning(f"解析 archive 页面失败：{e}")
        return ""

def fetch_and_parse_url(
    url: str,
    timeout: int = 10,
    output_format: str = 'markdown'
) -> str:
    """
    获取网页并提取正文：
    1) trafilatura
    2) newspaper3k
    3) BeautifulSoup
    """
    html = _fetch_url(url, timeout)
    if not html:
        archive_link = _get_archive_link(url)
        if archive_link:
            html = _fetch_url(archive_link, timeout)
        if not html:
            logger.error(f"无法获取 HTML: {url}")
            return ""

    # 1. trafilatura 提取
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
    logger.info(f"trafilatura 提取失败，切换到 newspaper3k")

    # 2. newspaper3k
    try:
        article = Article(url)
        article.download()
        article.parse()
        if len(article.text.strip()) > 20:
            return article.text.strip()
    except Exception:
        logger.info(f"newspaper3k 提取失败")

    # 3. BeautifulSoup 段落解析
    paragraphs = [p.get_text(strip=True) for p in BeautifulSoup(html, 'lxml').find_all('p')]
    joined = "\n\n".join([p for p in paragraphs if p])
    if len(joined) > 20:
        return joined

    logger.error(f"所有提取方法均失败: {url}")
    return ""

def main():
    topics = read_file('../data/topics.json')
    for tpc in tqdm(
        topics, desc="Procedding topics", unit="topic"
    ):
        query = tpc["topic"]
        meta_date = tpc["meta_date"]
        query = f"{query}. {meta_date}"
        logger.info(f"Input query: {query}")
        docs = get_docs(query, doc_num=10)
        docs = filter_and_sort_docs(docs, doc_num=10, meta_date=meta_date, do_filter=True)
        if docs is not None:
            logger.info(f"Got {len(docs)} documents for topic: {query}")
        else:
            logger.error(f"No valid documents left for topic: {query}")
            continue

        for doc in tqdm(
            docs, desc="Fetching and parsing documents", unit="doc"
        ):
            url = doc.get('link')
            content = fetch_and_parse_url(url, timeout=10, output_format='markdown', max_retries=3)
            doc['content'] = content
            time.sleep(random.uniform(0.5, 1.5))
        docs = list(filter(lambda x: len(x.get('content', '')) > 0, docs))
        logger.info(f"{query} has {len(docs)} valid documents after content extraction.")
        tpc['docs'] = docs
        write_line(tpc, "../data/raw_docs.jsonl")
    convert_to_json_list("../data/raw_doc.jsonl", "../data/raw_docs.json")

if __name__ == "__main__":
    main()
            