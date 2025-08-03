import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time
import random
import json
import sqlite3
from urllib.parse import urljoin, urlparse
import re
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from fake_useragent import UserAgent
import cloudscraper
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class SteamDemoScraper:
    def __init__(self, max_workers=5, delay_range=(1, 3), proxy_list=None):
        self.max_workers = max_workers
        self.delay_range = delay_range
        self.proxy_list = proxy_list or []
        self.current_proxy_index = 0
        self.ua = UserAgent()
        self.session_pool = []
        self.lock = threading.Lock()
        self.request_count = 0
        self.last_request_time = datetime.now()
        self.rate_limit_delay = 1
        
        self.setup_database()
        self.setup_logging()
        self.initialize_session_pool()
        
    def setup_database(self):
        self.conn = sqlite3.connect('steam_demo_data.db', check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS demo_data (
                app_id INTEGER PRIMARY KEY,
                demo_available BOOLEAN,
                demo_score FLOAT,
                demo_review_count INTEGER,
                demo_release_date TEXT,
                full_game_release_date TEXT,
                demo_timing TEXT,
                wishlist_count INTEGER,
                review_summary TEXT,
                price FLOAT,
                discount_percent INTEGER,
                tags TEXT,
                developer TEXT,
                publisher TEXT,
                genre TEXT,
                screenshots_count INTEGER,
                videos_count INTEGER,
                achievement_count INTEGER,
                trading_cards BOOLEAN,
                early_access BOOLEAN,
                mature_content BOOLEAN,
                system_requirements TEXT,
                supported_languages TEXT,
                controller_support TEXT,
                vr_support BOOLEAN,
                mac_support BOOLEAN,
                linux_support BOOLEAN,
                multiplayer BOOLEAN,
                single_player BOOLEAN,
                co_op BOOLEAN,
                last_updated TEXT,
                scrape_timestamp TEXT
            )
        ''')
        self.conn.commit()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('steam_scraper.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def initialize_session_pool(self):
        for _ in range(self.max_workers):
            scraper = cloudscraper.create_scraper(
                browser={
                    'browser': 'chrome',
                    'platform': 'windows',
                    'mobile': False
                }
            )
            
            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504]
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            scraper.mount("http://", adapter)
            scraper.mount("https://", adapter)
            
            headers = {
                'User-Agent': self.ua.random,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Cache-Control': 'max-age=0'
            }
            scraper.headers.update(headers)
            self.session_pool.append(scraper)
            
    def get_session(self):
        with self.lock:
            session = self.session_pool[self.request_count % len(self.session_pool)]
            self.request_count += 1
            return session
            
    def apply_rate_limiting(self):
        with self.lock:
            current_time = datetime.now()
            time_since_last = (current_time - self.last_request_time).total_seconds()
            
            if time_since_last < self.rate_limit_delay:
                sleep_time = self.rate_limit_delay - time_since_last + random.uniform(*self.delay_range)
                time.sleep(sleep_time)
                
            self.last_request_time = datetime.now()
            
    def rotate_proxy(self, session):
        if self.proxy_list:
            with self.lock:
                proxy = self.proxy_list[self.current_proxy_index % len(self.proxy_list)]
                self.current_proxy_index += 1
                
            session.proxies.update({
                'http': proxy,
                'https': proxy
            })
            
    def scrape_demo_data(self, app_id):
        session = self.get_session()
        self.apply_rate_limiting()
        self.rotate_proxy(session)
        
        url = f"https://store.steampowered.com/app/{app_id}"
        
        try:
            response = session.get(url, timeout=30)
            response.raise_for_status()
            
            if "agecheck" in response.url:
                age_verify_data = {
                    'snr': '',
                    'ageDay': '1',
                    'ageMonth': 'January',
                    'ageYear': '1990'
                }
                response = session.post(url, data=age_verify_data, timeout=30)
                response.raise_for_status()
                
            soup = BeautifulSoup(response.content, 'html.parser')
            
            demo_data = {
                'app_id': app_id,
                'demo_available': self.check_demo_presence(soup),
                'demo_score': self.extract_demo_score(soup),
                'demo_review_count': self.extract_demo_review_count(soup),
                'demo_release_date': self.extract_demo_release_date(soup),
                'full_game_release_date': self.extract_release_date(soup),
                'wishlist_count': self.extract_wishlist_count(soup),
                'review_summary': self.extract_review_summary(soup),
                'price': self.extract_price(soup),
                'discount_percent': self.extract_discount(soup),
                'tags': self.extract_tags(soup),
                'developer': self.extract_developer(soup),
                'publisher': self.extract_publisher(soup),
                'genre': self.extract_genre(soup),
                'screenshots_count': self.count_screenshots(soup),
                'videos_count': self.count_videos(soup),
                'achievement_count': self.extract_achievement_count(soup),
                'trading_cards': self.check_trading_cards(soup),
                'early_access': self.check_early_access(soup),
                'mature_content': self.check_mature_content(soup),
                'system_requirements': self.extract_system_requirements(soup),
                'supported_languages': self.extract_languages(soup),
                'controller_support': self.extract_controller_support(soup),
                'vr_support': self.check_vr_support(soup),
                'mac_support': self.check_mac_support(soup),
                'linux_support': self.check_linux_support(soup),
                'multiplayer': self.check_multiplayer(soup),
                'single_player': self.check_single_player(soup),
                'co_op': self.check_coop(soup),
                'last_updated': self.extract_last_updated(soup),
                'scrape_timestamp': datetime.now().isoformat()
            }
            
            demo_data['demo_timing'] = self.classify_demo_timing(demo_data)
            
            self.save_to_database(demo_data)
            self.logger.info(f"Successfully scraped data for app {app_id}")
            
            return demo_data
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request error for app {app_id}: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error for app {app_id}: {str(e)}")
            return None
            
    def check_demo_presence(self, soup):
        demo_indicators = [
            soup.find('div', class_='game_area_dlc_bubble'),
            soup.find('a', href=re.compile(r'/app/\d+.*demo', re.I)),
            soup.find(text=re.compile(r'download.*demo', re.I)),
            soup.find('div', {'class': 'demo_area_button'}),
            soup.find('span', text=re.compile(r'demo available', re.I))
        ]
        
        return any(indicator for indicator in demo_indicators)
        
    def extract_demo_score(self, soup):
        try:
            demo_reviews = soup.find('div', {'data-tooltip-html': re.compile(r'demo', re.I)})
            if demo_reviews:
                score_text = demo_reviews.get('data-tooltip-html', '')
                score_match = re.search(r'(\d+)%', score_text)
                if score_match:
                    return float(score_match.group(1))
                    
            review_score = soup.find('div', class_='user_reviews_summary_bar')
            if review_score:
                score_text = review_score.get('data-tooltip-html', '')
                score_match = re.search(r'(\d+)%.*positive', score_text, re.I)
                if score_match:
                    return float(score_match.group(1))
                    
            return None
        except Exception:
            return None
            
    def extract_demo_review_count(self, soup):
        try:
            review_count_elem = soup.find('span', class_='responsive_hidden')
            if review_count_elem:
                count_text = review_count_elem.text
                count_match = re.search(r'([\d,]+)', count_text.replace(',', ''))
                if count_match:
                    return int(count_match.group(1))
            return 0
        except Exception:
            return 0
            
    def extract_demo_release_date(self, soup):
        try:
            demo_section = soup.find('div', class_='block', text=re.compile(r'demo', re.I))
            if demo_section:
                date_elem = demo_section.find_next('div', class_='date')
                if date_elem:
                    return date_elem.text.strip()
            return None
        except Exception:
            return None
            
    def extract_release_date(self, soup):
        try:
            release_date_elem = soup.find('div', class_='release_date')
            if release_date_elem:
                date_span = release_date_elem.find('span', class_='date')
                if date_span:
                    return date_span.text.strip()
            return None
        except Exception:
            return None
            
    def extract_wishlist_count(self, soup):
        try:
            script_tags = soup.find_all('script')
            for script in script_tags:
                if script.string and 'wishlist' in script.string.lower():
                    wishlist_match = re.search(r'"wishlist_count":\s*(\d+)', script.string)
                    if wishlist_match:
                        return int(wishlist_match.group(1))
            return 0
        except Exception:
            return 0
            
    def extract_review_summary(self, soup):
        try:
            summary_elem = soup.find('span', class_='game_review_summary')
            if summary_elem:
                return summary_elem.text.strip()
            return None
        except Exception:
            return None
            
    def extract_price(self, soup):
        try:
            price_elem = soup.find('div', class_='game_purchase_price price')
            if not price_elem:
                price_elem = soup.find('div', class_='discount_final_price')
            if not price_elem:
                price_elem = soup.find('div', class_='game_purchase_price')
                
            if price_elem:
                price_text = price_elem.text.strip()
                price_match = re.search(r'\$?([\d,]+\.?\d*)', price_text.replace(',', ''))
                if price_match:
                    return float(price_match.group(1))
            
            if soup.find(text=re.compile(r'free to play', re.I)):
                return 0.0
                
            return None
        except Exception:
            return None
            
    def extract_discount(self, soup):
        try:
            discount_elem = soup.find('div', class_='discount_pct')
            if discount_elem:
                discount_text = discount_elem.text.strip()
                discount_match = re.search(r'-(\d+)%', discount_text)
                if discount_match:
                    return int(discount_match.group(1))
            return 0
        except Exception:
            return 0
            
    def extract_tags(self, soup):
        try:
            tags = []
            tag_elements = soup.find_all('a', class_='app_tag')
            for tag_elem in tag_elements:
                tag_text = tag_elem.text.strip()
                if tag_text and tag_text not in tags:
                    tags.append(tag_text)
            return json.dumps(tags) if tags else None
        except Exception:
            return None
            
    def extract_developer(self, soup):
        try:
            dev_elem = soup.find('div', {'id': 'developers_list'})
            if dev_elem:
                dev_link = dev_elem.find('a')
                if dev_link:
                    return dev_link.text.strip()
            return None
        except Exception:
            return None
            
    def extract_publisher(self, soup):
        try:
            pub_elem = soup.find('div', class_='summary column').find('div', string=re.compile('Publisher'))
            if pub_elem:
                pub_link = pub_elem.find_next('a')
                if pub_link:
                    return pub_link.text.strip()
            return None
        except Exception:
            return None
            
    def extract_genre(self, soup):
        try:
            genres = []
            details_block = soup.find('div', class_='details_block')
            if details_block:
                genre_links = details_block.find_all('a', href=re.compile(r'genre'))
                for link in genre_links:
                    genre_text = link.text.strip()
                    if genre_text and genre_text not in genres:
                        genres.append(genre_text)
            return json.dumps(genres) if genres else None
        except Exception:
            return None
            
    def classify_demo_timing(self, demo_data):
        if not demo_data['demo_available'] or not demo_data['demo_release_date']:
            return 'no_demo'
            
        try:
            demo_date = datetime.strptime(demo_data['demo_release_date'], '%b %d, %Y')
            game_date = datetime.strptime(demo_data['full_game_release_date'], '%b %d, %Y')
            
            if demo_date < game_date:
                return 'pre_release'
            elif demo_date == game_date:
                return 'simultaneous'
            else:
                return 'post_release'
        except Exception:
            return 'unknown'
            
    def save_to_database(self, data):
        try:
            placeholders = ', '.join(['?' for _ in data.keys()])
            columns = ', '.join(data.keys())
            sql = f"INSERT OR REPLACE INTO demo_data ({columns}) VALUES ({placeholders})"
            
            self.cursor.execute(sql, list(data.values()))
            self.conn.commit()
        except Exception as e:
            self.logger.error(f"Database error: {str(e)}")
            
    def batch_scrape(self, app_ids, batch_size=100):
        total_scraped = 0
        failed_scrapes = []
        
        for i in range(0, len(app_ids), batch_size):
            batch = app_ids[i:i + batch_size]
            self.logger.info(f"Processing batch {i//batch_size + 1}, apps {i} to {min(i + batch_size, len(app_ids))}")
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_appid = {executor.submit(self.scrape_demo_data, app_id): app_id for app_id in batch}
                
                for future in as_completed(future_to_appid):
                    app_id = future_to_appid[future]
                    try:
                        result = future.result(timeout=60)
                        if result:
                            total_scraped += 1
                        else:
                            failed_scrapes.append(app_id)
                    except Exception as e:
                        self.logger.error(f"Error processing app {app_id}: {str(e)}")
                        failed_scrapes.append(app_id)
                        
            self.logger.info(f"Batch complete. Total scraped: {total_scraped}, Failed: {len(failed_scrapes)}")
            time.sleep(random.uniform(5, 10))
            
        return total_scraped, failed_scrapes
        
    def export_data(self, filename='steam_demo_data.csv'):
        try:
            df = pd.read_sql_query("SELECT * FROM demo_data", self.conn)
            df.to_csv(filename, index=False)
            self.logger.info(f"Data exported to {filename}")
            return df
        except Exception as e:
            self.logger.error(f"Export error: {str(e)}")
            return None
            
    def cleanup(self):
        if hasattr(self, 'conn'):
            self.conn.close()
        for session in self.session_pool:
            session.close()
