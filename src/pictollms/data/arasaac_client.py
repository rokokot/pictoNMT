# pictollms/data/arasaac_client.py
import os
import json
import requests
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class ArasaacClient:
    def __init__(self, cache_dir: str, base_url: str = "https://api.arasaac.org/v1/pictograms"):
        self.base_url = base_url
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.session = requests.Session()
        
    def get_pictogram_metadata(self, picto_id: int, language: str = 'fr'):
        cache_path = os.path.join(self.cache_dir, f"{picto_id}_{language}_meta.json")
        if os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        url = f"{self.base_url}/{language}/{picto_id}"
        try:
            response = self.session.get(url)
            
            if response.status_code == 200:
                data = response.json()
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                return data
            else:
                logger.warning(f"Error fetching metadata for {picto_id}: {response.status_code}")
                return {}
        except Exception as e:
            logger.error(f"Network error fetching metadata for {picto_id}: {e}")
            return {}
            
    def download_pictogram(self, picto_id: int, resolution: int = 500):
        cache_path = os.path.join(self.cache_dir, f"{picto_id}_{resolution}.png")
        if os.path.exists(cache_path):
            return cache_path
        url = f"https://static.arasaac.org/pictograms/{picto_id}/{picto_id}_{resolution}.png"
        try:
            response = self.session.get(url)
            if response.status_code == 200:
                with open(cache_path, 'wb') as f:
                    f.write(response.content)
                return cache_path
            else:
                logger.warning(f"Error downloading pictogram {picto_id}: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Network error downloading pictogram {picto_id}: {e}")
            return None