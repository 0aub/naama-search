"""
Production Data Loader - Winner Configuration ONLY
"""

from langchain.docstore.document import Document
from typing import List, Dict
import requests
from bs4 import BeautifulSoup
import re
import pickle
import os
from datetime import datetime

def load_services_from_internet() -> Dict:
    """
    Load services data from naama.sa website
    Returns a dictionary with service data
    """
    # Clean Arabic text function
    sen_ar = lambda x: re.sub(r'\s\s+', ' ', re.sub(r'[^\u0621-\u064A]', ' ', x)).strip()
    
    try:
        html = BeautifulSoup(requests.get("https://naama.sa/services/search").text, 'html.parser')
        all_service = html.find_all(class_='service-content')
        
        service_map = {}
        for service in all_service:
            _id_href = service.get('href')
            
            # Initialize service_id to None
            service_id = None
            
            # Check for the two different ways the ID is stored
            if _id_href and _id_href.startswith("/services/details"):
                service_id = _id_href.split("id=")[-1]
            else:
                service_id = service.get('data-id')

            # Make sure we found an ID and the necessary text elements before proceeding
            service_title_element = service.find(class_='service-title')
            service_desc_element = service.find(class_='service-desc')

            if service_id and service_title_element and service_desc_element:
                service_map[service_id] = {
                    "name": sen_ar(service_title_element.text),
                    "desc": sen_ar(service_desc_element.text)
                }
        
        # Save data with current date
        current_date = datetime.now().strftime("%Y-%m-%d")
        cache_filename = f'data/naama_services_{current_date}.pkl'
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        with open(cache_filename, 'wb') as file:
            pickle.dump(service_map, file)
        
        print(f"Found and processed {len(service_map)} services.")
        print(f"Data saved to {cache_filename}")
        
        return service_map
        
    except Exception as e:
        print(f"Error loading services from internet: {e}")
        return {}

class ProductionDataLoader:
    """
    Production data loader that loads services from naama.sa website.
    """

    def __init__(self):
        self.documents = self._load_documents()

    def _load_documents(self) -> List[Document]:
        """Load and process internet data into documents"""
        # Load services from internet
        service_map = load_services_from_internet()
        
        documents = []
        for service_id, service_data in service_map.items():
            service_name = service_data.get('name', '')
            service_desc = service_data.get('desc', '')
            
            # Combine name and description
            combined_text = f"{service_name} {service_desc}".strip()
            
            doc = Document(
                page_content=combined_text,
                metadata={
                    'service': service_name,
                    'classification': 'خدمة حكومية',  # Default classification
                    'sector': 'القطاع الحكومي',  # Default sector
                    'description_short': service_name,
                    'description': service_desc,
                    'beneficiaries': 'المواطنين والمقيمين',  # Default beneficiaries
                    'service_id': service_id
                }
            )
            documents.append(doc)

        return documents