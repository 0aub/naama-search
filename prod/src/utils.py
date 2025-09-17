"""
Production Data Loader - Winner Configuration ONLY
"""

import pandas as pd
from langchain.docstore.document import Document
from typing import List

class ProductionDataLoader:
    """
    Hardcoded data loader for production.
    Only loads the specific Excel file format needed.
    """

    def __init__(self):
        # Hardcoded Excel path and column mapping
        self.excel_path = 'data/NaamaServiceIn full Details.xlsx'
        self.rename_map = {
            'الاسم عربي': 'service',
            'التصنيف عربي': 'classification',
            'القطاع عربي': 'sector',
            'الوصف المختصر عربي': 'description_short',
            'الوصف عربي': 'description',
            'المستفيدين من الخدمة': 'beneficiaries',
        }
        self.combine_cols = (
            'service', 'service', 'service', 'classification',
            'sector', 'description_short', 'description', 'beneficiaries'
        )

        self.documents = self._load_documents()

    def _load_documents(self) -> List[Document]:
        """Load and process Excel data into documents"""
        df = pd.read_excel(self.excel_path)
        df = df.rename(columns=self.rename_map)

        documents = []
        for _, row in df.iterrows():
            # Combine columns as specified
            combined_text = ' '.join([
                str(row.get(col, '')) for col in self.combine_cols
                if pd.notna(row.get(col, ''))
            ])

            doc = Document(
                page_content=combined_text,
                metadata={
                    'service': row.get('service', ''),
                    'classification': row.get('classification', ''),
                    'sector': row.get('sector', ''),
                    'description_short': row.get('description_short', ''),
                    'description': row.get('description', ''),
                    'beneficiaries': row.get('beneficiaries', ''),
                }
            )
            documents.append(doc)

        return documents