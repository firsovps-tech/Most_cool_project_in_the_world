import pandas as pd
import json
import os
import config

class DataLoader:
    def __init__(self):
        if os.path.exists(config.DATA_PATH):
            self.df = pd.read_csv(config.DATA_PATH, sep=';', dtype=str, encoding='utf-8')
            
            if 'ID' in self.df.columns:
                self.df['clean_id'] = self.df['ID'].str.replace(r's|_', '', regex=True)
        else:
            self.df = pd.DataFrame()
            print(f"Критическая ошибка: Файл {config.DATA_PATH} не найден")

    def load_type1(self):
        """Загрузка стандартных тестов (Point-wise)."""
        if os.path.exists(config.TYPE1_QUERIES_PATH):
            with open(config.TYPE1_QUERIES_PATH, "r", encoding="utf-8") as f:
                return json.load(f).get("queries", [])
        return []

    def load_type2(self):
        """Загрузка 'Золотого стандарта' (Top-20)."""
        if os.path.exists(config.TYPE2_QUERIES_PATH):
            with open(config.TYPE2_QUERIES_PATH, "r", encoding="utf-8") as f:
                return json.load(f).get("queries", [])
        return None
