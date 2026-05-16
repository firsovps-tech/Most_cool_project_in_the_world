class SearchInterface:
    def __init__(self, data_df):
        self.df = data_df

    def get_results(self, query, k=20):
        if self.df.empty:
            return []
        
        # Заглушка: берем случайные товары
        sample = self.df.sample(min(k, len(self.df)))
        
        results = []
        for _, row in sample.iterrows():
            title = f"{row.get('Тип товара', '')} {row.get('Модель', '')}".strip()
            attributes = f"Цвет: {row.get('Цвет', '-')}, Материал: {row.get('Материал', '-')}"
            
            results.append({
                "id": row['clean_id'],
                "title": f"{title} ({attributes})"
            })
        return results
