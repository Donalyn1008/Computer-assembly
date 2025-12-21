import pandas as pd
import numpy as np

class PCBuilderAI:
    def __init__(self, file_map):
        self.data = {}
        for key, path in file_map.items():
            try:
                self.data[key] = pd.read_csv(path)
                # 統一將價格分數轉為正數處理（您的資料中部分價格為負值）
                if 'price_分數' in self.data[key].columns:
                    self.data[key]['abs_price'] = self.data[key]['price_分數'].abs()
                elif 'Price_分數' in self.data[key].columns:
                    self.data[key]['abs_price'] = self.data[key]['Price_分數'].abs()
            except Exception as e:
                print(f"警告：無法讀取 {key}, 錯誤: {e}")

    def optimize_build(self, total_budget, prefs):
        """
        total_budget: 總預算 (例如 50000)
        prefs: {'color': 'white', 'cpu_brand': 'Intel', 'cooling': 'water'}
        """
  
        weights = {
            'VGA': 0.35, 'CPU': 0.20, 'MB': 0.12, 'RAM': 0.08,
            'SSD': 0.07, 'PSU': 0.07, 'CHASSIS': 0.06, 'FAN': 0.02, 'HDD': 0.03
        }
        
     
        cooling_type = 'WATER' if prefs.get('cooling') == 'water' else 'HEAT'
        weights[cooling_type] = 0.05
        
        build = {}
        current_spent = 0
        
        # --- 核心零件挑選 (CPU & VGA) ---
        for part in ['CPU', 'VGA']:
            df = self.data[part].copy()
            
            if part == 'CPU' and 'cpu_brand' in prefs:
                df = df[df['BRAND'].str.contains(prefs['cpu_brand'], case=False)]
            
            
            part_limit = total_budget * weights[part]
            affordable = df[df['abs_price'] <= part_limit]
            
           
            target = affordable if not affordable.empty else df
            build[part] = target.sort_values('總分', ascending=False).iloc[0]
            current_spent += build[part]['abs_price']

        # --- 相容性連鎖過濾 ---
        
        vga_length_score = build['VGA'].get('Length_分數', 0)
        chassis_pool = self.data['CHASSIS'].copy()
      
        chassis_pool = chassis_pool[chassis_pool['GPU_Max_Length_分數'] >= vga_length_score]
        
        if prefs.get('color') == 'white':
            chassis_pool = chassis_pool[chassis_pool['white_分數'] > 0]

      
        remaining_parts = ['MB', 'RAM', 'SSD', 'HDD', 'PSU', 'CHASSIS', 'FAN', cooling_type]
        
        for part in remaining_parts:
            if part not in self.data: continue
            
            df = self.data[part].copy()
            
           
            if prefs.get('color') == 'white' and 'white_分數' in df.columns:
                df = df[df['white_分數'] > 0]
            
           
            remaining_budget = total_budget - current_spent
           
            limit = max(remaining_budget * 0.15, 1000) 
            
            affordable = df[df['abs_price'] <= limit]
            target = affordable if not affordable.empty else df.sort_values('abs_price')
            
          
            choice = target.sort_values('總分', ascending=False).iloc[0]
            build[part] = choice
            current_spent += choice['abs_price']

        return build, current_spent
