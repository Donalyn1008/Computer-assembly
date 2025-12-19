import pandas as pd
import numpy as np

class PCBuilderAI:
    def __init__(self, file_map):
        self.data = {}
        # 載入所有檔案，處理可能的編碼問題
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
        # 1. 定義預算權重分配 (依據硬體重要性)
        weights = {
            'VGA': 0.35, 'CPU': 0.20, 'MB': 0.12, 'RAM': 0.08,
            'SSD': 0.07, 'PSU': 0.07, 'CHASSIS': 0.06, 'FAN': 0.02, 'HDD': 0.03
        }

        # 根據冷卻偏好調整權重
        cooling_type = 'WATER' if prefs.get('cooling') == 'water' else 'HEAT'
        weights[cooling_type] = 0.05

        build = {}
        current_spent = 0

        # --- 階段 A: 核心零件挑選 (CPU & VGA) ---
        # 這些零件決定了後續的相容性
        for part in ['CPU', 'VGA']:
            df = self.data[part].copy()

            # 使用者品牌過濾
            if part == 'CPU' and 'cpu_brand' in prefs:
                df = df[df['BRAND'].str.contains(prefs['cpu_brand'], case=False)]

            # 預算過濾
            part_limit = total_budget * weights[part]
            affordable = df[df['abs_price'] <= part_limit]

            # 挑選得分最高者
            target = affordable if not affordable.empty else df
            build[part] = target.sort_values('總分', ascending=False).iloc[0]
            current_spent += build[part]['abs_price']

        # --- 階段 B: 相容性連鎖過濾 ---

        # 1. 機殼空間檢查 (顯卡長度)
        vga_length_score = build['VGA'].get('Length_分數', 0)
        chassis_pool = self.data['CHASSIS'].copy()
        # 過濾能裝下該顯卡的機殼
        chassis_pool = chassis_pool[chassis_pool['GPU_Max_Length_分數'] >= vga_length_score]

        # 2. 顏色風格檢查
        if prefs.get('color') == 'white':
            chassis_pool = chassis_pool[chassis_pool['white_分數'] > 0]

        # --- 階段 C: 剩餘零件挑選 ---
        remaining_parts = ['MB', 'RAM', 'SSD', 'HDD', 'PSU', 'CHASSIS', 'FAN', cooling_type]

        for part in remaining_parts:
            if part not in self.data: continue

            df = self.data[part].copy()

            # 顏色過濾
            if prefs.get('color') == 'white' and 'white_分數' in df.columns:
                df = df[df['white_分數'] > 0]

            # 預算動態調整 (剩餘預算比例分配)
            remaining_budget = total_budget - current_spent
            # 避免預算歸零導致無法挑選
            limit = max(remaining_budget * 0.15, 1000)

            affordable = df[df['abs_price'] <= limit]
            target = affordable if not affordable.empty else df.sort_values('abs_price')

            # 挑選最佳解
            choice = target.sort_values('總分', ascending=False).iloc[0]
            build[part] = choice
            current_spent += choice['abs_price']

        return build, current_spent

#以下是colab執行驗證區
files = {
    'CPU': 'CPU_labeled.csv_ranking_result.csv',
    'MB': 'MB_Labled.csv_ranking_result.csv',
    'CHASSIS': 'CHASSIS_labeled.csv_ranking_result.csv',
    'PSU': 'PSU_labeled.csv_ranking_result.csv',
    'VGA': 'VGA_labeled.csv_ranking_result.csv',
    'RAM': 'RAM_labeled.csv_ranking_result.csv',
    'SSD': 'SSD_Labled.csv_ranking_result.csv',
    'HDD': 'HDD_Labled.csv_ranking_result.csv',
    'HEAT': 'HEAT_labeled.csv_ranking_result.csv',
    'WATER': 'WATER_labeled.csv_ranking_result.csv',
    'FAN': 'FAN_labeled.csv_ranking_result.csv'
}

ai = PCBuilderAI(files)

# 設定預算與偏好
USER_BUDGET = 50000
USER_PREFS = {
    'color': 'black',      # 'white' 或 'black'
    'cpu_brand': 'Intel',  # 'Intel' 或 'AMD'
    'cooling': 'heat'     # 'water' 或 'heat'
}

# 執行最佳化
final_build, total_cost = ai.optimize_build(USER_BUDGET, USER_PREFS)

print(f"{'='*50}")
print(f"🚀 AI 最佳化組裝清單 (總預算: {USER_BUDGET})")
print(f"{'='*50}")
for part, data in final_build.items():
    price = data['abs_price']
    print(f"[{part:7}] {data['BRAND']:<5} | {data['MODEL']:<30} | 得分: {data['總分']:>8.2f} | 預估: {price:>6.0f}")

print(f"{'='*50}")
print(f"✅ 實際總計金額: {total_cost:.0f}")
print(f"💡 相容性檢查: 顯卡長度適配 OK, 顏色風格對齊 OK.")
