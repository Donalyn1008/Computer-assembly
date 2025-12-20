from __future__ import annotations
import pandas as pd
import numpy as np

class PCBuilderAI:
    def __init__(self, file_map: dict[str, str]):
        self.data: dict[str, pd.DataFrame] = {}
        for key, path in file_map.items():
            try:
                df = pd.read_csv(path)
                df.columns = df.columns.astype(str).str.strip()
                df.columns = df.columns.str.replace("\u3000", " ", regex=False).str.strip()
                
              
                if "price_分數" in df.columns:
                    df["abs_price"] = df["price_分數"].abs()
                elif "Price_分數" in df.columns:
                    df["abs_price"] = df["Price_分數"].abs()
                elif "Price" in df.columns:
                    df["abs_price"] = pd.to_numeric(df["Price"], errors="coerce").fillna(0).abs()
                else:
                    df["abs_price"] = 0
                self.data[key] = df
            except Exception as e:
                print(f"[WARN] Failed to read '{key}' from '{path}': {e}")

    def _safe_str(self, x) -> str:
        if x is None: return ""
        try:
            if pd.isna(x): return ""
        except: pass
        return str(x).strip()

    def optimize_build(self, total_budget: int, prefs: dict) -> tuple[dict[str, pd.Series], float]:

        weights = {
            "VGA": 0.35, "CPU": 0.20, "MB": 0.12, "RAM": 0.08,
            "SSD": 0.07, "PSU": 0.07, "CHASSIS": 0.06, "FAN": 0.02, "HDD": 0.03
        }

        purpose = prefs.get("purpose")
        if purpose == "programming":
            weights["CPU"], weights["RAM"], weights["VGA"] = 0.35, 0.20, 0.10
        elif purpose == "video_editing":
            weights["CPU"], weights["RAM"], weights["SSD"] = 0.30, 0.15, 0.15
        elif purpose == "word_processing":
            weights["CPU"], weights["VGA"], weights["RAM"] = 0.25, 0.05, 0.10
            # 文書機通常不需要獨立顯卡，若無顯卡預算，後續過濾會選最便宜的

        cooling_pref = self._safe_str(prefs.get("cooling"))
        cooling_type = "WATER" if cooling_pref.lower() == "water" else "HEAT"
        weights[cooling_type] = 0.05


        specified_brands = prefs.get("specified_brands", {}) # 格式: {'CPU': 'Intel', 'VGA': 'ASUS'}

        build: dict[str, pd.Series] = {}
        current_spent = 0.0

        for part in ["CPU", "VGA"]:
            if part not in self.data: continue
            df = self.data[part].copy()

        
            target_brand = specified_brands.get(part, "")
            if target_brand and "BRAND" in df.columns:
                df = df[df["BRAND"].astype(str).str.contains(target_brand, case=False, na=False)]

            part_limit = float(total_budget) * weights.get(part, 0.10)
            affordable = df[df["abs_price"] <= part_limit] if "abs_price" in df.columns else df
            target = affordable if not affordable.empty else df
            
            if not target.empty:
                choice = target.sort_values("總分", ascending=False).iloc[0]
                build[part] = choice
                current_spent += float(choice.get("abs_price", 0))

        chassis_df = self.data.get("CHASSIS", pd.DataFrame()).copy()
        if not chassis_df.empty and "VGA" in build:
            vga_len = build["VGA"].get("Length_分數", 0)
            if "GPU_Max_Length_分數" in chassis_df.columns:
                chassis_df = chassis_df[chassis_df["GPU_Max_Length_分數"] >= vga_len]
        
        remaining_parts = ["MB", "RAM", "SSD", "HDD", "PSU", "CHASSIS", "FAN", cooling_type]
        for part in remaining_parts:
            if part == "CHASSIS":
                df = chassis_df
            elif part in self.data:
                df = self.data[part].copy()
            else:
                continue

            target_brand = specified_brands.get(part, "")
            if target_brand and "BRAND" in df.columns:
                df = df[df["BRAND"].astype(str).str.contains(target_brand, case=False, na=False)]

            remaining_budget = float(total_budget) - current_spent
      
            limit = max(remaining_budget * 0.2, 500.0) 

            affordable = df[df["abs_price"] <= limit] if "abs_price" in df.columns else df
            target = affordable if not affordable.empty else df.sort_values("abs_price")

            if not target.empty:
                choice = target.sort_values("總分", ascending=False).iloc[0]
                build[part] = choice
                current_spent += float(choice.get("abs_price", 0))

        return build, current_spent
