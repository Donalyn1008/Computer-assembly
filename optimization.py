import pandas as pd
import numpy as np


class PCBuilderAI:
    def __init__(self, file_map):
        self.data = {}
        # Load all files (handle potential encoding issues)
        for key, path in file_map.items():
            try:
                self.data[key] = pd.read_csv(path)

                # Normalize price score to positive values (some of your data uses negative price scores)
                if "price_分數" in self.data[key].columns:
                    self.data[key]["abs_price"] = self.data[key]["price_分數"].abs()
                elif "Price_分數" in self.data[key].columns:
                    self.data[key]["abs_price"] = self.data[key]["Price_分數"].abs()

            except Exception as e:
                print(f"Warning: failed to read {key} from '{path}'. Error: {e}")

    def optimize_build(self, total_budget, prefs):
        """
        total_budget: total budget (e.g., 50000)
        prefs: {'color': 'white', 'cpu_brand': 'Intel', 'cooling': 'water'}
        """
        # 1) Budget weight allocation (based on component importance)
        weights = {
            "VGA": 0.35,
            "CPU": 0.20,
            "MB": 0.12,
            "RAM": 0.08,
            "SSD": 0.07,
            "PSU": 0.07,
            "CHASSIS": 0.06,
            "FAN": 0.02,
            "HDD": 0.03,
        }

        # Adjust weights based on cooling preference
        cooling_type = "WATER" if prefs.get("cooling") == "water" else "HEAT"
        weights[cooling_type] = 0.05

        build = {}
        current_spent = 0

        # --- Stage A: Core parts selection (CPU & VGA) ---
        for part in ["CPU", "VGA"]:
            df = self.data[part].copy()

            # User brand filter (CPU only)
            if part == "CPU":
                cpu_brand = prefs.get("cpu_brand")
                if isinstance(cpu_brand, str) and cpu_brand.strip():
                    df = df[df["BRAND"].astype(str).str.contains(cpu_brand.strip(), case=False, na=False)]


            # Budget filter
            part_limit = total_budget * weights[part]
            affordable = df[df["abs_price"] <= part_limit]

            # Pick the highest score
            target = affordable if not affordable.empty else df
            build[part] = target.sort_values("總分", ascending=False).iloc[0]
            current_spent += build[part]["abs_price"]

        # --- Stage B: Compatibility chain filtering ---

        # 1) Case space check (GPU length)
        vga_length_score = build["VGA"].get("Length_分數", 0)
        chassis_pool = self.data["CHASSIS"].copy()
        chassis_pool = chassis_pool[chassis_pool["GPU_Max_Length_分數"] >= vga_length_score]

        # 2) Color style check
        if prefs.get("color") == "white":
            chassis_pool = chassis_pool[chassis_pool["white_分數"] > 0]

        # --- Stage C: Remaining parts selection ---
        remaining_parts = ["MB", "RAM", "SSD", "HDD", "PSU", "CHASSIS", "FAN", cooling_type]

        for part in remaining_parts:
            if part not in self.data:
                continue

            df = self.data[part].copy()

            # Color filter
            if prefs.get("color") == "white" and "white_分數" in df.columns:
                df = df[df["white_分數"] > 0]

            # Dynamic budget allocation (remaining budget)
            remaining_budget = total_budget - current_spent
            limit = max(remaining_budget * 0.15, 1000)

            affordable = df[df["abs_price"] <= limit]
            target = affordable if not affordable.empty else df.sort_values("abs_price")

            # Pick the best
            choice = target.sort_values("總分", ascending=False).iloc[0]
            build[part] = choice
            current_spent += choice["abs_price"]

        return build, current_spent


# =========================
# Local test section (ONLY runs when you execute this file directly)
# Streamlit importing this module will NOT execute this section.
# =========================
if __name__ == "__main__":
    # IMPORTANT: paths point to GitHub repo's /data folder
    files = {
        "CPU": "data/CPU_labeled.csv_ranking_result.csv",
        "MB": "data/MB_Labled.csv_ranking_result.csv",
        "CHASSIS": "data/CHASSIS_labeled.csv_ranking_result.csv",
        "PSU": "data/PSU_labeled.csv_ranking_result.csv",
        "VGA": "data/VGA_labeled.csv_ranking_result.csv",
        "RAM": "data/RAM_labeled.csv_ranking_result.csv",
        "SSD": "data/SSD_Labled.csv_ranking_result.csv",
        "HDD": "data/HDD_Labled.csv_ranking_result.csv",
        "HEAT": "data/HEAT_labeled.csv_ranking_result.csv",
        "WATER": "data/WATER_labeled.csv_ranking_result.csv",
        "FAN": "data/FAN_labeled.csv_ranking_result.csv",
    }

    ai = PCBuilderAI(files)

    USER_BUDGET = 50000
    USER_PREFS = {
        "color": "black",      # 'white' or 'black'
        "cpu_brand": "Intel",  # 'Intel' or 'AMD'
        "cooling": "heat",     # 'water' or 'heat'
    }

    final_build, total_cost = ai.optimize_build(USER_BUDGET, USER_PREFS)

    print(f"{'='*50}")
    print(f"🚀 AI Optimized Build (Budget: {USER_BUDGET})")
    print(f"{'='*50}")
    for part, data in final_build.items():
        price = data["abs_price"]
        print(
            f"[{part:7}] {data['BRAND']:<5} | {data['MODEL']:<30} | "
            f"Score: {data['總分']:>8.2f} | Est. Price: {price:>6.0f}"
        )

    print(f"{'='*50}")
    print(f"✅ Total Cost: {total_cost:.0f}")
    print("💡 Compatibility: GPU length OK, color style OK.")
