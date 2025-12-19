from __future__ import annotations

import pandas as pd
import numpy as np


class PCBuilderAI:
    def __init__(self, file_map: dict[str, str]):
        self.data: dict[str, pd.DataFrame] = {}

        for key, path in file_map.items():
            try:
                df = pd.read_csv(path)

                # Normalize price score to positive values
                if "price_分數" in df.columns:
                    df["abs_price"] = df["price_分數"].abs()
                elif "Price_分數" in df.columns:
                    df["abs_price"] = df["Price_分數"].abs()
                elif "Price" in df.columns:
                    df["abs_price"] = pd.to_numeric(df["Price"], errors="coerce").fillna(0).abs()
                else:
                    # still allow running, but prices will be 0
                    df["abs_price"] = 0

                self.data[key] = df

            except Exception as e:
                # Do NOT crash at init; just record missing dataset
                print(f"[WARN] Failed to read '{key}' from '{path}': {e}")

    def _safe_str(self, x) -> str:
        """Convert anything to a safe string for matching."""
        if x is None:
            return ""
        # handle numpy/pandas NaN
        try:
            if pd.isna(x):
                return ""
        except Exception:
            pass
        return str(x).strip()

    def optimize_build(self, total_budget: int, prefs: dict) -> tuple[dict[str, pd.Series], float]:
        """
        total_budget: e.g. 50000
        prefs: {
            'color': 'white' or 'black',
            'cpu_brand': 'Intel' or 'AMD' or '',
            'cooling': 'water' or 'heat'
        }
        """

        # Ensure required core datasets exist
        missing = [k for k in ["CPU", "VGA", "CHASSIS"] if k not in self.data]
        if missing:
            raise KeyError(
                f"Missing required datasets: {missing}. "
                f"Loaded keys = {sorted(self.data.keys())}. "
                f"Check your /data filenames and FILE_MAP paths."
            )

        # 1) Budget weight allocation
        weights = {
            "VGA": 0.35, "CPU": 0.20, "MB": 0.12, "RAM": 0.08,
            "SSD": 0.07, "PSU": 0.07, "CHASSIS": 0.06, "FAN": 0.02, "HDD": 0.03
        }

        cooling_pref = self._safe_str(prefs.get("cooling"))
        cooling_type = "WATER" if cooling_pref.lower() == "water" else "HEAT"
        weights[cooling_type] = 0.05

        color_pref = self._safe_str(prefs.get("color")).lower()  # 'white' or 'black'
        cpu_brand = self._safe_str(prefs.get("cpu_brand"))       # safe string always

        build: dict[str, pd.Series] = {}
        current_spent = 0.0

        # --- Stage A: Core parts (CPU & VGA) ---
        for part in ["CPU", "VGA"]:
            df = self.data[part].copy()

            # CPU brand filter (SAFE)
            if part == "CPU" and cpu_brand != "" and "BRAND" in df.columns:
                df = df[df["BRAND"].astype(str).str.contains(cpu_brand, case=False, na=False, regex=False)]

            # Budget filter
            part_limit = float(total_budget) * float(weights.get(part, 0.10))
            if "abs_price" in df.columns:
                affordable = df[df["abs_price"] <= part_limit]
            else:
                affordable = df

            target = affordable if not affordable.empty else df
            if target.empty:
                raise ValueError(f"Dataset '{part}' has no rows after filtering.")

            # Pick the highest score
            if "總分" not in target.columns:
                raise KeyError(f"Dataset '{part}' missing required column '總分'.")

            choice = target.sort_values("總分", ascending=False).iloc[0]
            build[part] = choice
            current_spent += float(choice.get("abs_price", 0))

        # --- Stage B: Compatibility chain filtering (Case GPU length + color) ---
        chassis_df = self.data["CHASSIS"].copy()

        vga_length_score = build["VGA"].get("Length_分數", 0)
        if "GPU_Max_Length_分數" in chassis_df.columns:
            chassis_df = chassis_df[chassis_df["GPU_Max_Length_分數"] >= vga_length_score]

        if color_pref == "white" and "white_分數" in chassis_df.columns:
            chassis_df = chassis_df[chassis_df["white_分數"] > 0]

        # --- Stage C: Remaining parts ---
        remaining_parts = ["MB", "RAM", "SSD", "HDD", "PSU", "CHASSIS", "FAN", cooling_type]

        for part in remaining_parts:
            if part == "CHASSIS":
                df = chassis_df.copy()
            else:
                if part not in self.data:
                    # allow missing optional datasets
                    continue
                df = self.data[part].copy()

            # Color filter
            if color_pref == "white" and "white_分數" in df.columns:
                df = df[df["white_分數"] > 0]

            # Dynamic budget allocation
            remaining_budget = float(total_budget) - current_spent
            limit = max(remaining_budget * 0.15, 1000.0)

            affordable = df[df["abs_price"] <= limit] if "abs_price" in df.columns else df
            target = affordable if not affordable.empty else df.sort_values("abs_price") if "abs_price" in df.columns else df

            if target.empty:
                continue

            if "總分" not in target.columns:
                raise KeyError(f"Dataset '{part}' missing required column '總分'.")

            choice = target.sort_values("總分", ascending=False).iloc[0]
            build[part] = choice
            current_spent += float(choice.get("abs_price", 0))

        return build, current_spent


# =========================
# Local test (ONLY runs when executing this file directly)
# =========================
if __name__ == "__main__":
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
        "color": "black",
        "cpu_brand": "Intel",
        "cooling": "heat",
    }

    build, total_cost = ai.optimize_build(USER_BUDGET, USER_PREFS)

    print("=" * 50)
    print(f"🚀 AI Optimized Build (Budget: {USER_BUDGET})")
    print("=" * 50)
    for part, row in build.items():
        price = float(row.get("abs_price", 0))
        brand = str(row.get("BRAND", ""))
        model = str(row.get("MODEL", ""))
        score = float(row.get("總分", 0))
        print(f"[{part:7}] {brand:<10} | {model:<35} | score: {score:>8.2f} | price: {price:>8.0f}")
    print("=" * 50)
    print(f"✅ Total Cost: {total_cost:.0f}")
