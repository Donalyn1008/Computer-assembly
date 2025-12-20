from __future__ import annotations
import pandas as pd
import numpy as np
import random

TOP_K = 3  # Top-K 高分零件數量

class PCBuilderAI:
    def __init__(self, file_map: dict[str, str]):
        self.data: dict[str, pd.DataFrame] = {}
        for key, path in file_map.items():
            df = pd.read_csv(path)
            df.columns = (
                df.columns.astype(str)
                .str.replace("\u3000", " ", regex=False)
                .str.strip()
            )

            # 建立 abs_price 欄位
            if "price_分數" in df.columns:
                df["abs_price"] = df["price_分數"].abs()
            elif "Price" in df.columns:
                df["abs_price"] = pd.to_numeric(df["Price"], errors="coerce").fillna(0)
            else:
                df["abs_price"] = 0

            self.data[key] = df

    # -------------------------
    # 相容性檢查
    # -------------------------
    def _compatible(self, build, part, row) -> bool:
        if part == "MB" and "CPU" in build:
            if "Socket" in row and "Socket" in build["CPU"]:
                return row["Socket"] == build["CPU"]["Socket"]
        if part == "RAM" and "MB" in build:
            if "RAM_Type" in row and "RAM_Type" in build["MB"]:
                return row["RAM_Type"] == build["MB"]["RAM_Type"]
        if part == "CHASSIS" and "VGA" in build:
            if "GPU_Max_Length" in row and "Length" in build["VGA"]:
                return row["GPU_Max_Length"] >= build["VGA"]["Length"]
        return True

    # -------------------------
    # Top-K + 價格感知選擇
    # -------------------------
    def _pick_one(self, df, budget, total_budget, fixed_brand=False):
        df = df[df["abs_price"] <= budget]
        if df.empty:
            return None

        df = df.sort_values("總分", ascending=False).head(TOP_K)

        if fixed_brand:
            # 指定品牌 → 直接拿總分最高的零件，結果固定
            return df.iloc[0]

        # 未指定品牌 → Top-K 隨機選擇，價格加權
        score = df["總分"].astype(float)
        price = df["abs_price"].astype(float)
        alpha = 0.3 * (budget / max(total_budget, 1))
        final = score - alpha * (price / price.max())
        final = final.clip(lower=0.0001)
        return df.sample(1, weights=final).iloc[0]

    # -------------------------
    # 主流程
    # -------------------------
    def optimize_build(self, total_budget: int, prefs: dict):
        specified_brands = prefs.get("specified_brands", {})
        purpose = prefs.get("purpose")

        weights = {
            "CPU": 0.25,
            "VGA": 0.30,
            "MB": 0.15,
            "RAM": 0.10,
            "SSD": 0.10,
            "HDD": 0.05,
            "PSU": 0.05,
            "CHASSIS": 0.05,
            "FAN": 0.02,
        }

        if purpose == "programming":
            weights.update({"CPU": 0.35, "VGA": 0.15})
        elif purpose == "video_editing":
            weights.update({"CPU": 0.30, "RAM": 0.15, "SSD": 0.15})

        # cooling 對應 HEAT / WATER
        cooling = prefs.get("cooling", "heat")
        cooler_key = "WATER" if cooling.lower() == "water" else "HEAT"
        weights[cooler_key] = 0.05

        # 建議順序
        order = ["CPU", "VGA", "MB", "RAM", "SSD", "HDD", "PSU", "CHASSIS", cooler_key, "FAN"]

        build = {}
        spent = 0.0

        for part in order:
            if part not in self.data:
                continue

            df = self.data[part].copy()

            # 指定品牌（硬限制）
            brand = specified_brands.get(part)
            fixed_brand = False
            if brand and "BRAND" in df.columns:
                mask = df["BRAND"].astype(str).str.contains(brand, case=False, na=False)
                df = df[mask]
                if df.empty:
                    continue
                fixed_brand = True  # 標記此零件使用固定選擇

            # 相容性
            df = df[df.apply(lambda r: self._compatible(build, part, r), axis=1)]

            remaining = total_budget - spent
            part_budget = remaining * weights.get(part, 0.1)

            choice = self._pick_one(df, part_budget, total_budget, fixed_brand=fixed_brand)
            if choice is None:
                continue

            build[part] = choice
            spent += float(choice["abs_price"])

        return build, spent
