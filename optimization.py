from __future__ import annotations
import pandas as pd
import numpy as np
import random


class PCBuilderAI:
    def __init__(self, file_map: dict[str, str]):
        self.data: dict[str, pd.DataFrame] = {}

        for key, path in file_map.items():
            try:
                df = pd.read_csv(path)

                # ---------- normalize columns ----------
                df.columns = (
                    df.columns.astype(str)
                    .str.replace("\u3000", " ", regex=False)
                    .str.strip()
                )

                # ---------- price ----------
                if "price_分數" in df.columns:
                    df["abs_price"] = pd.to_numeric(df["price_分數"], errors="coerce").abs()
                elif "Price_分數" in df.columns:
                    df["abs_price"] = pd.to_numeric(df["Price_分數"], errors="coerce").abs()
                elif "Price" in df.columns:
                    df["abs_price"] = pd.to_numeric(df["Price"], errors="coerce").abs()
                else:
                    df["abs_price"] = np.inf

                # ---------- score ----------
                if "總分" not in df.columns:
                    num_cols = df.select_dtypes(include=[np.number]).columns
                    df["總分"] = df[num_cols].mean(axis=1) if len(num_cols) else 0.0

                # ---------- key mapping ----------
                store_key = "COOLER" if key in {"HEAT", "WATER"} else key
                self.data.setdefault(store_key, pd.DataFrame())
                self.data[store_key] = pd.concat(
                    [self.data[store_key], df], ignore_index=True
                )

            except Exception as e:
                print(f"[WARN] Failed loading {key}: {e}")

    # ======================================================
    # Utilities
    # ======================================================
    def _safe_str(self, x) -> str:
        if x is None:
            return ""
        try:
            if pd.isna(x):
                return ""
        except Exception:
            pass
        return str(x).strip()

    # ======================================================
    # Compatibility checks
    # ======================================================
    def _cpu_mb_ok(self, cpu, mb) -> bool:
        if cpu is None or mb is None:
            return True
        return self._safe_str(cpu.get("Socket")) == self._safe_str(mb.get("Socket")) \
            if cpu.get("Socket") and mb.get("Socket") else True

    def _ram_mb_ok(self, ram, mb) -> bool:
        if ram is None or mb is None:
            return True
        return self._safe_str(ram.get("DDR_Type")) == self._safe_str(mb.get("DDR_Type")) \
            if ram.get("DDR_Type") and mb.get("DDR_Type") else True

    def _psu_vga_ok(self, psu, vga) -> bool:
        if psu is None or vga is None:
            return True
        try:
            return float(psu.get("Watt", 0)) >= float(vga.get("TDP", 0)) * 1.3
        except Exception:
            return True

    def _vga_case_ok(self, vga, case) -> bool:
        if vga is None or case is None:
            return True
        try:
            return float(case.get("GPU_Max_Length_分數", 0)) >= float(vga.get("Length_分數", 0))
        except Exception:
            return True

    # ======================================================
    # Main optimizer (Top-K stochastic)
    # ======================================================
    def optimize_build(
        self, total_budget: int, prefs: dict
    ) -> tuple[dict[str, pd.Series], float]:

        # ---------- weights ----------
        weights = {
            "CPU": 0.22, "VGA": 0.30, "MB": 0.12, "RAM": 0.10,
            "SSD": 0.08, "HDD": 0.04, "PSU": 0.08,
            "CHASSIS": 0.04, "COOLER": 0.04, "FAN": 0.02,
        }

        purpose = prefs.get("purpose")
        if purpose == "programming":
            weights.update({"CPU": 0.30, "RAM": 0.18, "VGA": 0.15})
        elif purpose == "video_editing":
            weights.update({"CPU": 0.28, "RAM": 0.16, "SSD": 0.14})
        elif purpose == "word_processing":
            weights.update({"CPU": 0.25, "VGA": 0.10})

        # normalize weights
        s = sum(weights.values())
        weights = {k: v / s for k, v in weights.items()}

        spec_brands = prefs.get("specified_brands", {})

        build: dict[str, pd.Series] = {}
        spent = 0.0

        order = [
            "CPU", "MB", "RAM", "VGA",
            "SSD", "HDD", "PSU",
            "CHASSIS", "COOLER", "FAN",
        ]

        TOP_K = 3  # ⭐ 核心參數：可以改 3 / 5

        for part in order:
            if part not in self.data:
                continue

            df = self.data[part].copy()

            # ---------- STRICT brand ----------
            brand = spec_brands.get(part)
            if brand and "BRAND" in df.columns:
                df = df[df["BRAND"].astype(str).str.contains(brand, case=False, na=False)]
                if df.empty:
                    return {}, 0.0

            # ---------- budget ----------
            remaining = total_budget - spent
            part_budget = total_budget * weights.get(part, 0.05)
            limit = min(part_budget, remaining)

            df = df[df["abs_price"] <= limit]
            if df.empty:
                return {}, 0.0

            # ---------- compatibility ----------
            if part == "MB" and "CPU" in build:
                df = df[df.apply(lambda r: self._cpu_mb_ok(build["CPU"], r), axis=1)]
            if part == "RAM" and "MB" in build:
                df = df[df.apply(lambda r: self._ram_mb_ok(r, build["MB"]), axis=1)]
            if part == "PSU" and "VGA" in build:
                df = df[df.apply(lambda r: self._psu_vga_ok(r, build["VGA"]), axis=1)]
            if part == "CHASSIS" and "VGA" in build:
                df = df[df.apply(lambda r: self._vga_case_ok(build["VGA"], r), axis=1)]

            if df.empty:
                return {}, 0.0

            # ==================================================
            # ⭐ Top-K stochastic selection
            # ==================================================
            topk = df.sort_values("總分", ascending=False).head(TOP_K)

            scores = topk["總分"].astype(float)
            probs = scores / scores.sum()   # 分數越高，機率越高

            choice = topk.sample(1, weights=probs).iloc[0]


            build[part] = choice
            spent += float(choice.get("abs_price", 0))

        return build, spent
