import os
import random
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple


class NSGAPCBuilder:

    def __init__(self, file_map: Dict[str, str]):
        self.data: Dict[str, pd.DataFrame] = {}
        # population setting
        self.pop_size = 40
        self.generations = 40
        self.mutation_rate = 0.2
        self.crossover_rate = 0.8
        self.elitism = 0.1          # 保留前緣精英比例（非嚴格使用，NSGA-II 已含精英）
        self.tournament_size = 2
        self.crossover_type = "uniform"  # single_point / uniform
        self.mutation_type = "swap"           # swap: 隨機替換 index 或切換 None

        # 預算可接受範圍
        self.budget_tolerance = 0.1 # 10%
        for key, path in file_map.items():
            try:
                df = pd.read_csv(path, encoding="utf-8-sig")
                # 統一價格絕對值欄位：優先使用真實價格 PRICE/Price
                if "PRICE" in df.columns:
                    df["abs_price"] = pd.to_numeric(df["PRICE"], errors="coerce").fillna(0).abs()
                elif "Price" in df.columns:
                    df["abs_price"] = pd.to_numeric(df["Price"], errors="coerce").fillna(0).abs()
                elif "price_分數" in df.columns:
                    df["abs_price"] = df["price_分數"].abs()
                elif "Price_分數" in df.columns:
                    df["abs_price"] = df["Price_分數"].abs()
                else:
                    # 若沒有價格欄，fallback 為 0
                    df["abs_price"] = 0
                self.data[key] = df.reset_index(drop=True)
            except Exception as e:
                print(f"Warning: Could not read {key}, Error: {e}")

    # -------- NSGA-II 核心工具 --------
    def _fast_non_dominated_sort(self, objectives: List[Tuple[float, float]]) -> List[List[int]]:
        S = [[] for _ in range(len(objectives))]
        n = [0 for _ in range(len(objectives))]
        rank = [0 for _ in range(len(objectives))]
        fronts = [[]]

        for p in range(len(objectives)):
            for q in range(len(objectives)):
                if self._dominates(objectives[p], objectives[q]):
                    S[p].append(q)
                elif self._dominates(objectives[q], objectives[p]):
                    n[p] += 1
            if n[p] == 0:
                rank[p] = 0
                fronts[0].append(p)

        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in S[p]:
                    n[q] -= 1
                    if n[q] == 0:
                        rank[q] = i + 1
                        next_front.append(q)
            i += 1
            fronts.append(next_front)

        fronts.pop()
        return fronts

    def _dominates(self, a: Tuple[float, float], b: Tuple[float, float]) -> bool:
        return (a[0] <= b[0] and a[1] <= b[1]) and (a[0] < b[0] or a[1] < b[1])

    def _crowding_distance(self, front: List[int], objectives: List[Tuple[float, float]]) -> Dict[int, float]:
        distance = {i: 0 for i in front}
        if not front:
            return distance
        for m in range(2):  # 兩個目標
            front_sorted = sorted(front, key=lambda x: objectives[x][m])
            distance[front_sorted[0]] = distance[front_sorted[-1]] = float("inf")
            vals = [objectives[i][m] for i in front_sorted]
            min_v, max_v = min(vals), max(vals)
            if max_v - min_v == 0:
                continue
            for i in range(1, len(front_sorted) - 1):
                prev_v = objectives[front_sorted[i - 1]][m]
                next_v = objectives[front_sorted[i + 1]][m]
                distance[front_sorted[i]] += (next_v - prev_v) / (max_v - min_v)
        return distance

    # -------- 偏好與資料前處理 --------
    def _filter_df_by_prefs(self, part: str, df: pd.DataFrame, prefs: Dict[str, Any]) -> pd.DataFrame:
        df_filtered = df.copy()
        specified_brands = prefs.get("specified_brands", {})
        if part in specified_brands and specified_brands[part]:
            brand = specified_brands[part]
            df_filtered = df_filtered[df_filtered.get("BRAND", "").astype(str).str.contains(brand, case=False, na=False)]
            if df_filtered.empty:
                df_filtered = df.copy()

        # 顏色偏好（白色）
        if prefs.get("color") == "white" and "white_分數" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["white_分數"] > 0]
            if df_filtered.empty:
                df_filtered = df.copy()
        return df_filtered.reset_index(drop=True)

    def _purpose_weights(self, purpose: Optional[str], cooling_type: str) -> Dict[str, float]:
        if purpose == "programming":
            weights = {"CPU": 0.40, "VGA": 0.05, "MB": 0.12, "RAM": 0.15,
                       "SSD": 0.10, "PSU": 0.07, "CHASSIS": 0.06, "FAN": 0.02, "HDD": 0.03}
        elif purpose == "graphic_design":
            weights = {"CPU": 0.20, "VGA": 0.35, "MB": 0.10, "RAM": 0.15,
                       "SSD": 0.08, "PSU": 0.07, "CHASSIS": 0.03, "FAN": 0.01, "HDD": 0.01}
        elif purpose == "video_editing":
            weights = {"CPU": 0.25, "VGA": 0.25, "MB": 0.10, "RAM": 0.15,
                       "SSD": 0.10, "PSU": 0.07, "CHASSIS": 0.05, "FAN": 0.01, "HDD": 0.02}
        elif purpose == "word_processing":
            weights = {"CPU": 0.20, "VGA": 0.10, "MB": 0.15, "RAM": 0.15,
                       "SSD": 0.15, "PSU": 0.10, "CHASSIS": 0.10, "FAN": 0.03, "HDD": 0.02}
        else:
            weights = {"VGA": 0.35, "CPU": 0.20, "MB": 0.12, "RAM": 0.08,
                       "SSD": 0.07, "PSU": 0.07, "CHASSIS": 0.06, "FAN": 0.02, "HDD": 0.03}

        # 冷卻器類型
        weights[cooling_type] = 0.05
        return weights

    def _required_optional_parts(self, purpose: Optional[str], cooling_type: str) -> Tuple[List[str], List[str]]:
        base_required = ["CPU", "VGA", "MB", "RAM", "SSD", "PSU", "CHASSIS"]
        optional = ["HDD", "FAN"]
        if purpose == "word_processing":
            # 低需求：冷卻器可選
            required = base_required
            optional = optional + [cooling_type]
        else:
            # 其他情境：冷卻器必選
            required = base_required + [cooling_type]
        return required, optional

    # -------- 個體建立與評估 --------
    def _create_individual(self, parts_pool: Dict[str, pd.DataFrame], required: List[str], optional: List[str],
                           all_parts: List[str]) -> Dict[str, Optional[int]]:
        indiv = {}
        for part in all_parts:
            df = parts_pool.get(part, pd.DataFrame())
            if part in required:
                if df.empty:
                    indiv[part] = None
                else:
                    indiv[part] = random.randrange(len(df))
            else:
                # optional: 50% 機率不選
                if random.random() < 0.5 or df.empty:
                    indiv[part] = None
                else:
                    indiv[part] = random.randrange(len(df))
        return indiv

    def _evaluate(self, indiv: Dict[str, Optional[int]], parts_pool: Dict[str, pd.DataFrame],
                  required: List[str], weights: Dict[str, float],
                  total_budget: float) -> Tuple[float, float, float, bool]:
        total_cost = 0.0
        total_score = 0.0
        penalty = 0.0

        # 取 VGA 長度，供機殼相容性
        vga_len_score = 0
        if "VGA" in indiv and indiv["VGA"] is not None:
            vga_row = parts_pool["VGA"].iloc[indiv["VGA"]]
            vga_len_score = vga_row.get("Length_分數", 0)

        for part, idx in indiv.items():
            if idx is None:
                if part in required:
                    penalty += 1e6  # 缺必選件
                continue
            row = parts_pool[part].iloc[idx]
            cost = row.get("abs_price", 0)
            score = row.get("總分", 0)
            total_cost += cost
            total_score += weights.get(part, 0) * score

            # 機殼相容性：GPU 長度
            if part == "CHASSIS" and vga_len_score:
                chassis_len = row.get("GPU_Max_Length_分數", vga_len_score)
                if chassis_len < vga_len_score:
                    penalty += 1e5  # 不相容

        # 預算懲罰：超出預算上限給極大懲罰
        upper = total_budget * (1 + self.budget_tolerance)
        deviation = abs(total_cost - total_budget)
        feasible = not (total_cost > upper)
        if not feasible:
            # 硬約束：直接視為不可行
            penalty = float("inf")

        # 目標1: 最大化分數 -> 最小化 -score
        # 目標2: 花費最接近預算 -> 最小化 deviation + penalty
        obj1 = -total_score
        obj2 = deviation + penalty
        return obj1, obj2, total_score, feasible

    # -------- 交叉與突變 --------
    def _crossover(self, p1: Dict[str, Optional[int]], p2: Dict[str, Optional[int]], parts: List[str]) -> Tuple[Dict[str, Optional[int]], Dict[str, Optional[int]]]:
        if len(parts) < 2:
            return p1.copy(), p2.copy()
        c1, c2 = {}, {}
        if self.crossover_type == "single_point":
            point = random.randint(1, len(parts) - 1)
            for i, part in enumerate(parts):
                if i < point:
                    c1[part] = p1[part]
                    c2[part] = p2[part]
                else:
                    c1[part] = p2[part]
                    c2[part] = p1[part]
        elif self.crossover_type == "uniform":
            for part in parts:
                if random.random() < 0.5:
                    c1[part] = p1[part]
                    c2[part] = p2[part]
                else:
                    c1[part] = p2[part]
                    c2[part] = p1[part]
        else:
            # fallback: single_point
            point = random.randint(1, len(parts) - 1)
            for i, part in enumerate(parts):
                if i < point:
                    c1[part] = p1[part]
                    c2[part] = p2[part]
                else:
                    c1[part] = p2[part]
                    c2[part] = p1[part]
        return c1, c2

    def _mutate(self, indiv: Dict[str, Optional[int]], parts_pool: Dict[str, pd.DataFrame],
                required: List[str], optional: List[str]):
        for part, df in parts_pool.items():
            if random.random() < self.mutation_rate:
                if self.mutation_type == "swap":
                    if part in required:
                        if not df.empty:
                            indiv[part] = random.randrange(len(df))
                    else:
                        if random.random() < 0.5 or df.empty:
                            indiv[part] = None
                        else:
                            indiv[part] = random.randrange(len(df))
                else:
                    # fallback same as swap
                    if part in required:
                        if not df.empty:
                            indiv[part] = random.randrange(len(df))
                    else:
                        if random.random() < 0.5 or df.empty:
                            indiv[part] = None
                        else:
                            indiv[part] = random.randrange(len(df))

    # -------- 主流程：NSGA-II 優化 --------
    def optimize_build(self, total_budget: float, prefs: Dict[str, Any],
                       pop_size: Optional[int] = None, generations: Optional[int] = None) -> Tuple[Dict[str, Any], float]:
        pop_size = pop_size or self.pop_size
        generations = generations or self.generations
        purpose = prefs.get("purpose")
        cooling_type = "WATER" if prefs.get("cooling") == "water" else "HEAT"
        weights = self._purpose_weights(purpose, cooling_type)
        required, optional = self._required_optional_parts(purpose, cooling_type)
        upper_budget = total_budget * 1.05  # 用於前置過濾的單件價格上限

        # 準備零件池（依偏好過濾）
        parts_pool: Dict[str, pd.DataFrame] = {}
        for part, df in self.data.items():
            df_filtered = self._filter_df_by_prefs(part, df, prefs)
            # 先過濾遠高於預算的單件（> 預算上限），避免無謂昂貴件
            if "abs_price" in df_filtered.columns:
                affordable = df_filtered[df_filtered["abs_price"] <= upper_budget]
                if not affordable.empty:
                    df_filtered = affordable
            df_filtered = df_filtered.reset_index(drop=True)
            parts_pool[part] = df_filtered.reset_index(drop=True)

        # 僅保留實際有資料的零件類別，避免 crossover 時缺鍵
        all_parts = [p for p in set(required + optional) if p in parts_pool]

        # 初始化族群
        population = [self._create_individual(parts_pool, required, optional, all_parts) for _ in range(pop_size)]

        for _ in range(generations):
            # 評估
            objectives = []
            scores = []
            for indiv in population:
                obj1, obj2, sc, feasible = self._evaluate(indiv, parts_pool, required, weights, total_budget)
                # 不可行者設為極大值，確保被支配
                if not feasible:
                    obj1 = float("inf")
                    obj2 = float("inf")
                objectives.append((obj1, obj2))
                scores.append(sc)

            # 非支配排序
            fronts = self._fast_non_dominated_sort(objectives)
            new_population = []

            for front in fronts:
                if len(new_population) + len(front) > pop_size:
                    distance = self._crowding_distance(front, objectives)
                    front_sorted = sorted(front, key=lambda i: distance[i], reverse=True)
                    needed = pop_size - len(new_population)
                    new_population.extend([population[i] for i in front_sorted[:needed]])
                    break
                else:
                    new_population.extend([population[i] for i in front])

            population = new_population

            # 產生子代
            offspring = []
            while len(offspring) < pop_size:
                # tournament selection
                def _tournament():
                    contenders = random.sample(population, min(self.tournament_size, len(population)))
                    cont_objs = [objectives[population.index(c)] for c in contenders]
                    # 非嚴格：取目標和最小者
                    best_idx = min(range(len(contenders)), key=lambda i: (cont_objs[i][0], cont_objs[i][1]))
                    return contenders[best_idx]

                if len(population) >= self.tournament_size:
                    p1 = _tournament()
                    p2 = _tournament()
                else:
                    p1, p2 = random.sample(population, 2)

                if random.random() < self.crossover_rate:
                    c1, c2 = self._crossover(p1, p2, all_parts)
                else:
                    c1, c2 = p1.copy(), p2.copy()

                self._mutate(c1, parts_pool, required, optional)
                self._mutate(c2, parts_pool, required, optional)

                offspring.append(c1)
                if len(offspring) < pop_size:
                    offspring.append(c2)

            population.extend(offspring)

            # 評估合併族群，取前 pop_size
            objectives = []
            scores = []
            for indiv in population:
                obj1, obj2, sc, feasible = self._evaluate(indiv, parts_pool, required, weights, total_budget)
                if not feasible:
                    obj1 = float("inf")
                    obj2 = float("inf")
                objectives.append((obj1, obj2))
                scores.append(sc)
            fronts = self._fast_non_dominated_sort(objectives)
            new_population = []
            for front in fronts:
                if len(new_population) + len(front) > pop_size:
                    distance = self._crowding_distance(front, objectives)
                    front_sorted = sorted(front, key=lambda i: distance[i], reverse=True)
                    needed = pop_size - len(new_population)
                    new_population.extend([population[i] for i in front_sorted[:needed]])
                    break
                else:
                    new_population.extend([population[i] for i in front])
            population = new_population

        # 最終選擇：取第一前緣中分數最高者
        final_objectives = []
        final_scores = []
        feas_flags = []
        for indiv in population:
            obj1, obj2, sc, feasible = self._evaluate(indiv, parts_pool, required, weights, total_budget)
            if not feasible:
                obj1 = float("inf")
                obj2 = float("inf")
            final_objectives.append((obj1, obj2))
            final_scores.append(sc)
            feas_flags.append(feasible)
        fronts = self._fast_non_dominated_sort(final_objectives)
        # 優先挑選可行解；若全不可行則仍選第一前緣中分數最高者
        feasible_indices = [i for i, f in enumerate(feas_flags) if f]
        if feasible_indices:
            best_idx = max(feasible_indices, key=lambda i: final_scores[i])
        else:
            best_front = fronts[0]
            best_idx = max(best_front, key=lambda i: final_scores[i])
        best_indiv = population[best_idx]

        # 組合輸出
        build = {}
        current_spent = 0.0
        for part, idx in best_indiv.items():
            if idx is None:
                continue
            row = parts_pool[part].iloc[idx]
            build[part] = row
            current_spent += row.get("abs_price", 0)

        return build, current_spent

