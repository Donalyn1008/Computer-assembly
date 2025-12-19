from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import streamlit as st

# =========================
# 1) Import your optimizer
# =========================
# optimization.py must be in the SAME repo root as app.py
from optimization import PCBuilderAI  # type: ignore


# =========================
# 2) Data files in GitHub repo
#    (repo root) /data/*.csv
# =========================
FILES = {
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

# =========================
# 3) UI constants
# =========================
PARTS = [
    ("AIO", "AIO Liquid Cooler"),
    ("FAN", "Fan"),
    ("COOLER", "Air Cooler"),
    ("PSU", "Power Supply"),
    ("CASE", "Case"),
    ("CPU", "CPU"),
    ("HDD", "HDD"),
    ("MB", "Motherboard"),
    ("RAM", "RAM"),
    ("SSD", "SSD"),
    ("VGA", "GPU"),
]
PART_CODES = [p[0] for p in PARTS]
PART_LABEL = {p[0]: p[1] for p in PARTS}

PURPOSES = {
    "programming": "Programming",
    "graph_design": "Graphic Design",
    "video_editing": "Video Editing",
    "word_processing": "Word Processing",
}
RAM_OPTIONS = [8, 16, 32, 48, 64, 96, 128, 192]

ORDER = ["CPU", "MB", "RAM", "VGA", "SSD", "HDD", "PSU", "CASE", "COOLER", "AIO", "FAN"]


def part_fmt(code: str) -> str:
    return "Select..." if code == "" else f"{PART_LABEL.get(code, code)} ({code})"


def map_optimizer_part_to_ui(part_key: str, cooling: str) -> str:
    """
    Optimizer uses: CHASSIS / HEAT / WATER
    UI wants: CASE / COOLER / AIO
    """
    if part_key == "CHASSIS":
        return "CASE"
    if part_key == "HEAT":
        return "COOLER"
    if part_key == "WATER":
        return "AIO"
    return part_key


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


@st.cache_resource
def get_ai() -> PCBuilderAI:
    # cache the optimizer instance so it doesn't reload CSV every rerun
    return PCBuilderAI(FILES)


def build_items_from_optimizer(
    build: Dict[str, Any],
    cooling: str,
    excludes: List[str],
) -> List[Dict[str, Any]]:
    """
    build: dict returned by ai.optimize_build -> {part_key: pd.Series}
    """
    ex = set(excludes or [])
    items: List[Dict[str, Any]] = []

    for part_key, row in build.items():
        ui_part = map_optimizer_part_to_ui(part_key, cooling)

        if ui_part in ex:
            continue

        brand = str(row.get("BRAND", "")) if hasattr(row, "get") else ""
        model = str(row.get("MODEL", "")) if hasattr(row, "get") else ""
        price = safe_float(row.get("abs_price", 0)) if hasattr(row, "get") else 0.0

        # Use CSV detail column (prefer Detail/detail), fallback to score if missing
        detail_val = ""
        if hasattr(row, "get"):
            if "Detail" in row:
                detail_val = str(row.get("Detail", "") or "")
            elif "detail" in row:
                detail_val = str(row.get("detail", "") or "")
            elif "DETAIL" in row:
                detail_val = str(row.get("DETAIL", "") or "")

        detail_val = detail_val.strip()

        # Fallback: if your CSV doesn't have Detail/detail, show score
        if detail_val == "" and hasattr(row, "get") and "總分" in row:
            detail_val = f"score={safe_float(row.get('總分')):.2f}"

        detail = detail_val

        items.append(
            {
                "part": ui_part,
                "brand": brand,
                "model": model,
                "detail": detail,
                "price": int(round(price)),
            }
        )

    items.sort(key=lambda x: ORDER.index(x["part"]) if x["part"] in ORDER else 999)
    return items


# =========================
# 4) Page
# =========================
st.set_page_config(page_title="PC Builder", layout="wide")
st.title("PC Builder")

with st.sidebar:
    st.caption(
        "This UI calls your optimizer (optimization.py) and reads CSV files from /data in this GitHub repo."
    )
    st.caption(
        "If deployment fails, check: file names in /data and columns "
        "(BRAND/MODEL/總分/price_分數 or Price_分數)."
    )

# Session state init
if "spec_rows" not in st.session_state:
    st.session_state.spec_rows = [{"part": "", "brand": ""}]
if "exclude_rows" not in st.session_state:
    st.session_state.exclude_rows = [{"part": ""}]


# =========================
# 5) Basic Options
# =========================
st.subheader("Basic Options")
c1, c2 = st.columns(2)

with c1:
    budget = st.number_input("Budget limit", min_value=0, value=50000, step=1000)
    budget_val = int(budget)

with c2:
    purpose_key = st.selectbox(
        "Purpose",
        [""] + list(PURPOSES.keys()),
        format_func=lambda k: "Select..." if k == "" else PURPOSES[k],
    )
    purpose_val = None if purpose_key == "" else purpose_key


# =========================
# 6) Advanced Options
# =========================
st.subheader("Advanced Options")
c3, c4, c5, c6 = st.columns(4)

with c3:
    color = st.selectbox(
        "Color",
        ["black", "white"],
        format_func=lambda x: "Black" if x == "black" else "White",
    )
    color_val = color

with c4:
    rgb = st.selectbox(
        "RGB lighting",
        ["", "yes", "no"],
        format_func=lambda x: {"": "Not specified", "yes": "Yes", "no": "No"}[x],
    )
    rgb_val = None if rgb == "" else rgb

with c5:
    ram = st.selectbox(
        "RAM (GB)",
        [""] + RAM_OPTIONS,
        format_func=lambda x: "Not specified" if x == "" else str(x),
    )
    ram_val = None if ram == "" else int(ram)

with c6:
    cooling = st.selectbox(
        "Cooling type",
        ["heat", "water"],
        format_func=lambda x: "Air Cooler (HEAT)" if x == "heat" else "Liquid Cooler (WATER)",
    )
    cooling_val = cooling

st.divider()


# =========================
# 7) Specify component -> brand (multiple)
# =========================
st.markdown("### Specify Components → Specify Brand (Multiple Allowed)")


def add_spec() -> None:
    st.session_state.spec_rows.append({"part": "", "brand": ""})


def remove_spec(idx: int) -> None:
    if len(st.session_state.spec_rows) > 1:
        st.session_state.spec_rows.pop(idx)


st.button("➕ Add one more", on_click=add_spec, key="btn_add_spec")

for i, row in enumerate(st.session_state.spec_rows):
    a, b, c = st.columns([2, 3, 1])

    with a:
        part = st.selectbox(
            f"Component #{i+1}",
            [""] + PART_CODES,
            key=f"spec_part_{i}",
            index=([""] + PART_CODES).index(row["part"]) if row["part"] in ([""] + PART_CODES) else 0,
            format_func=part_fmt,
        )
        st.session_state.spec_rows[i]["part"] = part

    with b:
        brand = st.text_input(
            f"Brand #{i+1} (optional)",
            value=row.get("brand", ""),
            key=f"spec_brand_{i}",
            placeholder="e.g., Intel / AMD / ASUS / MSI ...",
        )
        st.session_state.spec_rows[i]["brand"] = brand.strip()

    with c:
        st.button("Remove", key=f"btn_rm_spec_{i}", on_click=remove_spec, args=(i,))

st.divider()


# =========================
# 8) Exclude components (multiple)
# =========================
st.markdown("### Exclude Components (Multiple Allowed)")


def add_exclude() -> None:
    st.session_state.exclude_rows.append({"part": ""})


def remove_exclude(idx: int) -> None:
    if len(st.session_state.exclude_rows) > 1:
        st.session_state.exclude_rows.pop(idx)


st.button("➕ Add one more", on_click=add_exclude, key="btn_add_exclude")

for i, row in enumerate(st.session_state.exclude_rows):
    a, b = st.columns([4, 1])

    with a:
        ex = st.selectbox(
            f"Exclude component #{i+1}",
            [""] + PART_CODES,
            key=f"ex_part_{i}",
            index=([""] + PART_CODES).index(row["part"]) if row["part"] in ([""] + PART_CODES) else 0,
            format_func=part_fmt,
        )
        st.session_state.exclude_rows[i]["part"] = ex

    with b:
        st.button("Remove", key=f"btn_rm_ex_{i}", on_click=remove_exclude, args=(i,))

# Build payload from UI
specified = [{"part": r["part"], "brand": r["brand"]} for r in st.session_state.spec_rows if r["part"]]
excludes = [r["part"] for r in st.session_state.exclude_rows if r["part"]]

payload: Dict[str, Any] = {
    "price": budget_val,
    "purpose": purpose_val,
    "color": color_val,
    "rgb": rgb_val,
    "ram": ram_val,
    "cooling": cooling_val,
    "specified": specified,
    "excludes": excludes,
}

with st.expander("Payload (debug)"):
    st.json(payload)


# =========================
# 9) Run optimizer and output
# =========================
if st.button("Generate Results", type="primary", key="btn_generate"):
    cpu_brand = None
    brand_overrides: Dict[str, str] = {}

    for s in specified:
        part = s.get("part")
        brand = (s.get("brand") or "").strip()
        if not part or not brand:
            continue

        if part == "CPU":
            cpu_brand = brand
        else:
            # Optional extra info (won't break if optimizer ignores it)
            if part == "CASE":
                brand_overrides["CHASSIS"] = brand
            elif part in {"AIO", "COOLER"}:
                brand_overrides["WATER" if cooling_val == "water" else "HEAT"] = brand
            else:
                brand_overrides[part] = brand

    prefs = {
        "color": color_val,  # 'white' or 'black'
        "cpu_brand": cpu_brand if isinstance(cpu_brand, str) else "",
        "cooling": cooling_val,  # 'water' or 'heat'
        "brand_overrides": brand_overrides,  # optional (if your optimizer supports later)
        "purpose": purpose_val,  # optional, kept for future
        "rgb": rgb_val,  # optional, kept for future
        "ram": ram_val,  # optional, kept for future
    }

    try:
        ai = get_ai()
        final_build, total_cost = ai.optimize_build(budget_val, prefs)

        items = build_items_from_optimizer(final_build, cooling_val, excludes)

        if not items:
            st.warning("No results returned (possibly excluded everything).")
        else:
            df = pd.DataFrame(items)
            df["part"] = df["part"].map(lambda x: PART_LABEL.get(x, x))
            df = df[["part", "brand", "model", "detail", "price"]]

            total_price = int(df["price"].fillna(0).sum())

            st.subheader("Output")
            st.dataframe(df, use_container_width=True)

            st.markdown(f"### Total Price: **{total_price:,} NTD**")

            st.download_button(
                "Download CSV",
                data=df.to_csv(index=False).encode("utf-8-sig"),
                file_name="pc_recommendation.csv",
                mime="text/csv",
            )

            st.caption(f"Optimizer reported total (debug): {int(round(total_cost)):,} NTD")

    except Exception as e:
        st.error("Optimizer failed to run. Please check your CSV paths and column names in /data.")
        st.exception(e)
