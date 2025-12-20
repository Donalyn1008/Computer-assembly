from __future__ import annotations

from typing import Any, Dict, List, Optional
import pandas as pd
import streamlit as st

from optimization import PCBuilderAI  # type: ignore

# =========================
# CSV files (repo /data)
# =========================
FILES = {
    "CPU": "data/CPU_labeled.csv_ranking_result.csv",
    "MB": "data/MB_Labled.csv_ranking_result.csv",
    "VGA": "data/VGA_labeled.csv_ranking_result.csv",
    "RAM": "data/RAM_labeled.csv_ranking_result.csv",
    "SSD": "data/SSD_Labled.csv_ranking_result.csv",
    "HDD": "data/HDD_Labled.csv_ranking_result.csv",
    "PSU": "data/PSU_labeled.csv_ranking_result.csv",
    "HEAT": "data/HEAT_labeled.csv_ranking_result.csv",
    "FAN": "data/FAN_labeled.csv_ranking_result.csv",
    "CHASSIS": "data/CHASSIS_labeled.csv_ranking_result.csv",
    "WATER": "data/WATER_labeled.csv_ranking_result.csv",
}

# =========================
# UI Component Options
# =========================
COMPONENT_OPTIONS = [
    ("CPU", "CPU"),
    ("MB", "Motherboard"),
    ("RAM", "RAM"),
    ("VGA", "GPU"),
    ("SSD", "SSD"),
    ("HDD", "HDD"),
    ("PSU", "Power Supply"),
    ("COOLER", "Air Cooler"),
    ("FAN", "Fan"),
]
COMP_CODES = [x[0] for x in COMPONENT_OPTIONS]
COMP_LABEL = {x[0]: x[1] for x in COMPONENT_OPTIONS}

ORDER = ["CPU", "MB", "RAM", "VGA", "SSD", "HDD", "PSU", "COOLER", "FAN"]

# =========================
# Purpose options (⬅️ 補回來)
# =========================
PURPOSES = {
    "programming": "Programming",
    "graphic_design": "Graphic Design",
    "video_editing": "Video Editing",
    "word_processing": "Word Processing",
}

# =========================
# Helpers
# =========================
def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def normalize_col(s: Any) -> str:
    return str(s).replace("\u3000", " ").strip().lower()


def pick_detail(row: Any) -> str:
    if not hasattr(row, "to_dict"):
        return ""

    d = row.to_dict()
    norm = {normalize_col(k): k for k in d.keys()}

    for c in ["detail", "description", "spec", "規格", "描述", "細節", "說明"]:
        k = norm.get(normalize_col(c))
        if k:
            v = str(d.get(k, "")).strip()
            if v and v.lower() != "nan":
                return v
    return ""


@st.cache_resource
def get_ai() -> PCBuilderAI:
    return PCBuilderAI(FILES)


def optimizer_key_to_ui(k: str) -> str:
    if k in {"HEAT", "WATER"}:
        return "COOLER"
    return k


def build_items(build: Dict[str, Any], excludes: List[str]) -> List[Dict[str, Any]]:
    items = []
    ex = set(excludes)

    for k, row in build.items():
        ui_part = optimizer_key_to_ui(k)
        if ui_part in ex:
            continue

        detail = pick_detail(row)
        if not detail and "總分" in row:
            detail = f"score={safe_float(row['總分']):.2f}"

        items.append({
            "part": ui_part,
            "brand": str(row.get("BRAND", "")),
            "model": str(row.get("MODEL", "")),
            "detail": detail,
            "price": int(round(safe_float(row.get("abs_price", 0)))),
        })

    items.sort(key=lambda x: ORDER.index(x["part"]) if x["part"] in ORDER else 999)
    return items


# =========================
# Page
# =========================
st.set_page_config(page_title="PC Builder", layout="wide")
st.title("PC Builder")

# =========================
# Session state
# =========================
if "spec_rows" not in st.session_state:
    st.session_state.spec_rows = [{"part": "", "brand": ""}]
if "exclude_rows" not in st.session_state:
    st.session_state.exclude_rows = [{"part": ""}]

# =========================
# Basic Options
# =========================
st.subheader("Basic Options")

budget = st.number_input("Budget", min_value=0, value=50000, step=1000)
budget_val = int(budget)

purpose_key = st.selectbox(
    "Purpose",
    [""] + list(PURPOSES.keys()),
    format_func=lambda k: "Select purpose..." if k == "" else PURPOSES[k],
)
purpose_val = None if purpose_key == "" else purpose_key

# =========================
# Specify Components
# =========================
st.subheader("Specify Components (Optional)")

def add_spec():
    st.session_state.spec_rows.append({"part": "", "brand": ""})

def remove_spec(i: int):
    if len(st.session_state.spec_rows) > 1:
        st.session_state.spec_rows.pop(i)

st.button("➕ Add one", on_click=add_spec)

for i, row in enumerate(st.session_state.spec_rows):
    c1, c2, c3 = st.columns([3, 3, 1])
    with c1:
        part = st.selectbox(
            f"Component #{i+1}",
            [""] + COMP_CODES,
            key=f"spec_part_{i}",
            format_func=lambda x: "Select..." if x == "" else COMP_LABEL[x],
        )
        st.session_state.spec_rows[i]["part"] = part
    with c2:
        brand = st.text_input(f"Brand #{i+1}", value=row["brand"], key=f"spec_brand_{i}")
        st.session_state.spec_rows[i]["brand"] = brand.strip()
    with c3:
        st.button("Remove", key=f"rm_spec_{i}", on_click=remove_spec, args=(i,))

# =========================
# Exclude Components
# =========================
st.subheader("Exclude Components")

def add_ex():
    st.session_state.exclude_rows.append({"part": ""})

def remove_ex(i: int):
    if len(st.session_state.exclude_rows) > 1:
        st.session_state.exclude_rows.pop(i)

st.button("➕ Add exclude", on_click=add_ex)

for i, row in enumerate(st.session_state.exclude_rows):
    c1, c2 = st.columns([4, 1])
    with c1:
        ex = st.selectbox(
            f"Exclude #{i+1}",
            [""] + COMP_CODES,
            key=f"ex_part_{i}",
            format_func=lambda x: "Select..." if x == "" else COMP_LABEL[x],
        )
        st.session_state.exclude_rows[i]["part"] = ex
    with c2:
        st.button("Remove", key=f"rm_ex_{i}", on_click=remove_ex, args=(i,))

# =========================
# Run Optimizer 
# =========================
specified = [{"part": r["part"], "brand": r["brand"]} for r in st.session_state.spec_rows if r["part"]]
excludes = [r["part"] for r in st.session_state.exclude_rows if r["part"]]

if st.button("Generate Result", type="primary"):
 
    spec_brand_map = {}
    for s in specified:
        spec_brand_map[s["part"]] = s["brand"]

    prefs = {
        "cpu_brand": spec_brand_map.get("CPU", ""), 
        "specified_brands": spec_brand_map,         
        "cooling": "heat",
        "purpose": purpose_val,
    }

    ai = get_ai()
    try:
        build, total_price = ai.optimize_build(budget_val, prefs)
        items = build_items(build, excludes)

        if not items:
            st.warning("No results found (check your budget or filters).")
        else:
            df = pd.DataFrame(items)
   
            df["part"] = df["part"].map(lambda x: COMP_LABEL.get(x, x))
            
            st.subheader("Output")
            st.dataframe(df[["part", "brand", "model", "detail", "price"]], use_container_width=True)
            st.success(f"### Total Price: **{int(total_price):,} NTD**")
            
    except Exception as e:
        st.error(f"Error during optimization: {e}")
