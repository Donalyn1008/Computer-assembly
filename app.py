from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import streamlit as st

from optimization import NSGAPCBuilder  # type: ignore

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
    ("COOLER", "Cooler"),
    ("FAN", "Fan"),
]
COMP_CODES = [x[0] for x in COMPONENT_OPTIONS]
COMP_LABEL = {x[0]: x[1] for x in COMPONENT_OPTIONS}

ORDER = ["CPU", "MB", "RAM", "VGA", "SSD", "HDD", "PSU", "COOLER", "FAN"]

# =========================
# Purpose options
# =========================
PURPOSES = {
    "programming": "Programming",
    "graphic_design": "Graphic Design",
    "video_editing": "Video Editing",
    "word_processing": "Word Processing",
}

# =========================
# Cooling options
# =========================
COOLING_OPTIONS = {
    "heat": "Air Cooling",
    "water": "Water Cooling",
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
    # Replace Chinese space with regular space and strip
    return str(s).replace("\u3000", " ").strip().lower()


def pick_detail(row: Any) -> str:
    """Picks the most relevant detail/spec column from the row."""
    if not hasattr(row, "to_dict"):
        return ""

    d = row.to_dict()
    norm = {normalize_col(k): k for k in d.keys()}

    # Search for common detail column names (including Chinese ones to match CSV)
    for c in ["detail", "description", "spec", "規格", "描述", "細節", "說明"]:
        k = norm.get(normalize_col(c))
        if k:
            v = str(d.get(k, "")).strip()
            if v and v.lower() != "nan":
                return v

    # Fallback to score if available (using Chinese column name from CSV)
    if "總分" in row:
        return f"score={safe_float(row['總分']):.2f}"

    return ""


@st.cache_resource
def get_ai() -> NSGAPCBuilder:
    return NSGAPCBuilder(FILES)


def optimizer_key_to_ui(k: str) -> str:
    """Converts optimization key (HEAT/WATER) to UI key (COOLER)"""
    if k in {"HEAT", "WATER"}:
        return "COOLER"
    return k


def ui_key_to_optimizer(k: str, cooling_type: str) -> str:
    """Converts UI key (COOLER) to optimization key (HEAT/WATER)"""
    if k == "COOLER":
        return "WATER" if cooling_type == "water" else "HEAT"
    return k


def build_items(build: Dict[str, Any], excludes: List[str]) -> Tuple[List[Dict[str, Any]], int]:
    items: List[Dict[str, Any]] = []
    ex = set(excludes)
    total_price = 0

    for k, row in build.items():
        ui_part = optimizer_key_to_ui(k)
        if ui_part in ex:
            continue

        price = int(round(safe_float(row.get("abs_price", 0))))
        total_price += price

        items.append({
            "part": ui_part,
            "brand": str(row.get("BRAND", "")),
            "model": str(row.get("MODEL", "")),
            "detail": pick_detail(row),
            "price": price,
        })

    items.sort(key=lambda x: ORDER.index(x["part"]) if x["part"] in ORDER else 999)
    return items, total_price


# =========================
# Page
# =========================
st.set_page_config(page_title="PC Builder", layout="wide")
st.title("🖥️ PC Builder - PC Assembly Recommendation System")

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
st.subheader("📋 Basic Options")

col1, col2 = st.columns(2)

with col1:
    budget = st.number_input("💰 Budget", min_value=0, value=50000, step=1000)
    budget_val = int(budget)

with col2:
    purpose_key = st.selectbox(
        "🎯 Purpose",
        [""] + list(PURPOSES.keys()),
        format_func=lambda k: "Select purpose..." if k == "" else PURPOSES[k],
    )
    purpose_val = None if purpose_key == "" else purpose_key

# Cooling Type Selection
cooling_key = st.selectbox(
    "❄️ Cooling Type",
    list(COOLING_OPTIONS.keys()),
    format_func=lambda k: COOLING_OPTIONS[k],
)

# =========================
# Advanced Options (⬅️ 新增)
# =========================
st.subheader("⚙️ Advanced Options")

adv1, adv2 = st.columns(2)

with adv1:
    color_choice = st.selectbox(
        "🎨 Color (Black / White)",
        ["Any", "Black", "White"],
        index=0,
        help="This preference will be passed to the optimizer as prefs['color'] (None/black/white).",
    )

with adv2:
    rgb_choice = st.selectbox(
        "💡 RGB LIGHT",
        ["Any", "Yes (Need RGB)", "No (No RGB)"],
        index=0,
        help="This preference will be passed to the optimizer as prefs['rgb'] (None/True/False).",
    )

# convert to normalized values for backend
color_val: Optional[str] = None
if color_choice == "Black":
    color_val = "black"
elif color_choice == "White":
    color_val = "white"

rgb_val: Optional[bool] = None
if rgb_choice.startswith("Yes"):
    rgb_val = True
elif rgb_choice.startswith("No"):
    rgb_val = False

# =========================
# Specify Components
# =========================
st.subheader("🔧 Specify Components (Optional - Specify Brands)")
st.caption("Specify a brand for any component here.")

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
        brand = st.text_input(
            f"Brand #{i+1}",
            value=row["brand"],
            key=f"spec_brand_{i}",
            placeholder="e.g., Intel, AMD, ASUS, MSI...",
        )
        st.session_state.spec_rows[i]["brand"] = brand.strip()
    with c3:
        st.button("❌", key=f"rm_spec_{i}", on_click=remove_spec, args=(i,))

# =========================
# Exclude Components
# =========================
st.subheader("🚫 Exclude Components")
st.caption("Exclude any components you don't need.")

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
        st.button("❌", key=f"rm_ex_{i}", on_click=remove_ex, args=(i,))

# =========================
# Run Optimizer
# =========================
specified = [{"part": r["part"], "brand": r["brand"]} for r in st.session_state.spec_rows if r["part"]]
excludes = [r["part"] for r in st.session_state.exclude_rows if r["part"]]

st.markdown("---")

if st.button("🚀 Generate Result", type="primary", use_container_width=True):

    # Create brand map, converting UI COOLER to the correct HEAT/WATER for the optimizer
    spec_brand_map: Dict[str, str] = {}
    for s in specified:
        if s["brand"]:
            optimizer_part = ui_key_to_optimizer(s["part"], cooling_key)
            spec_brand_map[optimizer_part] = s["brand"]

    prefs = {
        "specified_brands": spec_brand_map,
        "cooling": cooling_key,
        "purpose": purpose_val,
        "color": color_val,  # None / "black" / "white"
        "rgb": rgb_val,      # None / True / False
    }

    ai = get_ai()
    try:
        with st.spinner("Analyzing optimal configuration..."):
            build, total_price = ai.optimize_build(budget_val, prefs)

        items, final_price = build_items(build, excludes)

        if not items:
            st.warning("⚠️ No results found. Try increasing the budget or relaxing brand restrictions.")
        else:
            st.success("✅ Configuration generated successfully!")

            st.subheader("📊 Recommended Build List")

            df = pd.DataFrame(items)
            df["part"] = df["part"].map(lambda x: COMP_LABEL.get(x, x))
            df.columns = ["Part Type", "Brand", "Model", "Detail", "Price (NTD)"]

            st.dataframe(df, use_container_width=True, hide_index=True)

            st.markdown(f"### 💵 Total Price: **NT$ {int(final_price):,}**")

            budget_usage = (final_price / budget_val) * 100 if budget_val > 0 else 0.0
            st.progress(min(budget_usage / 100, 1.0))
            st.caption(f"Budget Usage: {budget_usage:.1f}%")

    except Exception as e:
        st.error(f"❌ Execution Error: {e}")
        st.exception(e)


