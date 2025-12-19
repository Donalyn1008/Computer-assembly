from __future__ import annotations
import random
from typing import Any

import pandas as pd
import streamlit as st

# ===== Switch =====
# True  -> mock demo (no CSV required)
# False -> use optimizer (requires CSV files in repo)
MOCK_MODE = False

# Component categories (code -> display name)
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

DEFAULT_BUILD = ["CPU", "MB", "RAM", "SSD", "VGA", "PSU", "CASE", "COOLER"]

MOCK_BRANDS = {
    "CPU": ["Intel", "AMD"],
    "MB": ["ASUS", "MSI", "GIGABYTE", "ASRock"],
    "RAM": ["Kingston", "Corsair", "G.SKILL", "Crucial"],
    "VGA": ["ASUS", "MSI", "GIGABYTE", "ZOTAC", "SAPPHIRE"],
    "SSD": ["Samsung", "WD", "Kingston", "Crucial"],
    "HDD": ["Seagate", "WD", "Toshiba"],
    "PSU": ["Seasonic", "Corsair", "FSP", "Cooler Master"],
    "CASE": ["LIAN LI", "NZXT", "Fractal", "Cooler Master"],
    "COOLER": ["Noctua", "DeepCool", "Thermalright"],
    "FAN": ["Noctua", "ARCTIC", "LIAN LI"],
    "AIO": ["Corsair", "NZXT", "ASUS", "LIAN LI"],
}

PRICE_RANGE = {
    "CPU": (4500, 14000),
    "MB": (2800, 9000),
    "RAM": (1200, 8000),
    "SSD": (1200, 6500),
    "HDD": (1200, 4500),
    "VGA": (7000, 40000),
    "PSU": (1800, 6500),
    "CASE": (1500, 6500),
    "COOLER": (800, 3500),
    "FAN": (200, 1500),
    "AIO": (2000, 7000),
}

ORDER = ["CPU", "MB", "RAM", "VGA", "SSD", "HDD", "PSU", "CASE", "COOLER", "AIO", "FAN"]


def part_fmt(code: str) -> str:
    return "Select..." if code == "" else f"{PART_LABEL.get(code, code)} ({code})"


def purpose_multiplier(purpose: str | None) -> float:
    if purpose == "video_editing":
        return 1.15
    if purpose == "graph_design":
        return 1.10
    if purpose == "programming":
        return 1.05
    if purpose == "word_processing":
        return 0.95
    return 1.00


def make_model(part: str, brand: str, ram_gb: int | None) -> str:
    suffix = random.randint(100, 999)
    if part == "CPU":
        return f"{brand} {random.choice(['i5','i7','R5','R7'])}-{suffix}"
    if part == "MB":
        return f"{brand} {random.choice(['B760','Z790','B650','X670'])}-{suffix}"
    if part == "RAM":
        cap = f"{ram_gb}GB" if ram_gb else random.choice(["16GB", "32GB", "64GB"])
        return f"{brand} DDR{random.choice([4,5])} {cap}"
    if part == "VGA":
        return f"{brand} RTX{random.choice(['4060','4070','4080'])}-{suffix}"
    if part == "SSD":
        return f"{brand} {random.choice(['NVMe','PCIe4.0','PCIe5.0'])}-{suffix}"
    if part == "HDD":
        return f"{brand} {random.choice(['2TB','4TB','8TB'])}-{suffix}"
    if part == "PSU":
        return f"{brand} {random.choice(['650W','750W','850W'])} Gold"
    if part == "CASE":
        return f"{brand} {random.choice(['ATX','mATX','Mid Tower'])}-{suffix}"
    if part == "COOLER":
        return f"{brand} {random.choice(['Tower','Top-Down'])}-{suffix}"
    if part == "AIO":
        return f"{brand} {random.choice(['240mm','280mm','360mm'])} AIO"
    if part == "FAN":
        return f"{brand} {random.choice(['120mm','140mm'])} Fan"
    return f"{brand} {part}-{suffix}"


def make_detail(purpose: str | None, color: str | None, rgb: str | None, ram_gb: int | None, part: str) -> str:
    tags = []
    if purpose:
        tags.append(f"purpose={purpose}")
    if color:
        tags.append(f"color={color}")
    if rgb:
        tags.append(f"rgb={rgb}")
    if part == "RAM" and ram_gb:
        tags.append(f"capacity={ram_gb}GB")
    return " | ".join(tags) if tags else "mock detail"


def mock_recommend(payload: dict[str, Any]) -> list[dict[str, Any]]:
    price_cap: int | None = payload.get("price")
    purpose: str | None = payload.get("purpose")
    color: str | None = payload.get("color")
    rgb: str | None = payload.get("rgb")
    ram_gb: int | None = payload.get("ram")

    specified: list[dict[str, Any]] = payload.get("specified") or []
    excludes: set[str] = set(payload.get("excludes") or [])

    target_parts = list(DEFAULT_BUILD)
    for s in specified:
        p = s.get("part")
        if p and p not in target_parts:
            target_parts.append(p)

    target_parts = [p for p in target_parts if p not in excludes]

    items = []
    mult = purpose_multiplier(purpose)

    for part in target_parts:
        brands = MOCK_BRANDS.get(part, ["Generic"])
        specified_brand = ""
        for s in specified:
            if s.get("part") == part and s.get("brand"):
                specified_brand = s["brand"]
                break
        brand = specified_brand or random.choice(brands)

        lo, hi = PRICE_RANGE.get(part, (1000, 5000))
        price = round(random.randint(lo, hi) * mult)

        if rgb == "yes" and part in {"CASE", "FAN", "AIO", "COOLER", "RAM"}:
            price += 300
        if color == "white" and part in {"CASE", "AIO", "COOLER", "FAN"}:
            price += 200

        items.append(
            {
                "part": part,
                "brand": brand,
                "model": make_model(part, brand, ram_gb),
                "detail": make_detail(purpose, color, rgb, ram_gb, part),
                "price": price,
            }
        )

    items.sort(key=lambda x: ORDER.index(x["part"]) if x["part"] in ORDER else 999)
    return items


# ===== Optimizer wiring (IMPORTANT) =====
# Put your CSV files inside repo, e.g. data/*.csv
FILE_MAP = {
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


def run_optimizer(payload: dict[str, Any]) -> list[dict[str, Any]]:
    from optimization import recommend, OptimizerConfig
    cfg = OptimizerConfig(file_map=FILE_MAP)
    return recommend(payload, cfg)


# ============ UI ============
st.set_page_config(page_title="PC Builder", layout="wide")
st.title("PC Builder")

with st.sidebar:
    st.write("Mode:", "Mock (no CSV needed)" if MOCK_MODE else "Optimizer (CSV required)")
    if not MOCK_MODE:
        st.caption("Make sure all CSV files exist in your GitHub repo under the 'data/' folder.")


# Session state init
if "spec_rows" not in st.session_state:
    st.session_state.spec_rows = [{"part": "", "brand": ""}]
if "exclude_rows" not in st.session_state:
    st.session_state.exclude_rows = [{"part": ""}]


# Basic options
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


# Advanced options
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
    ram = st.selectbox("RAM (GB)", [""] + RAM_OPTIONS, format_func=lambda x: "Not specified" if x == "" else str(x))
    ram_val = None if ram == "" else int(ram)

with c6:
    cooling = st.selectbox(
        "Cooling type",
        ["heat", "water"],
        format_func=lambda x: "Air Cooler (HEAT)" if x == "heat" else "Liquid Cooler (WATER)",
    )
    cooling_val = cooling

st.divider()

# Specify component -> brand (multiple)
st.markdown("### Specify Components → Specify Brand (Multiple Allowed)")


def add_spec():
    st.session_state.spec_rows.append({"part": "", "brand": ""})


def remove_spec(idx: int):
    if len(st.session_state.spec_rows) > 1:
        st.session_state.spec_rows.pop(idx)


st.button("➕ Add one more", on_click=add_spec)

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
        if part:
            brands = MOCK_BRANDS.get(part, ["Generic"])  # UI list (real filter is handled by optimizer)
            brand = st.selectbox(
                f"Brand #{i+1} (optional)",
                [""] + brands,
                key=f"spec_brand_{i}",
                format_func=lambda x: "Not specified" if x == "" else x,
            )
        else:
            brand = st.selectbox(
                f"Brand #{i+1}",
                [""],
                key=f"spec_brand_{i}",
                format_func=lambda _: "Select a component first",
            )
        st.session_state.spec_rows[i]["brand"] = brand

    with c:
        st.button("Remove", key=f"rm_spec_{i}", on_click=remove_spec, args=(i,))

st.divider()

# Exclude components (multiple)
st.markdown("### Exclude Components (Multiple Allowed)")


def add_exclude():
    st.session_state.exclude_rows.append({"part": ""})


def remove_exclude(idx: int):
    if len(st.session_state.exclude_rows) > 1:
        st.session_state.exclude_rows.pop(idx)


st.button("➕ Add one more", on_click=add_exclude)

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
        st.button("Remove", key=f"rm_ex_{i}", on_click=remove_exclude, args=(i,))

# Build payload
specified = [{"part": r["part"], "brand": r["brand"]} for r in st.session_state.spec_rows if r["part"]]
excludes = [r["part"] for r in st.session_state.exclude_rows if r["part"]]

payload: dict[str, Any] = {
    "price": budget_val,
    "purpose": purpose_val,
    "color": color_val,
    "rgb": rgb_val,
    "ram": ram_val,
    "cooling": cooling_val,          # ✅ for optimizer
    "specified": specified,          # ✅ brand overrides
    "excludes": excludes,
}

with st.expander("Payload (debug)"):
    st.json(payload)

# Generate output
if st.button("Generate Results", type="primary"):
    if MOCK_MODE:
        items = mock_recommend(payload)
    else:
        items = run_optimizer(payload)

    if not items:
        st.warning("No results. (All components may have been excluded, or the optimizer returned empty output.)")
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
