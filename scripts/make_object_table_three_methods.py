from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("~/bop_datasets/ycbv_3dsgrasp_eval_filtered").expanduser()
IN_CSV = ROOT / "analysis_three_methods" / "metrics_three_methods.csv"

OUT_DIR = ROOT / "analysis_three_methods"
OUT_CSV = OUT_DIR / "table_cd_l2_by_object.csv"
OUT_MD = OUT_DIR / "table_cd_l2_by_object.md"
OUT_TEX = OUT_DIR / "table_cd_l2_by_object.tex"

OBJ_NAMES = {
    1: "Master Chef Can",
    2: "Cracker Box",
    3: "Sugar Box",
    4: "Tomato Soup Can",
    5: "Mustard Bottle",
    6: "Tuna Fish Can",
    7: "Pudding Box",
    8: "Gelatin Box",
    9: "Potted Meat Can",
    10: "Banana",
    11: "Pitcher Base",
    12: "Bleach Cleanser",
    13: "Bowl",
    14: "Mug",
    15: "Power Drill",
    16: "Wood Block",
    17: "Scissors",
    18: "Large Marker",
    19: "Large Clamp",
    20: "Extra Large Clamp",
    21: "Foam Brick",
}

METHODS = [
    ("Partial", "partial_cd_l2_x1000"),
    ("3DSGrasp", "3dsgrasp_cd_l2_x1000"),
    ("AdaPoinTr", "adapointr_cd_l2_x1000"),
    ("shape_comp", "shape_comp_cd_l2_x1000"),
]

if not IN_CSV.exists():
    raise FileNotFoundError(f"Missing {IN_CSV}. Run evaluate_and_visualize_three_methods.py first.")

df = pd.read_csv(IN_CSV)

present_obj_ids = sorted(df["obj_id"].unique())
present_obj_names = [OBJ_NAMES.get(i, f"obj_{i}") for i in present_obj_ids]

rows = []

for method_name, col in METHODS:
    row = {"Method": method_name}
    row["Avg"] = df[col].mean()

    for obj_id in present_obj_ids:
        obj_name = OBJ_NAMES.get(obj_id, f"obj_{obj_id}")
        sub = df[df["obj_id"] == obj_id]
        row[obj_name] = sub[col].mean()

    rows.append(row)

table = pd.DataFrame(rows)

# Round like paper tables
num_cols = [c for c in table.columns if c != "Method"]
table[num_cols] = table[num_cols].round(2)

# Save CSV
table.to_csv(OUT_CSV, index=False)

# Save Markdown
md = table.to_markdown(index=False)
OUT_MD.write_text(md)

# Save LaTeX
latex = table.to_latex(
    index=False,
    float_format="%.2f",
    escape=True,
    caption="Mean CD-L2 $\\times 1000$ per object category on filtered YCB-V partial point clouds. Lower is better.",
    label="tab:ycbv_object_cd",
)
OUT_TEX.write_text(latex)

print("")
print("Object-wise CD-L2 x1000 table")
print("=" * 80)
print(md)
print("")
print("Wrote:")
print(" ", OUT_CSV)
print(" ", OUT_MD)
print(" ", OUT_TEX)
