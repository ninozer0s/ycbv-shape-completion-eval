import argparse
import base64
import json
import os
from pathlib import Path
from openai import OpenAI


def read_api_key():
    env_key = os.environ.get("AQUEDUCT_API_KEY")
    if env_key:
        return env_key.strip()

    for p in [Path(".api_key"), Path("~/sasha_gpt/.api_key").expanduser()]:
        if p.exists():
            return p.read_text().strip()

    raise FileNotFoundError("No API key found.")


def image_to_data_url(path):
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    elif suffix == ".png":
        mime = "image/png"
    else:
        raise ValueError(f"Unsupported image type: {suffix}")

    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{data}"


def extract_json(text):
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output:\n" + text)

    return json.loads(text[start:end + 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--object", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    client = OpenAI(
        base_url="https://aqueduct.ai.datalab.tuwien.ac.at/v1",
        api_key=read_api_key(),
    )

    model = "qwen-3.6-35b"

    prompt = f"""
You are a robot manipulation reasoning module.

The robot has already generated many geometrically valid grasp candidates for the whole object.
Your task is NOT to generate 6D grasp poses.
Your task is to decide which object regions are suitable or unsuitable for the given manipulation task.

Object: {args.object}
Task: {args.task}

Return ONLY valid JSON with this schema:

{{
  "object": "{args.object}",
  "task": "{args.task}",
  "is_grasp_task": true,
  "preferred_grasp_region": "...",
  "avoid_grasp_region": "...",
  "task_reason": "...",
  "geometric_filter_hint": "...",
  "simple_rule": {{
    "preferred_z_range": [0.0, 1.0],
    "avoid_top": false,
    "avoid_bottom": false,
    "prefer_side": false,
    "prefer_edge_or_corner": false,
    "prefer_handle": false,
    "prefer_middle_body": false
  }},
  "confidence": 0.0
}}

Rules:
- If the task requires pushing, pressing, opening a button, or touching without holding, set "is_grasp_task" to false.
- Do not generate exact 6D poses.
- Use simple object-region words: top, side, handle, rim, cap, bottom, edge, corner, middle body.
- The simple_rule field must contain booleans and a normalized z-range from object bottom 0.0 to object top 1.0.
- The geometric_filter_hint should be usable by a point-cloud based filter.
"""

    res = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_to_data_url(args.image)
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ],
    )

    raw = res.choices[0].message.content
    print("RAW MODEL OUTPUT:")
    print(raw)

    parsed = extract_json(raw)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(parsed, indent=2))

    print("")
    print("Wrote clean JSON:", out)
    print(json.dumps(parsed, indent=2))


if __name__ == "__main__":
    main()
