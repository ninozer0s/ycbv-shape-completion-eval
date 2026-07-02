from pathlib import Path
from openai import OpenAI
import base64

api_key = Path(".api_key").read_text().strip()

client = OpenAI(
    base_url="https://aqueduct.ai.datalab.tuwien.ac.at/v1",
    api_key=api_key,
    timeout=60,
)

img_path = Path.home() / "bop_datasets/ycbv/test/000048/rgb/000001.png"
data = base64.b64encode(img_path.read_bytes()).decode("utf-8")
image_url = f"data:image/png;base64,{data}"

models = [m.id for m in client.models.list().data]

print("Testing image input support for:")
for m in models:
    print("-", m)

print("\n" + "=" * 80)

vision_models = []

for model in models:
    print(f"\nTesting model: {model}")
    try:
        res = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            }
                        },
                        {
                            "type": "text",
                            "text": "Can you see the image? If yes, briefly describe the visible objects."
                        }
                    ],
                }
            ],
        )

        answer = res.choices[0].message.content
        print("SUCCESS")
        print(answer[:500])
        vision_models.append(model)

    except Exception as e:
        msg = str(e)
        print("FAILED")
        print(msg[:500])

print("\n" + "=" * 80)
print("Vision-capable models found:")
if vision_models:
    for m in vision_models:
        print("-", m)
else:
    print("None")
