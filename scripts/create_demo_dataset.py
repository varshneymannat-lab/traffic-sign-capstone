from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


SIGN_CLASSES = ["stop", "speed_limit", "traffic_light", "pedestrian_crossing", "no_entry"]


def _regular_polygon(cx: int, cy: int, radius: int, sides: int, rotation: float = 0.0):
    return [
        (
            cx + radius * math.cos(rotation + 2 * math.pi * i / sides),
            cy + radius * math.sin(rotation + 2 * math.pi * i / sides),
        )
        for i in range(sides)
    ]


def draw_sign(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], label: str) -> None:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    w = x2 - x1
    h = y2 - y1
    font = ImageFont.load_default()

    if label == "stop":
        points = _regular_polygon(cx, cy, min(w, h) // 2, 8, rotation=math.pi / 8)
        draw.polygon(points, fill=(205, 35, 35), outline=(245, 245, 245))
        draw.text((cx - 13, cy - 5), "STOP", fill=(255, 255, 255), font=font)
    elif label == "speed_limit":
        draw.ellipse([x1, y1, x2, y2], fill=(245, 245, 245), outline=(210, 40, 40), width=5)
        draw.text((cx - 10, cy - 10), "50", fill=(25, 25, 25), font=font)
    elif label == "traffic_light":
        draw.rounded_rectangle([cx - w // 5, y1, cx + w // 5, y2], radius=8, fill=(35, 35, 35))
        for i, color in enumerate([(215, 35, 35), (235, 185, 30), (35, 170, 70)]):
            light_y = y1 + h // 5 + i * h // 3
            draw.ellipse([cx - 8, light_y - 8, cx + 8, light_y + 8], fill=color)
    elif label == "pedestrian_crossing":
        draw.polygon([(cx, y1), (x2, y2), (x1, y2)], fill=(40, 95, 190), outline=(245, 245, 245))
        draw.line([cx, y1 + 18, cx, y2 - 18], fill=(255, 255, 255), width=4)
        draw.line([cx, cy, x1 + 18, y2 - 12], fill=(255, 255, 255), width=4)
        draw.line([cx, cy, x2 - 18, y2 - 12], fill=(255, 255, 255), width=4)
        draw.ellipse([cx - 5, y1 + 10, cx + 5, y1 + 20], fill=(255, 255, 255))
    elif label == "no_entry":
        draw.ellipse([x1, y1, x2, y2], fill=(210, 35, 35), outline=(245, 245, 245), width=3)
        draw.rectangle([x1 + 12, cy - 7, x2 - 12, cy + 7], fill=(245, 245, 245))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a tiny synthetic traffic-sign dataset.")
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--samples-per-class", type=int, default=18)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for old_image in image_dir.glob("demo_*.jpg"):
        old_image.unlink()

    rows = []
    idx = 0
    for sign_label in SIGN_CLASSES:
        for _ in range(args.samples_per_class):
            canvas = Image.new(
                "RGB",
                (224, 160),
                color=(rng.randint(185, 225), rng.randint(205, 235), rng.randint(220, 245)),
            )
            draw = ImageDraw.Draw(canvas)
            draw.rectangle([0, 112, 224, 160], fill=(85, 90, 92))
            draw.line([0, 136, 224, 136], fill=(230, 220, 120), width=2)

            size = rng.randint(56, 78)
            x1 = rng.randint(58, 224 - size - 38)
            y1 = rng.randint(18, 92 - size // 2)
            x2 = x1 + size
            y2 = y1 + size
            pole_x = (x1 + x2) // 2
            pole_top = min(y2 - 2, 118)
            pole_bottom = max(y2 - 2, 118)
            draw.rectangle([pole_x - 3, pole_top, pole_x + 3, pole_bottom], fill=(90, 90, 90))
            draw_sign(draw, (x1, y1, x2, y2), sign_label)

            name = f"demo_{idx:04d}_{sign_label}.jpg"
            canvas.save(image_dir / name, quality=95)
            rows.append(
                {
                    "image_path": f"images/{name}",
                    "xmin": x1,
                    "ymin": y1,
                    "xmax": x2,
                    "ymax": y2,
                    "sign_label": sign_label,
                }
            )
            idx += 1

    annotations_path = output_dir / "annotations.csv"
    pd.DataFrame(rows).to_csv(annotations_path, index=False)
    print(f"Wrote {len(rows)} synthetic traffic-sign samples to {annotations_path}")


if __name__ == "__main__":
    main()
