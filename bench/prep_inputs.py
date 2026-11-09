#!/usr/bin/env python3
"""Make placeholder OCR pages so the pipeline can be smoke-tested end to end.

These are clean, typed English pages: far easier than real homework photos.
For the real baseline, replace them with 20-50 phone photos of homework
sheets (with handwriting) and textbook pages in your students' languages.

  python bench/prep_inputs.py --pages-dir /data/tokenfactory/bench-inputs/pages --n 12
"""
import argparse
import random
from pathlib import Path

LINES = [
    "Exercise 4.2  Answer the following questions.",
    "1. Find the area of a triangle whose base is 6 cm and height is 4 cm.",
    "2. A train travels 120 km in 2 hours. What is its average speed?",
    "3. Simplify: (3/4) + (5/6) - (1/3)",
    "4. Solve for x: 2x + 7 = 19",
    "5. State Newton's second law of motion and give one example.",
    "6. Why do we see lightning before we hear thunder?",
    "7. Write the chemical formula for water and for carbon dioxide.",
    "8. The ratio of boys to girls in a class is 3 : 2. If there are 40 students, how many are girls?",
    "Answer: Area = 1/2 x base x height = 1/2 x 6 x 4 = 12 sq cm",
    "Speed = distance / time = 120 / 2 = 60 km/h",
    "Photosynthesis: carbon dioxide + water -> glucose + oxygen (in sunlight)",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pages-dir", required=True)
    ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()
    from PIL import Image, ImageDraw, ImageFont

    out = Path(a.pages_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 28)
        head = ImageFont.truetype("DejaVuSans-Bold.ttf", 36)
    except OSError:
        font = head = ImageFont.load_default()
    rng = random.Random(0)
    for i in range(a.n):
        img = Image.new("RGB", (1240, 1754), "white")  # A4 at 150 dpi
        d = ImageDraw.Draw(img)
        d.text((80, 70), f"Chapter {i % 9 + 1}: Practice Sheet", font=head, fill="black")
        y = 160
        for line in rng.sample(LINES, k=len(LINES)):
            d.text((80, y), line, font=font, fill="black")
            y += 60
        d.text((600, 1680), str(100 + i), font=font, fill="black")
        img.save(out / f"synthetic_{i:03d}.png")
    print(f"wrote {a.n} placeholder pages to {out}")


if __name__ == "__main__":
    main()
