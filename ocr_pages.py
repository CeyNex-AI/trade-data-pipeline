#!/usr/bin/env python3
"""OCR PNG pages with a local Qwen3-VL-2B-Instruct model, preserving layout."""

import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

PROMPT = (
    "Transcribe all text visible in this image exactly as it appears. "
    "Preserve the original layout as closely as possible: keep line breaks, "
    "paragraph spacing, indentation, tables, and columns in their original "
    "positions using plain-text spacing/alignment. Do not summarize, "
    "translate, or add commentary. Output only the transcribed text."
)


def load_model(model_dir: str):
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_dir, dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_dir)
    return model, processor


def ocr_image(model, processor, image_path: Path, max_new_tokens: int) -> str:
    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    text = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return text.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images-dir", default="pages", help="Directory containing PNG pages"
    )
    parser.add_argument(
        "--model-dir",
        default="Qwen3-VL-2B-Instruct",
        help="Path to local Qwen3-VL-2B-Instruct model weights",
    )
    parser.add_argument(
        "--output-dir",
        default="ocr_output",
        help="Directory to write per-page .txt files",
    )
    parser.add_argument(
        "--combined-output",
        default="ocr_output/combined.txt",
        help="Path to write a single combined text file (set to '' to skip)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(images_dir.glob("*.png"))
    if not image_paths:
        raise SystemExit(f"No PNG files found in {images_dir}")

    print(f"Loading model from {args.model_dir} ...")
    model, processor = load_model(args.model_dir)

    combined_chunks = []
    for image_path in image_paths:
        print(f"OCR: {image_path.name}")
        text = ocr_image(model, processor, image_path, args.max_new_tokens)

        out_path = output_dir / f"{image_path.stem}.txt"
        out_path.write_text(text, encoding="utf-8")
        print(f"  -> {out_path}")

        combined_chunks.append(f"===== {image_path.name} =====\n{text}\n")

    if args.combined_output:
        combined_path = Path(args.combined_output)
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        combined_path.write_text("\n".join(combined_chunks), encoding="utf-8")
        print(f"Combined output written to {combined_path}")


if __name__ == "__main__":
    main()
