"""Пакетное сравнение трёх способов распознавания кубиков."""

import argparse
from collections import Counter
import csv
import json
from math import dist
from pathlib import Path

import cv2 as cv

from experimental_methods import recognize_correlation, recognize_watershed
from main import FORMATS, inside, read_photo, recognize
from visualization import save_method_distribution, save_method_preview

METHODS = {
    "geometry": "Геометрия",
    "watershed": "Водораздел + геометрия",
    "correlation": "Корреляция",
}
MATCH_RADIUS = 22
METRICS = ("reference_dice", "reference_points", "located", "correct", "missed", "extra", "wrong_value")


def compare_with_reference(faces, reference):
    # Сначала сопоставляем центры, затем смотрим на значения у полученных пар.
    pairs = sorted(
        (dist(face.center, (x, y)), index, target)
        for index, face in enumerate(faces)
        for target, (x, y, _) in enumerate(reference)
    )
    used_faces, used_targets = set(), set()
    correct = 0
    for distance, index, target in pairs:
        if distance > MATCH_RADIUS:
            break
        if index in used_faces or target in used_targets:
            continue
        used_faces.add(index)
        used_targets.add(target)
        correct += faces[index].value == reference[target][2]

    located = len(used_targets)
    return {
        "reference_dice": len(reference),
        "reference_points": sum(value for _, _, value in reference),
        "located": located,
        "correct": correct,
        "missed": len(reference) - located,
        "extra": len(faces) - located,
        "wrong_value": located - correct,
    }


def run_comparison(paths, output, reference):
    frequencies = {method: Counter() for method in METHODS}
    metrics = {method: Counter() for method in METHODS}
    preview = next((path for path in paths if path.name in reference), paths[0])
    value_fields = [f"value_{value}" for value in range(1, 7)]
    with (output / "per_image.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, ["image", "method", "dice", "points", *value_fields, *METRICS])
        writer.writeheader()
        for path in paths:
            image = read_photo(path)
            baseline = recognize(image)
            segmented, regions, region_crop = recognize_watershed(image, baseline)
            correlated = recognize_correlation(baseline)
            results = dict(zip(METHODS, (baseline, segmented, correlated)))
            visible = {
                method: [face for face in result.faces if inside(face.center, result.zone)]
                for method, result in results.items()
            }
            truth = reference.get(path.name)
            image_summary = []
            for method, faces in visible.items():
                count = Counter(face.value for face in faces)
                points = sum(face.value for face in faces)
                frequencies[method].update(count)
                scores = compare_with_reference(faces, truth) if truth is not None else {}
                metrics[method].update(scores)
                writer.writerow({
                    "image": path.name, "method": method, "dice": len(faces), "points": points,
                    **{field: count[value] for value, field in enumerate(value_fields, 1)}, **scores,
                })
                image_summary.append(f"{METHODS[method]} — {len(faces)} куб., {points} очк.")
            print(f"{path.name}: {' | '.join(image_summary)}")

            if path == preview:
                save_method_preview(
                    output / f"{path.name}.jpg", image, baseline.zone, visible, METHODS,
                    truth, regions, region_crop,
                )

    save_method_distribution(output / "distribution.png", frequencies, METHODS, len(paths))
    print("\nИтог по всем снимкам")
    for method, title in METHODS.items():
        count = frequencies[method]
        points = sum(value * amount for value, amount in count.items())
        values = ", ".join(str(count[value]) for value in range(1, 7))
        print(f"{title}: {sum(count.values())} куб., {points} очк. | частоты 1–6: {values}")
        score = metrics[method]
        if score:
            print(
                f"  Ручная разметка: {score['located']}/{score['reference_dice']} найдено, "
                f"{score['correct']} с верным значением, {score['missed']} пропусков, "
                f"{score['extra']} лишних, {score['wrong_value']} ошибок значения"
            )


def main():
    cv.setNumThreads(1)
    root = Path(__file__).resolve().parent
    dataset = root / "кубики "
    parser = argparse.ArgumentParser(description="Сравнение методов распознавания кубиков.")
    parser.add_argument("input", nargs="?", type=Path, default=dataset, help="Снимок или папка со снимками")
    parser.add_argument("-o", "--output", type=Path, default=root / "results" / "comparison")
    args = parser.parse_args()

    paths = sorted(args.input.iterdir()) if args.input.is_dir() else [args.input]
    paths = [path for path in paths if path.is_file() and path.suffix.lower() in FORMATS]
    if not paths:
        parser.error(f"Снимки не найдены: {args.input}")
    if any(path.resolve().parent == args.output.resolve() for path in paths):
        parser.error("Укажите отдельную папку для результатов.")

    try:
        reference = {}
        if all(path.resolve().parent == dataset for path in paths):
            reference = json.loads((root / "comparison_reference.json").read_text(encoding="utf-8"))
        args.output.mkdir(parents=True, exist_ok=True)
        run_comparison(paths, args.output, reference)
        print(f"Результаты сохранены в {args.output.resolve()}")
    except (OSError, ValueError, cv.error) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
