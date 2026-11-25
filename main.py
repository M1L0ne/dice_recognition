"""Распознавание точек на игральных кубиках с помощью OpenCV."""

import argparse
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import cv2 as cv
import numpy as np

from visualization import save_distribution, save_stages

WORK_WIDTH = 1200
# Примерные границы внутри коробки (доли ширины и высоты снимка).
BOX = (0.15, 0.025, 0.95, 0.98)
ZONE_FRACTION = 0.5
FORMATS = {".jpg", ".jpeg", ".png", ".webp"}
GREEN = (75, 255, 135)

# Координаты точек относительно центра верхней грани.
CORNERS = [(-1, -1), (1, 1), (-1, 1), (1, -1)]
PATTERNS = {
    1: np.array([(0, 0)]),
    2: np.array(CORNERS[:2]),
    3: np.array(CORNERS[:2] + [(0, 0)]),
    4: np.array(CORNERS),
    5: np.array(CORNERS + [(0, 0)]),
    6: np.array(CORNERS + [(0, -1), (0, 1)]),
}
# Сетка внутри шаблона для проверки цвета грани.
FACE_SAMPLES = np.array(
    [(x, y) for x in np.linspace(-1.45, 1.45, 7) for y in np.linspace(-1.45, 1.45, 7)]
)


@dataclass
class Face:
    center: np.ndarray
    value: int
    color: int
    pips: frozenset
    score: tuple


@dataclass
class Recognition:
    faces: list[Face]
    masks: list[np.ndarray]
    contrast: np.ndarray
    zone: tuple[int, int, int, int]
    groups: list[list[tuple[float, float, float]]]


def analysis_zone(shape):
    height, width = shape[:2]
    left, top, right, bottom = BOX
    margin = (1 - ZONE_FRACTION ** 0.5) / 2
    margin_x, margin_y = (right - left) * margin, (bottom - top) * margin
    return (
        round((left + margin_x) * width),
        round((top + margin_y) * height),
        round((right - margin_x) * width),
        round((bottom - margin_y) * height),
    )


def inside(point, zone, margin=0):
    x, y = point
    left, top, right, bottom = zone
    return left - margin <= x < right + margin and top - margin <= y < bottom + margin


def color_masks(image, gray):
    blue, green, red = np.moveaxis(image.astype(np.int16), -1, 0)
    red_dice = (red - green > 26) & (red - blue > 30) & (red > 95)
    green_dice = (green - red > 15) & (green - blue > 10) & (green > 52)
    white_dice = (gray > 128) & (red - blue < 14)
    black_dice = (gray < 91) & ~green_dice & ~red_dice
    # Закрытие заполняет отверстия от точек, сохраняя поверхность кубика.
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (13, 13))
    return [
        cv.morphologyEx(mask.astype(np.uint8), cv.MORPH_CLOSE, kernel)
        for mask in (black_dice, green_dice, red_dice, white_dice)
    ]


def find_pips(gray, contrast, masks, zone):
    bright = gray > 125
    thresholds = [
        ((contrast > 25) & bright, False),
        ((contrast > 42) & bright, False),  # Повтор для слившихся светлых точек.
        ((contrast < -24) & (gray < 113), True),
    ]
    groups = [[] for _ in masks]
    for binary, dark in thresholds:
        contours, _ = cv.findContours(
            binary.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            x, y, width, height = cv.boundingRect(contour)
            area = cv.contourArea(contour)
            if not inside((x, y), zone, margin=45):
                continue
            if not (22 <= area <= 145 and 6 <= min(width, height) <= max(width, height) <= 16):
                continue
            _, (width, height), _ = cv.minAreaRect(contour)
            if min(width, height) < 0.56 * max(width, height):
                continue  # Точки на боковой грани обычно вытянуты.
            moments = cv.moments(contour)
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
            px, py = round(cx), round(cy)
            patch = np.s_[max(0, py - 6) : py + 7, max(0, px - 6) : px + 7]
            coverage = [mask[patch].mean() for mask in masks]
            color = 3 if dark else int(np.argmax(coverage[:3]))
            if coverage[color] <= 0.55:
                continue
            if any((cx - a) ** 2 + (cy - b) ** 2 < 25 for a, b, *_ in groups[color]):
                continue
            groups[color].append((cx, cy, area))
    return groups


def face_candidates(points, mask, color):
    coords, areas = points[:, :2], points[:, 2]
    candidates = []

    def check(value, center, transform):
        # Проверяем, что шаблон лежит на поверхности нужного цвета.
        xs, ys = np.rint(center + FACE_SAMPLES @ transform).astype(int).T
        if min(xs.min(), ys.min()) < 0 or xs.max() >= mask.shape[1] or ys.max() >= mask.shape[0]:
            return
        coverage = mask[ys, xs].mean()
        if coverage < 0.73:
            return

        expected = center + PATTERNS[value] @ transform
        distances = np.linalg.norm(expected[:, None] - coords, axis=2)
        matches = distances.argmin(axis=1)
        errors = distances[np.arange(value), matches]
        if errors.max() > 4.4 or len(set(matches)) != value:
            return
        # Считаем лишние точки в координатах шаблона.
        local = (coords - center) @ np.linalg.inv(transform)
        extras = np.abs(local).max(axis=1) < 1.45
        extras[matches] = False
        matched = value - extras.sum()
        if matched <= 0:
            return
        # Сначала сравниваем число совпадений с учётом лишних точек,
        # затем точность совпадений, размер точек и цвет поверхности.
        score = (matched, -errors.mean(), areas[matches].mean(), coverage)
        candidates.append(Face(center, value, color, frozenset(matches), score))

    # У единицы единственная точка задаёт центр.
    single_dot = 12 * np.eye(2)
    for center in coords:
        check(1, center, single_dot)

    # У остальных граней две крайние точки задают диагональ.
    for first, second in combinations(coords, 2):
        dx, dy = (second - first) / 2
        if not 10.5 ** 2 <= dx * dx + dy * dy <= 19.5 ** 2:
            continue
        center = (first + second) / 2
        for ratio in (1.0, 0.67, -0.67, 0.8, -0.8, 1.5, -1.5):
            # Векторы вдоль сторон грани в сумме дают половину диагонали.
            axis_x = (dx + ratio * dy) / (1 + ratio ** 2)
            axis_y = (dy - ratio * dx) / (1 + ratio ** 2)
            transform = np.array([[axis_x, axis_y], [-ratio * axis_y, ratio * axis_x]])
            # У шестёрки шаг между тремя рядами меньше расстояния между столбцами.
            values = (2, 3, 4, 5) if ratio == 1 else (6,)
            for value in values:
                check(value, center, transform)
    return candidates


def recognize(image):
    zone = analysis_zone(image.shape)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # Разность двух размытий выделяет небольшие светлые и тёмные точки.
    smooth = cv.GaussianBlur(gray, (0, 0), 1).astype(np.int16)
    contrast = smooth - cv.GaussianBlur(gray, (0, 0), 5)
    masks = color_masks(image, gray)
    groups = find_pips(gray, contrast, masks, zone)

    candidates = []
    for color, points in enumerate(groups):
        if points:
            candidates.extend(face_candidates(np.array(points), masks[color], color))

    # Шаблоны с лучшей оценкой получают точки первыми.
    faces = []
    for face in sorted(candidates, key=lambda item: item.score, reverse=True):
        if any(
            np.linalg.norm(face.center - other.center) < 31
            or (face.color == other.color and face.pips & other.pips)
            for other in faces
        ):
            continue
        faces.append(face)
    return Recognition(faces, masks, contrast, zone, groups)


def die_box(face, mask, faces):
    # Выделяем тело кубика вокруг найденной грани. Общие пиксели
    # соседних кубиков оставляем ближайшей к ним грани.
    cx, cy = np.rint(face.center).astype(int)
    x0, x1 = max(0, cx - 35), min(mask.shape[1], cx + 36)
    y0, y1 = max(0, cy - 35), min(mask.shape[0], cy + 36)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    distance = (xx - face.center[0]) ** 2 + (yy - face.center[1]) ** 2
    body = (mask[y0:y1, x0:x1] != 0) & (distance <= 35 ** 2)
    for other in faces:
        if other is face or other.color != face.color:
            continue
        ox, oy = other.center
        if (ox - cx) ** 2 + (oy - cy) ** 2 < 70 ** 2:
            body &= distance <= (xx - ox) ** 2 + (yy - oy) ** 2

    # Заливка от центра находит связную часть маски и сразу даёт её рамку.
    body = body.astype(np.uint8)
    seed = (cx - x0, cy - y0)
    if body[seed[1], seed[0]]:
        area, _, _, (x, y, width, height) = cv.floodFill(body, None, seed, 2, flags=8)
        if area >= 400:
            return x0 + x, y0 + y, x0 + x + width - 1, y0 + y + height - 1
    return (
        max(0, cx - 24), max(0, cy - 24),
        min(mask.shape[1] - 1, cx + 24), min(mask.shape[0] - 1, cy + 24),
    )


def draw_zone(image, zone):
    left, top, right, bottom = zone
    color = (255, 190, 80)
    for x in range(left, right, 16):
        for y in (top, bottom):
            cv.line(image, (x, y), (min(x + 8, right), y), color, 1, cv.LINE_AA)
    for y in range(top, bottom, 16):
        for x in (left, right):
            cv.line(image, (x, y), (x, min(y + 8, bottom)), color, 1, cv.LINE_AA)


def process(path, output, save_contrast, *, stages=False):
    target = output / path.name
    if target.resolve() == path.resolve():
        raise ValueError("Укажите отдельную папку для результатов.")

    source = cv.imread(str(path))
    if source is None:
        raise ValueError(f"Не удалось прочитать снимок {path}")

    height = max(1, round(source.shape[0] * WORK_WIDTH / source.shape[1]))
    image = cv.resize(source, (WORK_WIDTH, height), interpolation=cv.INTER_AREA)
    recognition = recognize(image)
    zone = recognition.zone
    annotated = image.copy() if stages else image
    draw_zone(annotated, zone)
    visible = [face for face in recognition.faces if inside(face.center, zone)]
    for face in visible:
        x1, y1, x2, y2 = die_box(face, recognition.masks[face.color], recognition.faces)
        cv.rectangle(annotated, (x1, y1), (x2, y2), GREEN, 2)
        label = str(face.value)
        (width, _), _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        position = ((x1 + x2 - width) // 2, max(17, y1 - 5))
        for color, thickness in [((15, 15, 15), 4), (GREEN, 2)]:
            cv.putText(
                annotated, label, position, cv.FONT_HERSHEY_SIMPLEX, 0.65, color, thickness, cv.LINE_AA
            )

    if not cv.imwrite(str(target), annotated):
        raise ValueError(f"Не удалось сохранить {target}")
    if save_contrast:
        contrast_dir = output / "contrast"
        contrast_dir.mkdir(exist_ok=True)
        contrast_image = cv.convertScaleAbs(recognition.contrast, alpha=2.5)
        contrast_image = cv.cvtColor(contrast_image, cv.COLOR_GRAY2BGR)
        draw_zone(contrast_image, zone)
        contrast_path = contrast_dir / f"{path.name}.png"
        if not cv.imwrite(str(contrast_path), contrast_image, [cv.IMWRITE_PNG_COMPRESSION, 3]):
            raise ValueError(f"Не удалось сохранить {contrast_path}")
    if stages:
        stages_dir = output / "stages"
        stages_dir.mkdir(exist_ok=True)
        draw_zone(image, zone)
        save_stages(stages_dir / f"{path.name}.jpg", image, annotated, recognition, visible)
    return [face.value for face in visible]


def main():
    cv.setNumThreads(1)
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Распознавание цифр на кубиках.")
    parser.add_argument(
        "input", type=Path, nargs="?", default=project_dir / "кубики ",
        help="Снимок или папка со снимками",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=project_dir / "results",
        help="Папка результатов (по умолчанию рядом с main.py)",
    )
    parser.add_argument("--contrast", action="store_true", help="Сохранить карту контраста")
    parser.add_argument(
        "--stages", action="store_true", help="Сохранить картинки этапов для каждого снимка",
    )
    args = parser.parse_args()

    paths = sorted(args.input.iterdir()) if args.input.is_dir() else [args.input]
    paths = [path for path in paths if path.is_file() and path.suffix.lower() in FORMATS]
    if not paths:
        parser.error(f"Снимки не найдены: {args.input}")

    try:
        args.output.mkdir(parents=True, exist_ok=True)
        frequencies = Counter()
        for path in paths:
            values = process(path, args.output, args.contrast, stages=args.stages)
            frequencies.update(values)
            print(f"{path.name}: кубиков в зоне — {len(values)}, сумма очков — {sum(values)}")
        statistics = args.output / "statistics"
        statistics.mkdir(exist_ok=True)
        chart = statistics / "distribution.png"
        save_distribution(chart, frequencies, len(paths))
        print(f"Всего распознано кубиков: {sum(frequencies.values())}")
        summary = ", ".join(f"{value} → {frequencies[value]}" for value in PATTERNS)
        print(f"Частоты по значениям 1–6: {summary}")
        print(f"График распределения: {chart.resolve()}")
        print(f"Результаты сохранены в {args.output.resolve()}")
    except (OSError, ValueError, cv.error) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
