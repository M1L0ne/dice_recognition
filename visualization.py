"""Изображения этапов распознавания и график частот."""

import cv2 as cv
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator


def mark_pips(image, points, color):
    overlay = image.copy()
    for x, y, _ in points:
        center = (round(x), round(y))
        cv.circle(overlay, center, 8, (24, 24, 24), 3, cv.LINE_AA)
        cv.circle(overlay, center, 8, color, 1, cv.LINE_AA)
    return overlay


def save_stages(path, source, annotated, recognition, visible):
    left, top, right, bottom = recognition.zone
    # Показываем зону с запасом, в котором детектор также ищет точки.
    crop = (slice(max(0, top - 45), bottom + 45), slice(max(0, left - 45), right + 45))
    figure = Figure(figsize=(14, 8.4), layout="constrained", facecolor="#f6f8fc")
    figure.suptitle(f"Этапы распознавания · {path.stem}", fontsize=17, fontweight="bold")
    grid = figure.add_gridspec(2, 3)

    def show(ax, image, title, *, binary=False, small=False):
        image = image[crop]
        if image.ndim == 3:
            ax.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        else:
            ax.imshow(image, cmap="gray", vmin=0, vmax=1 if binary else 255)
        ax.set_title(title, loc="left", fontsize=9 if small else 12, fontweight="bold")
        ax.set_axis_off()

    candidates = [point for group in recognition.groups for point in group]
    matched = [recognition.groups[face.color][index] for face in visible for index in face.pips]
    show(figure.add_subplot(grid[0, 0]), source, "1. Зона анализа")
    contrast = cv.convertScaleAbs(recognition.contrast, alpha=2.5)
    show(figure.add_subplot(grid[0, 1]), contrast, "2. Локальный контраст")

    mask_panel = figure.add_subfigure(grid[0, 2], facecolor=figure.get_facecolor())
    mask_panel.suptitle("3. Маски по цвету кубиков", fontsize=12, fontweight="bold")
    for ax, mask, title in zip(
        mask_panel.subplots(2, 2).flat,
        recognition.masks,
        ("Чёрные", "Зелёные", "Красные", "Белые"),
    ):
        show(ax, mask, title, binary=True, small=True)

    show(
        figure.add_subplot(grid[1, 0]), mark_pips(source, candidates, (55, 196, 255)),
        f"4. Точки после фильтров ({len(candidates)})",
    )
    show(
        figure.add_subplot(grid[1, 1]), mark_pips(source, matched, (75, 255, 135)),
        f"5. Точки выбранных граней ({len(matched)})",
    )
    score = sum(face.value for face in visible)
    show(
        figure.add_subplot(grid[1, 2]), annotated,
        f"6. Результат · {len(visible)} кубиков, очков {score}",
    )
    figure.savefig(path, dpi=150, pil_kwargs={"quality": 90})


def save_distribution(path, frequencies, photo_count):
    values = range(1, 7)
    counts = [frequencies[value] for value in values]
    total = sum(counts)
    figure = Figure(figsize=(8, 4.8), layout="constrained", facecolor="white")
    figure.suptitle("Распределение очков на кубиках", fontsize=18, fontweight="bold")
    ax = figure.subplots()
    bars = ax.bar(values, counts, width=0.6, color="#256b85")
    labels = [
        f"{count}\n({100 * count / total if total else 0:.1f} %)".replace(".", ",")
        for count in counts
    ]
    ax.bar_label(bars, labels=labels, padding=4, fontsize=10)
    ax.set_title(f"Снимков: {photo_count}  ·  Распознано кубиков: {total}", fontsize=11, pad=12)
    ax.set(xlabel="Очки на верхней грани", ylabel="Количество кубиков", xticks=list(values))
    ax.set_ylim(0, max(counts) * 1.23 if total else 1)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#dde5eb", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    figure.savefig(path, dpi=175)


METHOD_COLORS = ("#177965", "#3c69af", "#b24f56")


def save_method_distribution(path, frequencies, names, photo_count):
    values = np.arange(1, 7)
    figure = Figure(figsize=(9, 5.2), layout="constrained", facecolor="white")
    figure.suptitle("Распределение очков по методам", fontsize=18, fontweight="bold")
    ax = figure.subplots()
    for offset, (method, color) in enumerate(zip(names, METHOD_COLORS), -1):
        counts = [frequencies[method][value] for value in values]
        bars = ax.bar(values + 0.25 * offset, counts, width=0.23, color=color, label=names[method])
        ax.bar_label(bars, padding=3, fontsize=8)
    ax.set_title(f"Снимков: {photo_count}", fontsize=11, pad=12)
    ax.set(xlabel="Очки на верхней грани", ylabel="Количество кубиков", xticks=values)
    maximum = max((amount for count in frequencies.values() for amount in count.values()), default=0)
    ax.set_ylim(top=maximum * 1.16 + 1)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#dde5eb", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncols=3, frameon=False)
    figure.savefig(path, dpi=155)


def save_method_preview(path, image, zone, visible, names, reference, regions, region_crop):
    figure = Figure(figsize=(14, 8.5), layout="constrained", facecolor="#f5f7fa")
    figure.suptitle(f"Сравнение методов · {path.stem}", fontsize=17, fontweight="bold")
    grid = figure.add_gridspec(2, 6)
    left, top, right, bottom = zone
    image_height, image_width = image.shape[:2]
    x0, x1 = max(0, left - 26), min(image_width, right + 26)
    y0, y1 = max(0, top - 26), min(image_height, bottom + 26)
    rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    def panel(slot, view, title, points=(), color="#f5b549"):
        ax = figure.add_subplot(slot)
        ax.imshow(view[y0:y1, x0:x1], extent=(x0 - 0.5, x1 - 0.5, y1 - 0.5, y0 - 0.5))
        ax.add_patch(Rectangle(
            (left, top), right - left, bottom - top, fill=False,
            edgecolor="#68d4ef", linewidth=0.8, linestyle="--",
        ))
        for x, y, value in points:
            ax.scatter(x, y, s=21, facecolors="none", edgecolors=color, linewidths=1.3)
            ax.annotate(str(value), (x, y), xytext=(5, 4), textcoords="offset points",
                        color="white", weight="bold", fontsize=9,
                        bbox={"boxstyle": "round,pad=0.13", "fc": color, "ec": "none"})
        ax.set(xlim=(x0, x1), ylim=(y1, y0), title=title)
        ax.set_axis_off()

    reference_title = "Ручная разметка" if reference is not None else "Исходный снимок"
    panel(grid[0, :3], rgb, reference_title, reference or (), "#79521b")
    segmented = rgb.copy()
    cropped = segmented[region_crop]
    palette = np.array([
        [249, 171, 87], [109, 215, 225], [214, 142, 245], [171, 224, 111],
        [249, 158, 174], [234, 223, 127], [145, 175, 255],
    ])
    for color, labels in enumerate(regions):
        foreground = labels > 1
        cropped[foreground] = (0.52 * cropped[foreground] + 0.48 * palette[(labels[foreground] + color) % len(palette)])
    panel(grid[0, 3:], segmented, "Области после водораздела")

    for index, (method, color) in enumerate(zip(names, METHOD_COLORS)):
        points = [(face.center[0], face.center[1], face.value) for face in visible[method]]
        score = sum(value for _, _, value in points)
        title = f"{names[method]} · {len(points)} куб. / {score} очк."
        panel(grid[1, 2 * index : 2 * index + 2], rgb, title, points, color)
    figure.savefig(path, dpi=145, pil_kwargs={"quality": 90})
