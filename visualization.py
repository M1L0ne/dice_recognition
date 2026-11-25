"""Изображения этапов распознавания и график частот."""

import cv2 as cv
from matplotlib.figure import Figure
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
