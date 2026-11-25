"""Водораздел и пиксельная корреляция поверх общей предобработки."""

from collections import defaultdict
from dataclasses import replace

import cv2 as cv
import numpy as np

from main import Face, PATTERNS, face_candidates, face_coverage, face_poses, select_faces

MIN_CORRELATION = 0.60
_axis = np.linspace(-1.55, 1.55, 35, dtype=np.float32)
PATCH_COORDS = np.stack(np.meshgrid(_axis, _axis), axis=-1)
PIP_SIGMA = 0.25
PIXEL_TEMPLATES = {
    value: np.exp(
        -np.square(PATCH_COORDS[:, :, None] - points).sum(axis=-1) / (2 * PIP_SIGMA ** 2)
    ).max(axis=-1).astype(np.float32)
    for value, points in PATTERNS.items()
}


def watershed_regions(image, mask):
    """Маркеры из локальных максимумов расстояния до фона."""
    # Заполняем отверстия от точек до поиска маркеров внутри тела кубика.
    closing_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (19, 19))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, closing_kernel)
    distance = cv.distanceTransform(mask, cv.DIST_L2, cv.DIST_MASK_PRECISE)
    distance = cv.GaussianBlur(distance, (0, 0), 2)
    neighborhood = cv.getStructuringElement(cv.MORPH_ELLIPSE, (29, 29))
    peaks = (distance >= 10) & (distance == cv.dilate(distance, neighborhood))
    seeds = cv.dilate(peaks.astype(np.uint8), np.ones((5, 5), dtype=np.uint8)) & mask
    _, components = cv.connectedComponents(seeds)

    # Ноль оставляет пиксель без метки. Фон и каждый пик имеют свои метки.
    markers = np.zeros(mask.shape, dtype=np.int32)
    markers[cv.dilate(mask, np.ones((5, 5), dtype=np.uint8)) == 0] = 1
    markers[seeds != 0] = components[seeds != 0] + 1
    cv.watershed(image, markers)
    markers[(mask == 0) | (markers == 1)] = 0
    return markers


def recognize_watershed(image, prepared):
    """Подбирает один геометрический шаблон в каждой области водораздела."""
    left, top, right, bottom = prepared.zone
    x0, y0 = max(0, left - 65), max(0, top - 65)
    crop = np.s_[y0 : bottom + 65, x0 : right + 65]
    smoothed = cv.medianBlur(image[crop], 13)
    regions = []
    candidates = []
    for color, (mask, points) in enumerate(zip(prepared.masks, prepared.groups)):
        labels = watershed_regions(smoothed, mask[crop])
        regions.append(labels)
        by_region = defaultdict(list)
        for index, (x, y, _) in enumerate(points):
            label = labels[round(y) - y0, round(x) - x0]
            if label > 1:
                by_region[label].append(index)

        coords = np.array(points)
        for label, indices in by_region.items():
            options = face_candidates(coords[indices], mask, color)
            options = [
                face for face in options
                if labels[round(face.center[1]) - y0, round(face.center[0]) - x0] == label
            ]
            if options:
                best = max(options, key=lambda face: face.score)
                candidates.append(replace(best, pips=frozenset(indices[i] for i in best.pips)))

    return replace(prepared, faces=select_faces(candidates)), regions, crop


def recognize_correlation(prepared):
    """Проверяет положения граней корреляцией патчей локального контраста."""
    contrast = prepared.contrast.astype(np.float32)
    candidates = []
    for color, (mask, points) in enumerate(zip(prepared.masks, prepared.groups)):
        if not points:
            continue
        # Тёмные точки на белом кубике приводим к той же полярности, что светлые.
        signal = np.maximum(-contrast if color == 3 else contrast, 0)
        for center, transform, values in face_poses(np.array(points)[:, :2]):
            coverage = face_coverage(mask, center, transform)
            if coverage < 0.73:
                continue
            coords = (center + PATCH_COORDS @ transform).astype(np.float32)
            patch = cv.remap(signal, coords, None, cv.INTER_LINEAR)
            for value in values:
                coefficient = float(cv.matchTemplate(
                    patch, PIXEL_TEMPLATES[value], cv.TM_CCOEFF_NORMED
                )[0, 0])
                if coefficient >= MIN_CORRELATION:
                    candidates.append(Face(center, value, color, frozenset(), (coefficient, coverage)))

    return replace(prepared, faces=select_faces(candidates))
