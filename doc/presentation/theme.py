"""Общий стиль и вспомогательные объекты для презентации KODES.

Здесь только «строительные блоки»: палитра, шрифт, текстовые хелперы и
несколько составных мобжектов (сетка, кристалл, полоса памяти, часы).
Сами сцены — в presentation.py.
"""

from __future__ import annotations

import numpy as np
from manim import *

# ──────────────────────────────────────────────────────────────────────────
#  Палитра
#
#  Светлая тема: доклад читают в ярко освещённом зале, поэтому фон белый, а
#  каждый акцент взят достаточно тёмным, чтобы держать контраст к белому не
#  ниже ~5:1 — иначе на проекторе цвет выцветает в серое.
# ──────────────────────────────────────────────────────────────────────────
BG = "#FFFFFF"          # фон (дублируется в manim.cfg)
FG = "#111418"          # основной текст
MUTED = "#4A5568"       # подписи, второстепенное
CPU_C = "#0A54B4"       # всё, что про CPU и про состояния
GPU_C = "#3D7000"       # всё, что про GPU
SCRATCH_C = "#6B3FBF"   # временные (рабочие) массивы
MEM_C = "#8A4600"       # память, счётчики
WARN_C = "#C62828"      # проблема, переполнение, простой
OK_C = "#067A38"        # решено, выигрыш

# Градиент «ключа» балансировщика: холодная ячейка → жёсткая ячейка
KEY_COLD = "#1A4FBF"
KEY_HOT = "#CC2222"


def key_color(t: float) -> str:
    """Цвет ячейки по значению ключа t ∈ [0, 1]."""
    return interpolate_color(ManimColor(KEY_COLD), ManimColor(KEY_HOT), float(np.clip(t, 0, 1)))


# ──────────────────────────────────────────────────────────────────────────
#  Шрифт
# ──────────────────────────────────────────────────────────────────────────
def _pick_font() -> str:
    """DejaVu Sans есть почти везде и знает кириллицу; иначе — шрифт по умолчанию."""
    try:
        import manimpango

        fonts = set(manimpango.list_fonts())
    except Exception:  # noqa: BLE001 — на этапе импорта нам всё равно почему
        return ""
    for candidate in ("DejaVu Sans", "Liberation Sans", "Noto Sans", "FreeSans"):
        if candidate in fonts:
            return candidate
    return ""


FONT = _pick_font()


# ──────────────────────────────────────────────────────────────────────────
#  Текст
# ──────────────────────────────────────────────────────────────────────────
def T(text: str, size: float = 30, color=FG, weight=NORMAL, **kwargs) -> Text:
    """Обычный текст презентации (кириллица идёт через Pango, не через LaTeX)."""
    return Text(text, font=FONT, font_size=size, color=color, weight=weight, **kwargs)


TITLE_MAX_W = 12.4
SAFE_W = 13.0   # кадр 14.22 в ширину; полe запаса, чтобы ничто не упиралось в край


def fit(mobj, max_w: float = SAFE_W):
    """Ужать объект, если он шире безопасной ширины кадра.

    Вешается на каждый текстовый блок: подписи меняются по ходу правок, и без
    этого длинная строка молча уезжает за границу слайда.
    """
    if mobj.width > max_w:
        mobj.scale(max_w / mobj.width)
    return mobj


def title(text: str, color=FG) -> Text:
    """Заголовок слайда, прижатый к верхнему краю (длинный — ужимается)."""
    head = T(text, size=42, color=color, weight=BOLD)
    return fit(head, TITLE_MAX_W).to_edge(UP, buff=0.42)


def subtitle(text: str, color=MUTED) -> Text:
    return T(text, size=26, color=color)


def caption(text: str, color=MUTED) -> Text:
    """Подпись у нижнего края кадра."""
    return T(text, size=24, color=color).to_edge(DOWN, buff=0.42)


def bullets(lines, size: float = 28, buff: float = 0.34, marker="—", color=FG) -> VGroup:
    """Список с маркерами, выровненный по левому краю."""
    rows = VGroup()
    for line in lines:
        dot = T(marker, size=size, color=MUTED)
        txt = T(line, size=size, color=color)
        rows.add(VGroup(dot, txt).arrange(RIGHT, buff=0.28, aligned_edge=UP))
    rows.arrange(DOWN, buff=buff, aligned_edge=LEFT)
    return fit(rows)


def tag(text: str, color=MEM_C, size: float = 22) -> VGroup:
    """Маленькая «плашка» — подпись в рамке."""
    label = T(text, size=size, color=color)
    box = RoundedRectangle(
        width=label.width + 0.34,
        height=label.height + 0.26,
        corner_radius=0.09,
        stroke_color=color,
        stroke_width=2,
        fill_color=color,
        fill_opacity=0.09,
    ).move_to(label)
    return VGroup(box, label)


# ──────────────────────────────────────────────────────────────────────────
#  Расчётная сетка
# ──────────────────────────────────────────────────────────────────────────
def mesh(rows: int = 6, cols: int = 6, cell: float = 0.52, gap: float = 0.06,
         color=CPU_C, fill: float = 0.16) -> VGroup:
    """Сетка ячеек. Порядок обхода — построчный: index = r * cols + c."""
    grid = VGroup()
    for r in range(rows):
        for c in range(cols):
            square = Square(
                side_length=cell,
                stroke_width=2.0,
                stroke_color=color,
                fill_color=color,
                fill_opacity=fill,
            )
            square.move_to(RIGHT * c * (cell + gap) + DOWN * r * (cell + gap))
            grid.add(square)
    grid.center()
    grid.rows, grid.cols = rows, cols
    return grid


def strip(n: int, width: float, height: float = 0.5, gap: float = 0.02,
          color=CPU_C, fill: float = 0.18) -> VGroup:
    """Линейный массив из n клеток суммарной ширины width."""
    w = (width - gap * (n - 1)) / n
    cells = VGroup()
    for _ in range(n):
        cells.add(
            Rectangle(
                width=w,
                height=height,
                stroke_width=1.4,
                stroke_color=color,
                fill_color=color,
                fill_opacity=fill,
            )
        )
    cells.arrange(RIGHT, buff=gap)
    return cells


# ──────────────────────────────────────────────────────────────────────────
#  Кристаллы
# ──────────────────────────────────────────────────────────────────────────
def chip(label: str, w: float = 3.2, h: float = 2.4, color=CPU_C,
         cores=(2, 2), fill: float = 0.05, core_fill: float = 0.32) -> VGroup:
    """Прямоугольник-кристалл с решёткой «ядер» внутри и подписью сверху."""
    body = RoundedRectangle(
        width=w, height=h, corner_radius=0.16,
        stroke_color=color, stroke_width=3.5,
        fill_color=color, fill_opacity=fill,
    )
    nx, ny = cores
    gw, gh = w * 0.68, h * 0.62
    side = min(gw / nx, gh / ny) * 0.74
    cores_group = VGroup()
    for j in range(ny):
        for i in range(nx):
            core = Square(
                side_length=side,
                stroke_width=1.0,
                stroke_color=color,
                fill_color=color,
                fill_opacity=core_fill,
            )
            core.move_to(
                body.get_center()
                + RIGHT * (i - (nx - 1) / 2) * gw / nx
                + DOWN * (j - (ny - 1) / 2) * gh / ny
            )
            cores_group.add(core)
    name = T(label, size=26, color=color, weight=BOLD).next_to(body, UP, buff=0.18)

    group = VGroup(body, cores_group, name)
    group.body, group.cores, group.name = body, cores_group, name
    return group


def cpu_chip(label: str = "CPU") -> VGroup:
    return chip(label, w=3.0, h=2.4, color=CPU_C, cores=(2, 2), core_fill=0.30)


def gpu_chip(label: str = "GPU", w: float = 5.2, h: float = 3.4) -> VGroup:
    return chip(label, w=w, h=h, color=GPU_C, cores=(14, 8), core_fill=0.30)


# ──────────────────────────────────────────────────────────────────────────
#  Память: рамка VRAM и полосы-аллокации
# ──────────────────────────────────────────────────────────────────────────
def vram_frame(label: str, w: float = 8.6, h: float = 4.4, color=GPU_C) -> VGroup:
    box = Rectangle(width=w, height=h, stroke_color=color, stroke_width=3.4,
                    fill_color=color, fill_opacity=0.045)
    # подпись — над рамкой, чтобы не сталкиваться с подписями аллокаций внутри
    name = T(label, size=24, color=color, weight=BOLD)
    name.next_to(box, UP, buff=0.14).align_to(box, LEFT)
    group = VGroup(box, name)
    group.box, group.name = box, name
    return group


def mem_bar(width: float, height: float, color, fill: float = 0.72, label: str | None = None,
            label_size: float = 20) -> VGroup:
    """Полоса-аллокация. У группы есть .bar (сам прямоугольник) и .lab."""
    bar = Rectangle(width=max(width, 0.02), height=height,
                    stroke_color=color, stroke_width=1.6,
                    fill_color=color, fill_opacity=fill)
    group = VGroup(bar)
    group.bar = bar
    group.lab = None
    if label:
        lab = T(label, size=label_size, color=color)
        lab.next_to(bar, LEFT, buff=0.22)
        group.add(lab)
        group.lab = lab
    return group


def resized(bar: Rectangle, width: float, left_x: float) -> Rectangle:
    """Копия полосы новой ширины, растущая вправо от left_x (для Transform)."""
    new = bar.copy().stretch_to_fit_width(max(width, 0.02))
    new.move_to([left_x + max(width, 0.02) / 2, bar.get_y(), 0])
    return new


# ──────────────────────────────────────────────────────────────────────────
#  Мелочи
# ──────────────────────────────────────────────────────────────────────────
def counter(name: str, value, color=MEM_C, size: float = 26) -> VGroup:
    """Подпись + число (Integer), доступное как .value."""
    lab = T(name, size=size, color=MUTED)
    num = Integer(value, color=color, font_size=size * 1.25)
    group = VGroup(lab, num).arrange(RIGHT, buff=0.26)
    group.lab, group.value = lab, num
    return group


def stiff_curve(color=WARN_C, w: float = 1.0, h: float = 0.5) -> VMobject:
    """Профиль воспламенения — «то, что считает солвер» в одной ячейке."""
    curve = FunctionGraph(
        lambda x: np.tanh(7 * (x - 0.15)),
        x_range=[-1, 1, 0.02],
        color=color,
        stroke_width=3,
    )
    curve.stretch_to_fit_width(w).stretch_to_fit_height(h)
    return curve


def brace_label(mobj, text: str, direction=DOWN, size: float = 22, color=MUTED,
                buff: float = 0.1) -> VGroup:
    br = Brace(mobj, direction, buff=buff, color=color)
    lab = T(text, size=size, color=color)
    lab.next_to(br, direction, buff=0.14)
    return VGroup(br, lab)


def cross_out(mobj, color=WARN_C, stroke_width: float = 5) -> VGroup:
    """Крест поверх объекта — «эта проблема снята»."""
    a = Line(mobj.get_corner(DL), mobj.get_corner(UR), color=color, stroke_width=stroke_width)
    b = Line(mobj.get_corner(UL), mobj.get_corner(DR), color=color, stroke_width=stroke_width)
    return VGroup(a, b)
