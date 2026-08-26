"""KODES — анимации для доклада.

Каждый класс ниже — одна сцена manim и одновременно группа слайдов
manim-slides (границы слайдов задаются вызовами ``self.next_slide()``).

Рендер и сборка в .pptx — ``./render.sh``.

Порядок сцен:
    S01Title          заглавие
    S02Splitting      операторное расщепление: откуда берутся ОДУ
    S03Independence   почему системы независимы
    S04CPU            как это решается на CPU
    S05GPUIdea        что именно параллелить
    S06NaiveTransfer  версия 1: перенести всё разом
    S07Batches        версия 2: батчи
    S08Chunks         версия 3: батч + куски по числу резидентных потоков
    S09Warp           варп: 32 потока в ногу
    S10Balancer       версия 4: балансировка батча
    S11Summary        итог
"""

from __future__ import annotations

import numpy as np
from manim import *
from manim_slides import Slide

from theme import *  # noqa: F403 — палитра и хелперы


class KodesSlide(Slide):
    """Общая база всех сцен.

    ``skip_reversing`` выключает генерацию «обратных» роликов, которыми
    manim-slides даёт листать презентацию назад. В .pptx они всё равно не
    попадают, а на длинных сценах их сборка в пуле процессов встаёт намертво.
    """

    skip_reversing = True


# ── Числа на слайдах ──────────────────────────────────────────────────────
#
# Со сцены называются только те, которые можно вывести или проверить:
#
#   GRI-Mech 3.0     53 вида, 325 реакций  ⇒  размер системы n = 53
#                    (состояние ячейки: температура и 52 массовые доли)
#   рабочие массивы  якобиан и матрица LU, обе n×n, double:
#                    2 · 53² · 8 Б = 44 944 Б ≈ 44 КБ на одну систему
#   RTX 5060 Ti      36 SM, 4608 ядер CUDA, 16 ГБ GDDR7 — на ней все замеры
#   потолок версии 1 16 ГБ / 44 КБ ≈ 380 тыс. систем
#   варп             32 потока, один счётчик команд на всех
#
# Всё остальное на слайдах — качественно, без придуманных величин.

# Замеры, которых у меня нет. Пока None — слайд говорит то же самое словами;
# поставьте число, и он назовёт его вместе с условиями из соседней строки.
GPU_VS_CPU_SPEEDUP: float | None = None
GPU_VS_CPU_NOTE = "GRI-Mech 3.0, RTX 5060 Ti против ..."

BALANCER_SPEEDUP: float | None = None
BALANCER_NOTE = "GRI-Mech 3.0, RTX 5060 Ti, балансировка по жёсткости"

# Карта, на которой всё считалось
GPU_NAME = "NVIDIA RTX 5060 Ti"
GPU_SPEC = "36 SM · 4608 ядер CUDA · 16 ГБ GDDR7"
GPU_VRAM_GB = 16


# ══════════════════════════════════════════════════════════════════════════
class S01Title(KodesSlide):
    """Заглавие — один статический слайд, без анимаций."""

    def construct(self):
        name = T("KODES", size=104, color=GPU_C, weight=BOLD)
        full = T("Kinetic Ordinary Differential Equations Solver", size=27, color=MUTED)
        sub = T("Решение больших ансамблей жёстких ОДУ на GPU", size=33)
        rule = Line(LEFT * 4.8, RIGHT * 4.8, stroke_width=2, color=MUTED).set_opacity(0.6)

        head = VGroup(name, full, sub).arrange(DOWN, buff=0.26)

        author = T("Павлов Егор", size=30, weight=BOLD)
        advisor = T("Научный руководитель: Епихин Андрей", size=25, color=MUTED)
        institute = T("Институт системного программирования\nим. В. П. Иванникова РАН",
                      size=25, color=MUTED, line_spacing=0.7)
        date = T("6 сентября 2026", size=24, color=MUTED)

        who = VGroup(author, advisor, institute, date).arrange(DOWN, buff=0.28)

        page = VGroup(head, rule, who).arrange(DOWN, buff=0.55)
        fit(page).move_to(ORIGIN)

        self.add(page)
        self.wait(2)


# ══════════════════════════════════════════════════════════════════════════
class S02Splitting(KodesSlide):
    """Откуда в CFD берутся независимые системы ОДУ."""

    def construct(self):
        head = title("Операторное расщепление")
        self.play(FadeIn(head, shift=DOWN * 0.2))

        # ── уравнение переноса реагирующей смеси ──────────────────────────
        lhs = MathTex(r"\frac{\partial (\rho Y_i)}{\partial t}", color=FG)
        eq = MathTex("=", color=FG)
        transport = MathTex(
            r"-\,\nabla\!\cdot\!(\rho \mathbf{u} Y_i) + \nabla\!\cdot\!(\rho D_i \nabla Y_i)",
            color=CPU_C,
        )
        plus = MathTex("+", color=FG)
        chem = MathTex(r"\dot{\omega}_i(\mathbf{Y}, T)", color=WARN_C)

        row = VGroup(lhs, eq, transport, plus, chem).arrange(RIGHT, buff=0.22)
        row.scale(1.05).move_to(UP * 1.55)

        self.play(Write(row), run_time=2.0)

        br_t = brace_label(transport, "перенос: связывает соседние ячейки", color=CPU_C)
        br_c = brace_label(chem, "химия: локальна", color=WARN_C)
        self.play(FadeIn(br_t), FadeIn(br_c))
        self.wait(0.2)
        self.next_slide()

        # ── масштабы времени ─────────────────────────────────────────────
        axis = NumberLine(
            x_range=[-10, -2, 1],
            length=10.4,
            color=MUTED,
            stroke_width=2.5,
            include_ticks=True,
            tick_size=0.09,
        ).move_to(DOWN * 1.1)

        ticks = VGroup()
        for p in range(-10, -1, 2):
            lab = MathTex(rf"10^{{{p}}}", color=MUTED).scale(0.62)
            lab.next_to(axis.n2p(p), DOWN, buff=0.22)
            ticks.add(lab)
        unit = T("характерное время, с", size=21, color=MUTED)
        unit.next_to(axis, DOWN, buff=0.72)

        def band(a, b, color, text):
            """Полоса над осью; подпись — внутри полосы, чтобы ничего не пересекалось."""
            x0, x1 = axis.n2p(a)[0], axis.n2p(b)[0]
            rect = Rectangle(
                width=x1 - x0, height=0.52,
                stroke_width=0, fill_color=color, fill_opacity=0.34,
            )
            rect.move_to([(x0 + x1) / 2, axis.get_y() + 0.46, 0])
            lab = T(text, size=22, color=FG).move_to(rect)
            return VGroup(rect, lab)

        chem_band = band(-10, -6, WARN_C, "химия")
        flow_band = band(-5.2, -2.4, CPU_C, "гидродинамика")

        self.play(Create(axis), FadeIn(ticks), FadeIn(unit), run_time=1.2)
        self.play(FadeIn(chem_band, shift=UP * 0.2), FadeIn(flow_band, shift=UP * 0.2))

        gap = T("4–6 порядков разницы  ⇒  система жёсткая", size=26, color=MEM_C)
        gap.next_to(unit, DOWN, buff=0.4)
        self.play(FadeIn(gap, shift=UP * 0.15))
        self.wait(0.2)
        self.next_slide()

        # ── вывод: расщепляем ────────────────────────────────────────────
        self.play(
            FadeOut(VGroup(axis, ticks, unit, chem_band, flow_band, gap)),
            VGroup(row, br_t, br_c).animate.move_to(UP * 2.1).scale(0.85),
            run_time=1.0,
        )

        why = bullets(
            [
                "явная схема по химии требует шага ~10⁻⁹ с",
                "неявная по всему уравнению — матрица во всю сетку",
                "выход: считать их по очереди, каждый своим методом",
            ],
            size=26,
        ).move_to(DOWN * 0.6)
        self.play(LaggedStart(*[FadeIn(b, shift=RIGHT * 0.25) for b in why],
                              lag_ratio=0.4), run_time=2.2)
        self.wait(0.2)
        self.next_slide()

        # ── как это устроено в reactingFoam ──────────────────────────────
        # Порядок взят из исходников OpenFOAM v2412:
        #   reactingFoam.C  — rhoEqn, затем цикл PIMPLE: UEqn, YEqn, EEqn, pEqn
        #   YEqn.H          — reaction->correct(), затем перенос Y_i с
        #                     источником reaction->R(Y_i)
        #   laminar.C       — correct() вызывает chemistryPtr_->solve(deltaT)
        #   StandardChemistryModel.C — RR = (c - c0)*W/deltaT
        self.play(FadeOut(why), FadeOut(VGroup(row, br_t, br_c)), run_time=0.7)

        sub = title("...как это сделано в reactingFoam")
        sub.set_color(MUTED).scale(0.62).next_to(head, DOWN, buff=0.18)
        self.play(FadeIn(sub), run_time=0.4)

        def line_box(text, color, w=5.9, h=0.56, size=22, weight=NORMAL):
            box = RoundedRectangle(width=w, height=h, corner_radius=0.1,
                                   stroke_color=color, stroke_width=2.4,
                                   fill_color=color, fill_opacity=0.10)
            lab = T(text, size=size, color=color, weight=weight)
            fit(lab, w - 0.3).move_to(box)
            return VGroup(box, lab)

        rho_box = line_box("rhoEqn — неразрывность", CPU_C)
        u_box = line_box("UEqn — импульс", CPU_C)
        chem_box = line_box("reaction->correct() — химия", WARN_C, weight=BOLD)
        y_box = line_box("перенос Yᵢ с источником reaction->R(Yᵢ)", CPU_C)
        e_box = line_box("EEqn — энергия", CPU_C)
        p_box = line_box("pEqn — давление", CPU_C)

        yeqn = VGroup(chem_box, y_box).arrange(DOWN, buff=0.10)
        stack = VGroup(rho_box, u_box, yeqn, e_box, p_box).arrange(DOWN, buff=0.18)
        stack.move_to([-2.3, -0.75, 0])          # столбец сдвинут влево

        # слева — скобка YEqn, справа — скобка PIMPLE: ничего не пересекается
        yeqn_br = Brace(yeqn, LEFT, buff=0.1, color=MUTED)
        yeqn_tag = T("YEqn", size=20, color=MUTED).next_to(yeqn_br, LEFT, buff=0.12)

        pimple_br = Brace(VGroup(u_box, yeqn, e_box, p_box), RIGHT, buff=0.25, color=MUTED)
        pimple_lab = T("цикл\nPIMPLE", size=20, color=MUTED, line_spacing=0.7)
        pimple_lab.next_to(pimple_br, RIGHT, buff=0.16)

        dt_lab = T("один шаг по времени Δt", size=22, color=MUTED)
        dt_lab.next_to(stack, UP, buff=0.28)

        self.play(FadeIn(dt_lab), run_time=0.3)
        self.play(LaggedStart(FadeIn(rho_box, shift=RIGHT * 0.2),
                              FadeIn(u_box, shift=RIGHT * 0.2),
                              FadeIn(VGroup(yeqn_br, yeqn_tag)),
                              FadeIn(chem_box, shift=RIGHT * 0.2),
                              FadeIn(y_box, shift=RIGHT * 0.2),
                              FadeIn(e_box, shift=RIGHT * 0.2),
                              FadeIn(p_box, shift=RIGHT * 0.2),
                              lag_ratio=0.5), run_time=2.6)
        self.play(FadeIn(pimple_br), FadeIn(pimple_lab), run_time=0.5)

        focus = SurroundingRectangle(chem_box, color=WARN_C, buff=0.09,
                                     stroke_width=3.5, corner_radius=0.12)

        # правая колонка: чем именно химия отдаёт результат в перенос
        rr = MathTex(r"RR = \frac{(c - c_0)\,W}{\Delta t}", color=CPU_C).scale(0.66)
        rr_note = T("химия интегрируется целиком,\n"
                    "в перенос уходит средняя\nпо шагу скорость",
                    size=18, color=MUTED, line_spacing=0.75)
        rr_group = VGroup(rr, rr_note).arrange(DOWN, buff=0.24)
        rr_group.move_to([4.6, -1.2, 0])

        note = T("шаг химии: в каждой ячейке — своя задача Коши", size=25, color=WARN_C)
        note.move_to([0, -3.45, 0])

        self.play(Create(focus), FadeIn(note, shift=UP * 0.2), run_time=0.8)
        self.play(Write(rr), FadeIn(rr_note), run_time=1.0)
        self.wait(0.3)


# ══════════════════════════════════════════════════════════════════════════
class S03Independence(KodesSlide):
    """Ячейки в шаге химии ничем не обмениваются."""

    def construct(self):
        head = title("Шаг химии: N независимых систем ОДУ")
        self.play(FadeIn(head, shift=DOWN * 0.2))

        grid = mesh(rows=6, cols=6, cell=0.66, gap=0.08).move_to(LEFT * 3.4 + DOWN * 0.35)
        self.play(LaggedStart(*[FadeIn(c, scale=0.7) for c in grid],
                              lag_ratio=0.012), run_time=1.4)

        # ── перенос: ячейки связаны ──────────────────────────────────────
        links = VGroup()
        rows, cols = grid.rows, grid.cols
        for r in range(rows):
            for c in range(cols):
                i = r * cols + c
                if c + 1 < cols:
                    links.add(Line(grid[i].get_right(), grid[i + 1].get_left(),
                                   color=CPU_C, stroke_width=2.4))
                if r + 1 < rows:
                    links.add(Line(grid[i].get_bottom(), grid[i + cols].get_top(),
                                   color=CPU_C, stroke_width=2.4))

        lab_transport = T("перенос: потоки через грани", size=25, color=CPU_C)
        lab_transport.next_to(grid, DOWN, buff=0.55)
        self.play(LaggedStart(*[Create(l) for l in links], lag_ratio=0.008),
                  FadeIn(lab_transport), run_time=1.6)
        self.wait(0.2)
        self.next_slide()

        # ── химия: связи исчезают ────────────────────────────────────────
        lab_chem = T("химия: связей нет", size=25, color=WARN_C)
        lab_chem.move_to(lab_transport)
        self.play(
            LaggedStart(*[Uncreate(l) for l in links], lag_ratio=0.006),
            FadeOut(lab_transport, shift=DOWN * 0.2),
            FadeIn(lab_chem, shift=DOWN * 0.2),
            run_time=1.4,
        )
        self.play(LaggedStart(*[c.animate.set_stroke(WARN_C).set_fill(WARN_C, 0.20)
                                for c in grid], lag_ratio=0.012), run_time=1.0)
        self.wait(0.2)
        self.next_slide()

        # ── что именно решается в одной ячейке ───────────────────────────
        picked = grid[14]
        halo = SurroundingRectangle(picked, color=MEM_C, buff=0.06, stroke_width=3)

        card = RoundedRectangle(width=6.6, height=5.2, corner_radius=0.16,
                                stroke_color=MEM_C, stroke_width=2.6,
                                fill_color=MEM_C, fill_opacity=0.05)
        card.move_to(RIGHT * 3.5 + DOWN * 0.25)

        sys_eq = MathTex(
            r"\frac{d\mathbf{y}}{dt} = \mathbf{f}(\mathbf{y}, p),\qquad t \in [0,\,\Delta t]",
            color=FG,
        ).scale(0.76)
        state = MathTex(
            r"\mathbf{y} = \big[\,T,\; Y_1, \dots, Y_{n-1}\,\big]",
            color=FG,
        ).scale(0.76)
        param = T("p — давление, постоянно на шаге", size=20, color=MUTED)
        curve = stiff_curve(color=WARN_C, w=3.4, h=1.0)

        # Одно обозначение на всю презентацию: n — размер системы.
        n_def = T("n — размер системы:\nсколько уравнений в одной ячейке",
                  size=20, color=MEM_C, line_spacing=0.75)
        size_note = T("GRI-Mech 3.0: 53 вида  ⇒  n = 53", size=20, color=MEM_C)

        inner = VGroup(sys_eq, state, curve, param, n_def, size_note)
        inner.arrange(DOWN, buff=0.26)
        # вписываем содержимое в карточку и по ширине, и по высоте
        fit(inner, card.width - 0.7)
        if inner.height > card.height - 0.6:
            inner.scale((card.height - 0.6) / inner.height)
        inner.move_to(card)

        link = Arrow(halo.get_right(), card.get_left(), buff=0.2,
                     color=MEM_C, stroke_width=3)

        self.play(Create(halo), GrowArrow(link), Create(card), run_time=1.0)
        self.play(Write(sys_eq), run_time=1.0)
        self.play(FadeIn(state), run_time=0.6)
        self.play(Create(curve), run_time=1.0)
        self.play(FadeIn(param), run_time=0.4)
        self.play(FadeIn(n_def), FadeIn(size_note), run_time=0.7)
        self.wait(0.2)
        self.next_slide()

        punch = T("Сколько ячеек в сетке — столько независимых задач.\n"
                  "Ни одна не ждёт другую.",
                  size=27, color=OK_C, line_spacing=0.8)
        punch.to_edge(DOWN, buff=0.35)
        self.play(FadeOut(lab_chem), FadeIn(punch, shift=UP * 0.2))
        self.wait(0.3)


# ══════════════════════════════════════════════════════════════════════════
class S04CPU(KodesSlide):
    """Классическая схема: ячейка за ячейкой на CPU."""

    def construct(self):
        head = title("Как это считается на CPU")
        self.play(FadeIn(head, shift=DOWN * 0.2))

        rows, cols = 6, 6
        grid = mesh(rows=rows, cols=cols, cell=0.6, gap=0.07, color=WARN_C, fill=0.10)
        grid.move_to(LEFT * 4.0 + DOWN * 0.3)

        cpu = cpu_chip("CPU")
        cpu.move_to(RIGHT * 3.2 + DOWN * 0.3)

        done = counter("решено:", 0, color=OK_C)
        done.next_to(grid, DOWN, buff=0.6)

        self.play(FadeIn(grid), FadeIn(cpu), FadeIn(done), run_time=1.0)
        self.wait(0.2)
        self.next_slide()

        core = cpu.cores[0]
        work_slot = cpu.body.get_center()

        # ── одна ячейка «в подробностях» ─────────────────────────────────
        first = grid[0]
        travel = first.copy().set_fill(WARN_C, 0.42)
        self.add(travel)
        self.play(travel.animate.move_to(work_slot).scale(2.4), run_time=0.8)

        curve = stiff_curve(color=WARN_C, w=1.5, h=0.7).move_to(work_slot)
        steps_lab = T("шаг подбирается по жёсткости:\nчем резче горит, тем мельче",
                      size=20, color=MUTED, line_spacing=0.75)
        steps_lab.next_to(cpu.body, DOWN, buff=0.3)
        self.play(Create(curve), FadeIn(steps_lab), run_time=1.2)
        self.play(FadeOut(curve), travel.animate.move_to(first).scale(1 / 2.4)
                  .set_fill(OK_C, 0.60).set_stroke(OK_C), run_time=0.7)
        self.remove(travel)
        self.play(first.animate.set_fill(OK_C, 0.60).set_stroke(OK_C),
                  done.value.animate.set_value(1), run_time=0.3)
        self.wait(0.2)
        self.next_slide()

        # ── ещё две, чтобы стало видно, что это очередь ──────────────────
        for k in (1, 2):
            cell = grid[k]
            trav = cell.copy().set_fill(WARN_C, 0.42)
            self.add(trav)
            self.play(trav.animate.move_to(work_slot).scale(2.4), run_time=0.35)
            self.play(Indicate(trav, color=MEM_C, scale_factor=1.1), run_time=0.3)
            self.play(trav.animate.move_to(cell).scale(1 / 2.4)
                      .set_fill(OK_C, 0.60).set_stroke(OK_C), run_time=0.35)
            self.remove(trav)
            self.play(cell.animate.set_fill(OK_C, 0.60).set_stroke(OK_C),
                      done.value.animate.set_value(k + 1), run_time=0.15)

        queue_note = T("...и так по очереди", size=24, color=MUTED)
        queue_note.next_to(steps_lab, DOWN, buff=0.35)
        self.play(FadeIn(queue_note))
        self.wait(0.2)
        self.next_slide()

        # ── остальные — быстро, по 4 (число ядер) ────────────────────────
        self.play(FadeOut(queue_note), FadeOut(steps_lab), run_time=0.4)
        omp = T("OpenMP: 4 ядра — 4 ячейки одновременно", size=21, color=CPU_C)
        omp.next_to(cpu.body, DOWN, buff=0.3)
        self.play(FadeIn(omp))

        remaining = list(range(3, rows * cols))
        for start in range(0, len(remaining), 4):
            chunk = remaining[start:start + 4]
            movers = []
            for slot, idx in enumerate(chunk):
                cell = grid[idx]
                trav = cell.copy().set_fill(WARN_C, 0.42)
                self.add(trav)
                movers.append((trav, cell, cpu.cores[slot % len(cpu.cores)]))
            self.play(*[t.animate.move_to(c.get_center()).scale(0.9) for t, _, c in movers],
                      run_time=0.16)
            self.play(*[t.animate.move_to(cell).scale(1 / 0.9)
                        .set_fill(OK_C, 0.60).set_stroke(OK_C) for t, cell, _ in movers],
                      run_time=0.16)
            for t, cell, _ in movers:
                self.remove(t)
                cell.set_fill(OK_C, 0.60).set_stroke(OK_C)
            done.value.set_value(chunk[-1] + 1)

        self.wait(0.2)
        self.next_slide()

        verdict = bullets(
            [
                "ячеек столько, сколько в сетке — и каждая идёт отдельно",
                "каждая требует своего числа неявных шагов",
                "ядер — единицы или десятки; очередь длиннее на порядки",
            ],
            size=28,
            color=FG,
        ).move_to(ORIGIN)
        self.play(FadeOut(VGroup(omp, grid, cpu, done)), run_time=0.6)
        self.play(LaggedStart(*[FadeIn(b, shift=RIGHT * 0.25) for b in verdict],
                              lag_ratio=0.35), run_time=1.8)
        self.wait(0.3)


# ══════════════════════════════════════════════════════════════════════════
class S05GPUIdea(KodesSlide):
    """Что именно параллелить: внутри системы или по системам."""

    def construct(self):
        head = title("Что параллелить")
        self.play(FadeIn(head, shift=DOWN * 0.2))

        # ── слева: классический подход ───────────────────────────────────
        left_title = T("Классический подход\n(SUNDIALS, Cantera)", size=26, color=CPU_C)
        left_title.move_to(LEFT * 3.6 + UP * 2.25)

        vec = strip(12, width=4.6, height=0.62, color=CPU_C, fill=0.25)
        vec.move_to(LEFT * 3.6 + UP * 0.35)
        vec_lab = brace_label(vec, "вектор состояния одной ячейки, n чисел",
                              UP, size=20, color=MUTED)

        arrows_l = VGroup(*[
            Arrow(vec[i].get_bottom(), vec[i].get_bottom() + DOWN * 0.7,
                  buff=0.05, color=CPU_C, stroke_width=2.5,
                  max_tip_length_to_length_ratio=0.3)
            for i in range(0, 12, 1)
        ])
        threads_l = T("потоки делят компоненты", size=21, color=MUTED)
        threads_l.next_to(arrows_l, DOWN, buff=0.2)

        left_note = bullets(
            ["n задан механизмом, а не задачей",
             "для GRI-Mech n = 53 — делить почти нечего"],
            size=21,
        )
        left_note.move_to([-3.6, -2.3, 0])

        self.play(FadeIn(left_title), FadeIn(vec), FadeIn(vec_lab), run_time=0.9)
        self.play(LaggedStart(*[GrowArrow(a) for a in arrows_l], lag_ratio=0.05),
                  FadeIn(threads_l), run_time=1.0)
        self.play(FadeIn(left_note, shift=UP * 0.2), run_time=0.7)
        self.wait(0.2)
        self.next_slide()

        divider = DashedLine(UP * 2.9, DOWN * 3.2, color=MUTED, stroke_width=2).set_opacity(0.55)
        self.play(Create(divider), run_time=0.5)

        # ── справа: KODES ────────────────────────────────────────────────
        right_title = T("KODES", size=26, color=GPU_C, weight=BOLD)
        right_title.move_to(RIGHT * 3.6 + UP * 2.25)

        systems = VGroup()
        for k in range(6):
            s = strip(12, width=3.9, height=0.26, color=GPU_C, fill=0.28)
            systems.add(s)
        systems.arrange(DOWN, buff=0.16).move_to(RIGHT * 3.6 + UP * 0.3)
        dots = T("⋮", size=30, color=MUTED).next_to(systems, DOWN, buff=0.1)

        arrows_r = VGroup(*[
            Arrow(s.get_left() + LEFT * 0.55, s.get_left(), buff=0.06,
                  color=GPU_C, stroke_width=2.5, max_tip_length_to_length_ratio=0.35)
            for s in systems
        ])
        threads_r = T("один поток — вся система целиком", size=21, color=MUTED)
        threads_r.next_to(dots, DOWN, buff=0.3)

        right_note = bullets(
            ["скорость растёт с числом ячеек,",
             "а ячеек в CFD только прибавляется"],
            size=21,
            color=OK_C,
        )
        right_note.move_to([3.6, -2.3, 0])

        self.play(FadeIn(right_title), run_time=0.4)
        self.play(LaggedStart(*[FadeIn(s, shift=LEFT * 0.2) for s in systems],
                              lag_ratio=0.12), FadeIn(dots), run_time=1.2)
        self.play(LaggedStart(*[GrowArrow(a) for a in arrows_r], lag_ratio=0.1),
                  FadeIn(threads_r), run_time=1.0)
        self.play(FadeIn(right_note, shift=UP * 0.2), run_time=0.7)
        self.wait(0.2)
        self.next_slide()

        punch = T("Пропускная способность — от числа систем, а не от размера одной",
                  size=22, color=OK_C)
        punch.move_to(DOWN * 3.45)
        box = SurroundingRectangle(punch, color=OK_C, buff=0.22, stroke_width=2,
                                   corner_radius=0.12)
        self.play(FadeIn(punch), Create(box))
        self.wait(0.2)
        self.next_slide()

        # ── на чём считалось ─────────────────────────────────────────────
        self.play(
            FadeOut(VGroup(left_title, vec, vec_lab, arrows_l, threads_l, left_note,
                           divider, right_title, systems, dots, arrows_r, threads_r,
                           right_note, punch, box)),
            run_time=0.7,
        )

        card_title = T(GPU_NAME, size=40, color=GPU_C, weight=BOLD)
        card_spec = T(GPU_SPEC, size=26, color=FG)
        card_note = T("все замеры в докладе сделаны на ней", size=23, color=MUTED)
        card_inner = VGroup(card_title, card_spec, card_note).arrange(DOWN, buff=0.34)

        card_box = RoundedRectangle(
            width=card_inner.width + 1.6, height=card_inner.height + 1.2,
            corner_radius=0.18, stroke_color=GPU_C, stroke_width=3,
            fill_color=GPU_C, fill_opacity=0.05,
        ).move_to(card_inner)

        self.play(Create(card_box), FadeIn(card_inner, shift=UP * 0.2), run_time=1.2)
        self.wait(0.3)


# ══════════════════════════════════════════════════════════════════════════
class S06NaiveTransfer(KodesSlide):
    """Версия 1: скопировать весь ансамбль разом."""

    def construct(self):
        head = title("Версия 1: перенести весь ансамбль разом")
        self.play(FadeIn(head, shift=DOWN * 0.2))

        # ── сетка выпрямляется в линейный массив ─────────────────────────
        rows, cols = 8, 8
        grid = mesh(rows=rows, cols=cols, cell=0.42, gap=0.05, color=CPU_C, fill=0.22)
        grid.move_to(UP * 0.55)
        self.play(LaggedStart(*[FadeIn(c, scale=0.6) for c in grid],
                              lag_ratio=0.006), run_time=1.2)

        line_target = strip(rows * cols, width=11.0, height=0.5, color=CPU_C, fill=0.22)
        line_target.move_to(UP * 1.95)

        flat_lab = T("линейный массив состояний, по одному на ячейку", size=23, color=MUTED)
        flat_lab.next_to(line_target, UP, buff=0.28)

        self.play(
            LaggedStart(*[Transform(grid[i], line_target[i]) for i in range(rows * cols)],
                        lag_ratio=0.006),
            run_time=2.2,
        )
        self.play(FadeIn(flat_lab), run_time=0.5)
        self.wait(0.2)
        self.next_slide()

        # ── переезд в VRAM ───────────────────────────────────────────────
        # рамка пониже и повыше по кадру: под ней должно остаться место для
        # подписи и текста ошибки, иначе они упираются в нижний край
        vram = vram_frame(f"{GPU_NAME} · VRAM  {GPU_VRAM_GB} ГБ", w=11.0, h=3.4)
        vram.move_to(DOWN * 1.2)
        self.play(Create(vram.box), FadeIn(vram.name), run_time=0.8)

        # стрелку уводим вправо от подписи рамки, иначе она через неё проходит
        h2d_x = vram.box.get_right()[0] - 2.4
        h2d = Arrow([h2d_x, line_target.get_bottom()[1] - 0.05, 0],
                    [h2d_x, vram.box.get_top()[1] + 0.05, 0],
                    buff=0.05, color=MEM_C, stroke_width=4)
        h2d_lab = T("cudaMemcpy H2D", size=21, color=MEM_C).next_to(h2d, RIGHT, buff=0.2)
        self.play(GrowArrow(h2d), FadeIn(h2d_lab), run_time=0.6)

        left_x = vram.box.get_left()[0] + 0.35
        state_w = 8.0
        state = Rectangle(width=state_w, height=0.42, stroke_color=CPU_C, stroke_width=1.6,
                          fill_color=CPU_C, fill_opacity=0.72)
        state.move_to([left_x + state_w / 2, vram.box.get_y() + 0.85, 0])
        state_lab = T("состояния:  n чисел на ячейку", size=21, color=CPU_C)
        state_lab.next_to(state, UP, buff=0.16).align_to(state, LEFT)

        self.play(TransformFromCopy(grid, state), FadeIn(state_lab), run_time=1.2)
        self.wait(0.2)
        self.next_slide()

        # ── и рабочие массивы под ним ────────────────────────────────────
        scratch_w = 8.0
        scratch = Rectangle(width=scratch_w, height=1.15, stroke_color=SCRATCH_C,
                            stroke_width=1.6, fill_color=SCRATCH_C, fill_opacity=0.68)
        scratch.move_to([left_x + scratch_w / 2, vram.box.get_y() - 0.55, 0])
        scratch_lab = T("рабочие массивы на каждую ячейку:", size=21, color=SCRATCH_C)
        scratch_lab.next_to(scratch, UP, buff=0.14).align_to(scratch, LEFT)
        scratch_what = T("якобиан и матрица LU — обе n × n, плюс векторы метода",
                         size=19, color=MUTED)
        scratch_what.next_to(scratch, DOWN, buff=0.16).align_to(scratch, LEFT)

        self.play(FadeIn(scratch_lab), GrowFromEdge(scratch, LEFT), run_time=1.0)
        self.play(FadeIn(scratch_what), run_time=0.5)

        cost = tag("для GRI-Mech:  2 · 53² · 8 Б ≈ 44 КБ на ячейку", color=SCRATCH_C)
        cost.next_to(vram.box, DOWN, buff=0.28)
        self.play(FadeIn(cost, shift=UP * 0.2))
        self.wait(0.2)
        self.next_slide()

        # ── задача растёт — массивы вылезают за границу памяти ───────────
        grow_lab = T("ячеек становится больше", size=24, color=MEM_C)
        grow_lab.move_to(cost)
        self.play(FadeOut(cost), FadeIn(grow_lab), run_time=0.4)

        for w_state, w_scratch, rt in ((9.3, 10.4, 0.8), (10.2, 11.7, 1.0)):
            self.play(
                Transform(state, resized(state, w_state, left_x)),
                Transform(scratch, resized(scratch, w_scratch, left_x)),
                run_time=rt,
            )

        # перерасход: то, что за правой границей рамки
        border_x = vram.box.get_right()[0]
        over_w = scratch.get_right()[0] - border_x
        overflow = Rectangle(width=max(over_w, 0.1), height=scratch.height,
                             stroke_width=0, fill_color=WARN_C, fill_opacity=0.85)
        overflow.move_to([border_x + max(over_w, 0.1) / 2, scratch.get_y(), 0])

        self.play(FadeIn(overflow), vram.box.animate.set_stroke(WARN_C, 4), run_time=0.6)
        self.play(Flash(vram.box.get_right(), color=WARN_C, line_length=0.4,
                        num_lines=14, flash_radius=0.5), run_time=0.8)

        # ошибка встаёт ровно туда, где была подпись, — так она гарантированно
        # не наедет ни на рамку сверху, ни на край кадра снизу
        err = T("cudaErrorMemoryAllocation:  out of memory", size=27, color=WARN_C)
        fit(err).move_to(grow_lab)
        self.play(FadeOut(grow_lab), FadeIn(err, shift=UP * 0.2))
        self.wait(0.2)
        self.next_slide()

        self.play(
            FadeOut(VGroup(grid, flat_lab, h2d, h2d_lab, state, state_lab, scratch,
                           scratch_lab, scratch_what, overflow, err, vram)),
            run_time=0.8,
        )
        problem = bullets(
            [
                "рабочая память растёт как «число ячеек × n²»",
                "GRI-Mech на 16 ГБ:  16 ГБ / 44 КБ ≈ 380 тыс. ячеек — и всё",
                "сетка крупнее — расчёт просто не стартует",
            ],
            size=27,
            color=WARN_C,
        ).move_to(UP * 0.2)

        detail = T("причём это на пустой карте: под состояния и под саму\n"
                   "программу тоже нужна память",
                   size=22, color=MUTED, line_spacing=0.8)
        detail.next_to(problem, DOWN, buff=0.7)

        self.play(LaggedStart(*[FadeIn(b, shift=RIGHT * 0.25) for b in problem],
                              lag_ratio=0.35), run_time=1.8)
        self.play(FadeIn(detail), run_time=0.6)
        self.wait(0.3)


# ══════════════════════════════════════════════════════════════════════════
class S07Batches(KodesSlide):
    """Версия 2: батчи."""

    def construct(self):
        head = title("Версия 2: батчами")
        self.play(FadeIn(head, shift=DOWN * 0.2))

        # ── ансамбль в памяти хоста, разрезанный на батчи ────────────────
        host = Rectangle(width=12.4, height=1.2, stroke_color=CPU_C, stroke_width=2.4,
                         fill_color=CPU_C, fill_opacity=0.05).move_to(UP * 1.6)
        host_lab = T("оперативная память хоста: вся сетка целиком",
                     size=21, color=CPU_C)
        host_lab.next_to(host, UP, buff=0.16)

        n_batches = 6
        batches = VGroup()
        bw = (12.4 - 0.3) / n_batches
        for k in range(n_batches):
            b = Rectangle(width=bw - 0.08, height=0.85, stroke_color=CPU_C,
                          stroke_width=1.6, fill_color=CPU_C, fill_opacity=0.45)
            b.move_to([host.get_left()[0] + 0.15 + bw * (k + 0.5), host.get_y(), 0])
            batches.add(b)

        self.play(Create(host), FadeIn(host_lab), run_time=0.7)
        self.play(LaggedStart(*[FadeIn(b, scale=0.9) for b in batches],
                              lag_ratio=0.12), run_time=1.2)
        # подпись короткая: батч крайний слева, длинная строка уходит за кадр
        bs_brace = brace_label(batches[0], "батч", DOWN, size=19, color=MUTED)
        self.play(FadeIn(bs_brace), run_time=0.5)
        self.wait(0.2)
        self.next_slide()

        # ── VRAM и конвейер ──────────────────────────────────────────────
        vram = vram_frame("GPU · VRAM", w=8.6, h=3.3).move_to(LEFT * 1.9 + DOWN * 1.5)
        self.play(Create(vram.box), FadeIn(vram.name), run_time=0.6)

        memcpy = counter("передач\ncudaMemcpy:", 0, color=MEM_C, size=23)
        memcpy.next_to(vram.box, RIGHT, buff=0.6).shift(UP * 0.9)
        self.play(FadeIn(memcpy), run_time=0.4)

        left_x = vram.box.get_left()[0] + 0.4
        inner_w = 7.8

        state = Rectangle(width=inner_w, height=0.4, stroke_color=CPU_C, stroke_width=1.6,
                          fill_color=CPU_C, fill_opacity=0.72)
        state.move_to([left_x + inner_w / 2, vram.box.get_y() + 0.95, 0])
        state_lab = T("состояния батча", size=20, color=CPU_C)
        state_lab.next_to(state, UP, buff=0.12).align_to(state, LEFT)

        scratch = Rectangle(width=inner_w, height=1.25, stroke_color=SCRATCH_C,
                            stroke_width=1.6, fill_color=SCRATCH_C, fill_opacity=0.68)
        scratch.move_to([left_x + inner_w / 2, vram.box.get_y() - 0.55, 0])
        scratch_lab = T("рабочие массивы: по комплекту на систему батча",
                        size=19, color=SCRATCH_C)
        scratch_lab.next_to(scratch, DOWN, buff=0.14).align_to(scratch, LEFT)

        # первый батч едет вниз
        travel = batches[0].copy()
        self.add(travel)
        self.play(travel.animate.move_to(state).stretch_to_fit_width(inner_w)
                  .stretch_to_fit_height(0.4), run_time=0.9)
        self.remove(travel)
        self.add(state)
        self.play(FadeIn(state_lab), memcpy.value.animate.set_value(1), run_time=0.4)
        self.play(GrowFromEdge(scratch, LEFT), FadeIn(scratch_lab), run_time=0.8)

        self.play(batches[0].animate.set_fill(OK_C, 0.55).set_stroke(OK_C), run_time=0.3)

        # остальные — быстро
        for k in range(1, n_batches):
            t = batches[k].copy()
            self.add(t)
            self.play(t.animate.move_to(state).stretch_to_fit_width(inner_w)
                      .stretch_to_fit_height(0.4).set_opacity(0.0), run_time=0.28)
            self.remove(t)
            self.play(Indicate(state, color=MEM_C, scale_factor=1.02),
                      batches[k].animate.set_fill(OK_C, 0.55).set_stroke(OK_C),
                      memcpy.value.animate.set_value(k + 1), run_time=0.28)

        fits = tag("влезает: батч подбирается под свободную VRAM", color=OK_C)
        fits.next_to(vram.box, DOWN, buff=0.35)
        self.play(FadeIn(fits, shift=UP * 0.2))
        self.wait(0.2)
        self.next_slide()

        # ── но: резидентных потоков куда меньше, чем систем в батче ──────
        self.play(FadeOut(fits), run_time=0.3)

        # доля намеренно «на глаз»: сколько потоков карта держит одновременно,
        # зависит от занятости конкретного ядра и считается в runtime
        resident_w = inner_w * 0.07
        resident = Rectangle(width=resident_w, height=1.25, stroke_width=0,
                             fill_color=OK_C, fill_opacity=0.9)
        resident.move_to([left_x + resident_w / 2, scratch.get_y(), 0])

        wasted = Rectangle(width=inner_w - resident_w, height=1.25, stroke_width=0,
                           fill_color=WARN_C, fill_opacity=0.28)
        wasted.move_to([left_x + resident_w + (inner_w - resident_w) / 2, scratch.get_y(), 0])

        res_lab = T("работают только те потоки, что карта держит разом",
                    size=19, color=OK_C)
        res_lab.next_to(vram.box, DOWN, buff=0.16).align_to(vram.box, LEFT)
        res_arrow = Arrow(res_lab.get_top() + LEFT * 1.4, resident.get_bottom(), buff=0.06,
                          color=OK_C, stroke_width=2.5,
                          max_tip_length_to_length_ratio=0.35)

        # подпись лежит на красной заливке, поэтому она не красная, а тёмная
        waste_lab = T("здесь не будет работать никто:\nстолько потоков GPU не держит",
                      size=19, color=FG)
        waste_lab.move_to(wasted)

        self.play(FadeOut(scratch_lab), FadeIn(resident), FadeIn(wasted), run_time=0.8)
        self.play(FadeIn(res_lab), GrowArrow(res_arrow), FadeIn(waste_lab), run_time=0.9)
        self.wait(0.2)
        self.next_slide()

        chain = bullets(
            [
                "бо́льшая часть рабочей памяти простаивает",
                "а её размер растёт как n² — крупнее механизм, хуже потери",
                "батч приходится мельчить — и передач становится больше",
            ],
            size=24,
            color=WARN_C,
        )
        chain.move_to(UP * 1.6)
        self.play(
            FadeOut(VGroup(waste_lab, res_lab, res_arrow, host, host_lab, batches, bs_brace)),
            FadeIn(chain, shift=UP * 0.2),
            run_time=1.2,
        )
        self.play(Indicate(memcpy, color=WARN_C, scale_factor=1.15), run_time=0.8)
        self.wait(0.3)


# ══════════════════════════════════════════════════════════════════════════
class S08Chunks(KodesSlide):
    """Версия 3: батч обходится кусками по числу резидентных потоков."""

    def construct(self):
        head = title("Версия 3: батч — данные, кусок — потоки")
        self.play(FadeIn(head, shift=DOWN * 0.2))

        vram = vram_frame("GPU · VRAM", w=12.0, h=4.6).move_to(DOWN * 0.55)
        self.play(Create(vram.box), FadeIn(vram.name), run_time=0.6)

        left_x = vram.box.get_left()[0] + 0.35
        inner_w = 11.3

        n_chunks = 12
        state_cells = VGroup()
        cw = inner_w / n_chunks
        for k in range(n_chunks):
            c = Rectangle(width=cw - 0.04, height=0.55, stroke_color=CPU_C,
                          stroke_width=1.2, fill_color=CPU_C, fill_opacity=0.45)
            c.move_to([left_x + cw * (k + 0.5), vram.box.get_y() + 1.35, 0])
            state_cells.add(c)
        state_lab = T("состояния батча — крупный, во всю свободную память",
                      size=21, color=CPU_C)
        state_lab.next_to(state_cells, UP, buff=0.18)

        self.play(LaggedStart(*[FadeIn(c, shift=DOWN * 0.15) for c in state_cells],
                              lag_ratio=0.05), FadeIn(state_lab), run_time=1.2)

        # ── рабочая память — только на резидентные потоки ────────────────
        scratch = Rectangle(width=cw - 0.04, height=1.5, stroke_color=SCRATCH_C,
                            stroke_width=1.8, fill_color=SCRATCH_C, fill_opacity=0.70)
        scratch.move_to([left_x + cw * 0.5, vram.box.get_y() - 1.0, 0])
        scratch_lab = T("рабочие массивы — только на те потоки, что работают разом",
                        size=21, color=SCRATCH_C)
        scratch_lab.next_to(scratch, DOWN, buff=0.2).align_to(vram.box, LEFT).shift(RIGHT * 0.35)

        self.play(GrowFromEdge(scratch, LEFT), FadeIn(scratch_lab), run_time=0.9)
        self.wait(0.2)
        self.next_slide()

        # ── окно едет по батчу: перенос → решение → перенос обратно ──────
        def load_arrow(cell):
            """Вниз, из батча в рабочую память."""
            return Arrow(cell.get_bottom() + LEFT * 0.09, scratch.get_top() + LEFT * 0.09,
                         buff=0.12, color=OK_C, stroke_width=3.5,
                         max_tip_length_to_length_ratio=0.16)

        def store_arrow(cell):
            """Вверх, из рабочей памяти обратно в батч."""
            return Arrow(scratch.get_top() + RIGHT * 0.09, cell.get_bottom() + RIGHT * 0.09,
                         buff=0.12, color=CPU_C, stroke_width=3.5,
                         max_tip_length_to_length_ratio=0.16)

        window = SurroundingRectangle(state_cells[0], color=OK_C, buff=0.05,
                                      stroke_width=3.5)
        load = load_arrow(state_cells[0])
        store = store_arrow(state_cells[0])

        load_lab = T("loadSystem\nперенос в рабочую память", size=19, color=OK_C,
                     line_spacing=0.7)
        store_lab = T("storeSystem\nрезультат обратно в батч", size=19, color=CPU_C,
                      line_spacing=0.7)
        # подписи ниже уровня стрелок: стрелки ходят от верха рабочей памяти
        # вверх по батчу, всё, что ниже, они не задевают
        labs = VGroup(load_lab, store_lab).arrange(DOWN, buff=0.34, aligned_edge=LEFT)
        labs.move_to([2.4, scratch.get_y(), 0])

        # первый кусок — по шагам, чтобы был виден весь цикл
        self.play(Create(window), run_time=0.4)
        self.play(GrowArrow(load), FadeIn(load_lab), run_time=0.6)
        self.play(Indicate(scratch, color=MEM_C, scale_factor=1.06), run_time=0.6)
        self.play(GrowArrow(store), FadeIn(store_lab), run_time=0.6)
        self.play(state_cells[0].animate.set_fill(OK_C, 0.6).set_stroke(OK_C), run_time=0.3)

        for k in range(1, n_chunks):
            self.play(
                window.animate.move_to(state_cells[k]),
                Transform(load, load_arrow(state_cells[k])),
                Transform(store, store_arrow(state_cells[k])),
                run_time=0.24,
            )
            self.play(
                Indicate(scratch, color=MEM_C, scale_factor=1.04),
                state_cells[k].animate.set_fill(OK_C, 0.6).set_stroke(OK_C),
                run_time=0.24,
            )

        stride = T("кусок за куском: перенёс — решил — вернул, и так весь батч",
                   size=22, color=MUTED)
        stride.next_to(vram.box, DOWN, buff=0.28)
        self.play(FadeIn(stride), run_time=0.5)
        self.wait(0.2)
        self.next_slide()

        # ── что это дало ─────────────────────────────────────────────────
        self.play(
            FadeOut(VGroup(window, load, store, labs, stride, state_lab, scratch_lab,
                           state_cells, scratch)),
            run_time=0.6,
        )

        table = VGroup(
            T("было", size=24, color=MUTED),
            T("рабочей памяти — на весь батч", size=24, color=WARN_C),
            T("стало", size=24, color=MUTED),
            T("рабочей памяти — на работающие потоки", size=24, color=OK_C),
            T("что это даёт", size=24, color=MUTED),
            T("освободившееся уходит под состояния:\n"
              "батч крупнее — передач меньше", size=24, color=OK_C, line_spacing=0.75),
        )
        table.arrange_in_grid(rows=3, cols=2, buff=(0.9, 0.6), col_alignments="rl")
        fit(table).move_to(DOWN * 0.4)
        self.play(FadeIn(table, shift=UP * 0.2), run_time=1.2)
        self.wait(0.2)
        self.next_slide()

        honest = bullets(
            [
                "объём переданных данных тот же — выигрыш умеренный",
                "но теперь потоки и данные развязаны: батч можно переупорядочить",
            ],
            size=25,
        )
        honest.to_edge(DOWN, buff=0.35)
        self.play(FadeIn(honest[0], shift=RIGHT * 0.2), run_time=0.7)
        self.play(FadeIn(honest[1], shift=RIGHT * 0.2), run_time=0.7)
        self.play(Indicate(honest[1], color=OK_C, scale_factor=1.06), run_time=0.8)
        self.wait(0.3)


# ══════════════════════════════════════════════════════════════════════════
class S09Warp(KodesSlide):
    """Варп: 32 потока идут в ногу."""

    LANES = 32
    MAX_STEPS = 400.0
    H = 3.1
    BASE_Y = -2.35

    def _profile(self, spread: bool):
        rng = np.random.default_rng(11)
        if spread:
            steps = rng.integers(45, 105, self.LANES).astype(float)
            steps[11] = 400.0
            steps[24] = 235.0
        else:
            steps = rng.integers(62, 82, self.LANES).astype(float)
        return steps

    def _bars(self, steps, tracker):
        n = self.LANES
        total_w = 10.6
        bw = total_w / n * 0.72
        gap = total_w / n
        x0 = -total_w / 2 + gap / 2

        def build(i):
            def f():
                s = steps[i]
                v = tracker.get_value()
                idle = v >= s
                h = min(v, s) / self.MAX_STEPS * self.H
                rect = Rectangle(
                    width=bw,
                    height=max(h, 0.012),
                    stroke_width=0,
                    fill_color=MUTED if idle else GPU_C,
                    fill_opacity=0.40 if idle else 0.92,
                )
                rect.move_to([x0 + i * gap, self.BASE_Y + max(h, 0.012) / 2, 0])
                return rect

            return f

        return VGroup(*[always_redraw(build(i)) for i in range(n)]), x0, gap, bw

    def construct(self):
        head = title("Варп: 32 потока с одним счётчиком команд")
        self.play(FadeIn(head, shift=DOWN * 0.2))

        explain = bullets(
            [
                "на CPU потоки независимы — каждый идёт своим путём",
                "на GPU они собраны в варпы по 32 и делают одну инструкцию",
                "варп занят, пока не закончит самый медленный из 32",
            ],
            size=27,
        ).move_to(ORIGIN)
        self.play(LaggedStart(*[FadeIn(b, shift=RIGHT * 0.25) for b in explain],
                              lag_ratio=0.35), run_time=2.0)
        self.wait(0.2)
        self.next_slide()
        self.play(FadeOut(explain), run_time=0.4)

        # ── несбалансированный варп ──────────────────────────────────────
        steps = self._profile(spread=True)
        tracker = ValueTracker(0.0)
        bars, x0, gap, bw = self._bars(steps, tracker)

        floor = Line(LEFT * 5.6, RIGHT * 5.6, color=MUTED, stroke_width=2)
        floor.move_to([0, self.BASE_Y, 0])
        lanes_lab = brace_label(floor, "32 полосы одного варпа", DOWN, size=22, color=MUTED)
        axis_lab = T("шагов интегрирования", size=21, color=MUTED)
        axis_lab.rotate(PI / 2).next_to(floor, LEFT, buff=0.35).shift(UP * 1.4)

        # счётчика шагов здесь нет намеренно: число шагов у полос — рисованное,
        # называть его со сцены было бы выдумкой
        self.play(Create(floor), FadeIn(lanes_lab), FadeIn(axis_lab), run_time=0.8)
        self.add(bars)
        self.play(tracker.animate.set_value(110), run_time=1.6, rate_func=linear)
        self.wait(0.2)
        self.next_slide()

        most_done = T("30 полос закончили — и стоят", size=24, color=MUTED)
        most_done.move_to(UP * 1.3 + LEFT * 2.2)
        self.play(FadeIn(most_done), run_time=0.5)
        self.play(tracker.animate.set_value(self.MAX_STEPS), run_time=3.0, rate_func=linear)

        stragglers = SurroundingRectangle(VGroup(bars[11], bars[24]), color=WARN_C,
                                          buff=0.08, stroke_width=3)
        verdict = T("варп занят, пока считает самая жёсткая полоса", size=29, color=WARN_C)
        verdict.next_to(head, DOWN, buff=0.35)

        self.play(FadeOut(most_done), Create(stragglers), FadeIn(verdict), run_time=1.0)
        limit = T("в худшем случае — работает 1 полоса из 32", size=24, color=WARN_C)
        limit.next_to(verdict, DOWN, buff=0.2)
        self.play(FadeIn(limit), run_time=0.5)
        self.wait(0.2)
        self.next_slide()

        # ── сбалансированный варп ────────────────────────────────────────
        steps2 = self._profile(spread=False)
        tracker2 = ValueTracker(0.0)
        bars2, *_ = self._bars(steps2, tracker2)

        new_head = T("...а если в варпе — похожие ячейки", size=30, color=OK_C)
        new_head.move_to(verdict)

        # у always_redraw-полос сначала снимаем апдейтер, иначе они переживут FadeOut
        for bar in bars:
            bar.clear_updaters()
        self.play(
            FadeOut(VGroup(bars, stragglers, verdict, limit)),
            FadeIn(new_head),
            run_time=0.8,
        )
        self.add(bars2)
        self.play(tracker2.animate.set_value(max(steps2) + 2), run_time=1.4, rate_func=linear)

        verdict2 = T("все 32 заканчивают примерно вместе", size=29, color=OK_C)
        verdict2.next_to(new_head, DOWN, buff=0.25)
        self.play(FadeIn(verdict2), run_time=0.6)
        self.wait(0.2)
        self.next_slide()

        # ── во что это обходится по времени ──────────────────────────────
        for bar in bars2:
            bar.clear_updaters()
        self.play(FadeOut(VGroup(bars2, floor, lanes_lab, axis_lab)), run_time=0.5)

        full_w = 9.0
        bad = Rectangle(width=full_w, height=0.6, stroke_width=0,
                        fill_color=WARN_C, fill_opacity=0.8)
        good = Rectangle(width=full_w * max(steps2) / 400.0, height=0.6, stroke_width=0,
                         fill_color=OK_C, fill_opacity=0.85)
        bad_lab = T("вперемешку", size=23, color=WARN_C)
        good_lab = T("похожие рядом", size=23, color=OK_C)

        for rect, lab, y in ((bad, bad_lab, -0.9), (good, good_lab, -2.3)):
            rect.move_to([-full_w / 2 + rect.width / 2, y, 0])
            lab.next_to(rect, UP, buff=0.14).align_to(rect, LEFT)

        time_head = T("время работы одного варпа — по схеме выше", size=24, color=MUTED)
        time_head.move_to([-full_w / 2, 0.3, 0], aligned_edge=LEFT)

        self.play(FadeIn(time_head), run_time=0.4)
        self.play(GrowFromEdge(bad, LEFT), FadeIn(bad_lab), run_time=0.7)
        self.play(GrowFromEdge(good, LEFT), FadeIn(good_lab), run_time=0.7)

        # если замер есть — называем его; если нет, слайд говорит то же словами
        if BALANCER_SPEEDUP is not None:
            gain = T(f"замер: ускорение в {BALANCER_SPEEDUP:g} раза", size=26, color=OK_C)
            gain_note = T(BALANCER_NOTE, size=19, color=MUTED)
            gain_group = VGroup(gain, gain_note).arrange(DOWN, buff=0.16)
            gain_group.next_to(good, DOWN, buff=0.5)
            self.play(FadeIn(gain_group, shift=UP * 0.15), run_time=0.7)

        idea = T("Идея: перед решением поставить рядом похожие ячейки", size=28, color=MEM_C)
        idea.to_edge(DOWN, buff=0.35)
        self.play(FadeIn(idea, shift=UP * 0.2), run_time=0.8)
        self.wait(0.3)


# ══════════════════════════════════════════════════════════════════════════
class S10Balancer(KodesSlide):
    """Версия 4: балансировка батча блочной сортировкой."""

    N = 48
    BUCKETS = 8

    def construct(self):
        head = title("Версия 4: балансировка батча")
        self.play(FadeIn(head, shift=DOWN * 0.2))

        rng = np.random.default_rng(3)
        keys = np.clip(rng.beta(2.0, 2.2, self.N), 0.02, 0.98)

        # ── батч как есть, после копирования ─────────────────────────────
        cells = VGroup()
        cw = 11.4 / self.N
        for i in range(self.N):
            c = Rectangle(width=cw - 0.03, height=0.6, stroke_width=0.8,
                          stroke_color=BG, fill_color=key_color(keys[i]), fill_opacity=0.95)
            c.move_to([-11.4 / 2 + cw * (i + 0.5), 1.85, 0])
            cells.add(c)
        batch_lab = T("батч на устройстве — в порядке копирования", size=22, color=MUTED)
        batch_lab.next_to(cells, UP, buff=0.22)

        self.play(LaggedStart(*[FadeIn(c, scale=0.7) for c in cells], lag_ratio=0.012),
                  FadeIn(batch_lab), run_time=1.4)
        self.wait(0.2)
        self.next_slide()

        # ── 1. fillKeys ──────────────────────────────────────────────────
        step1 = T("1.  fillKeys — каждой ячейке своё число", size=25, color=MEM_C)
        step1.next_to(cells, DOWN, buff=0.55)
        key_eq = MathTex(
            r"k \;=\; \log_{10}\!\sqrt{\frac{1}{n}\sum_i \left(\frac{1}{y_i}\frac{dy_i}{dt}\right)^{\!2}}",
            color=FG,
        ).scale(0.8).next_to(step1, DOWN, buff=0.35)
        key_note = T("обратный масштаб времени — прямая мера жёсткости\n"
                     "(есть и дешевле: температура, компонента 0 состояния)",
                     size=21, color=MUTED)
        key_note.next_to(key_eq, DOWN, buff=0.3)

        self.play(FadeIn(step1), run_time=0.4)
        self.play(LaggedStart(*[Flash(c, color=c.get_fill_color(), line_length=0.12,
                                      num_lines=8, flash_radius=0.22)
                                for c in cells], lag_ratio=0.01), run_time=1.0)
        self.play(Write(key_eq), run_time=1.2)
        self.play(FadeIn(key_note), run_time=0.5)
        self.wait(0.2)
        self.next_slide()

        # ── 2. fillBuckets ───────────────────────────────────────────────
        self.play(FadeOut(VGroup(key_eq, key_note)), run_time=0.4)
        step2 = T("2.  fillBuckets — диапазон ключа режется на равные корзины",
                  size=25, color=MEM_C)
        step2.move_to(step1)
        self.play(Transform(step1, step2), run_time=0.4)

        col_w = 1.25
        cols_x = [(-(self.BUCKETS - 1) / 2 + b) * col_w for b in range(self.BUCKETS)]
        columns = VGroup()
        for b in range(self.BUCKETS):
            frame = Rectangle(width=col_w - 0.12, height=3.2, stroke_color=MUTED,
                              stroke_width=1.4, fill_opacity=0)
            frame.set_stroke(opacity=0.55)
            frame.move_to([cols_x[b], -1.35, 0])
            columns.add(frame)
        self.play(LaggedStart(*[Create(f) for f in columns], lag_ratio=0.06), run_time=0.9)

        bucket_of = np.minimum((keys * self.BUCKETS).astype(int), self.BUCKETS - 1)
        heights = [0] * self.BUCKETS
        moves = []
        cell_h = 0.23
        for i in range(self.N):
            b = int(bucket_of[i])
            y = columns[b].get_bottom()[1] + 0.1 + heights[b] * (cell_h + 0.03) + cell_h / 2
            heights[b] += 1
            moves.append(
                cells[i].animate.stretch_to_fit_width(col_w - 0.26)
                .stretch_to_fit_height(cell_h)
                .move_to([cols_x[b], y, 0])
            )
        self.play(LaggedStart(*moves, lag_ratio=0.014), run_time=2.2)

        counts = VGroup()
        for b in range(self.BUCKETS):
            n = Integer(heights[b], color=MEM_C, font_size=28)
            n.next_to(columns[b], DOWN, buff=0.18)
            counts.add(n)
        counts_lab = T("гистограмма", size=20, color=MUTED).next_to(counts, LEFT, buff=0.3)
        self.play(FadeIn(counts), FadeIn(counts_lab), run_time=0.6)
        self.wait(0.2)
        self.next_slide()

        # ── 3. scanBuckets ───────────────────────────────────────────────
        step3 = T("3.  scanBuckets — префиксная сумма даёт смещение каждой корзины",
                  size=25, color=MEM_C)
        self.play(Transform(step1, step3.move_to(step1)), run_time=0.4)

        offsets = np.concatenate([[0], np.cumsum(heights)[:-1]])
        new_counts = VGroup()
        for b in range(self.BUCKETS):
            n = Integer(int(offsets[b]), color=OK_C, font_size=28).move_to(counts[b])
            new_counts.add(n)
        self.play(Transform(counts, new_counts),
                  Transform(counts_lab, T("смещения", size=20, color=MUTED)
                            .move_to(counts_lab)),
                  run_time=1.0)
        self.wait(0.2)
        self.next_slide()

        # ── 4. scatterOrder ──────────────────────────────────────────────
        step4 = T("4.  scatterOrder — каждая ячейка получает свой слот (order[])",
                  size=25, color=MEM_C)
        self.play(Transform(step1, step4.move_to(step1)), run_time=0.4)
        self.play(FadeOut(counts), FadeOut(counts_lab), FadeOut(columns), run_time=0.5)

        order = np.argsort(bucket_of, kind="stable")
        slot_of = np.empty(self.N, dtype=int)
        for slot, i in enumerate(order):
            slot_of[i] = slot

        sorted_moves = []
        for i in range(self.N):
            sorted_moves.append(
                cells[i].animate.stretch_to_fit_width(cw - 0.03)
                .stretch_to_fit_height(0.6)
                .move_to([-11.4 / 2 + cw * (slot_of[i] + 0.5), -0.55, 0])
            )
        self.play(LaggedStart(*sorted_moves, lag_ratio=0.012), run_time=2.0)

        sorted_lab = T("порядок обхода: похожие ячейки — рядом", size=22, color=OK_C)
        sorted_lab.next_to(cells, DOWN, buff=0.3)
        self.play(FadeIn(sorted_lab), run_time=0.5)
        self.wait(0.2)
        self.next_slide()

        # ── что достаётся варпу ──────────────────────────────────────────
        self.play(FadeOut(VGroup(step1, batch_lab)), run_time=0.4)

        warp_len = 12
        warp_box = Rectangle(width=cw * warp_len, height=0.8, stroke_color=GPU_C,
                             stroke_width=4.5, fill_opacity=0)
        warp_box.move_to([-11.4 / 2 + cw * (warp_len / 2), -0.55, 0])
        warp_lab = T("варп берёт 32 соседние позиции порядка\n"
                     "(на схеме — 12, чтобы было видно)", size=21, color=GPU_C)
        warp_lab.next_to(warp_box, UP, buff=0.4).align_to(warp_box, LEFT)

        self.play(Create(warp_box), FadeIn(warp_lab), run_time=0.8)
        for shift_k in (12, 24, 36):
            self.play(warp_box.animate.move_to(
                [-11.4 / 2 + cw * (shift_k + warp_len / 2), -0.55, 0]), run_time=0.55)
        self.wait(0.2)
        self.next_slide()

        facts = bullets(
            [
                "блочная сортировка: четыре ядра, O(n), ключи не покидают GPU",
                "переставляется только порядок обхода order[], не сами данные",
                "прежняя сортировка на хосте стоила ≈ 80 мс на миллион систем",
                "ячейка с NaN уходит в нулевую корзину и никого не тормозит",
            ],
            size=23,
            buff=0.3,
        )
        facts.move_to(DOWN * 2.55)
        self.play(
            FadeOut(VGroup(warp_box, warp_lab, sorted_lab)),
            LaggedStart(*[FadeIn(b, shift=RIGHT * 0.2) for b in facts], lag_ratio=0.3),
            run_time=2.0,
        )
        self.wait(0.3)


# ══════════════════════════════════════════════════════════════════════════
class S11Summary(KodesSlide):
    """Итог: четыре шага и снятое на каждом ограничение."""

    def construct(self):
        head = title("Четыре версии — четыре снятых ограничения")
        self.play(FadeIn(head, shift=DOWN * 0.2))

        rows_data = [
            ("1", "всё разом", "рабочая память — «число ячеек × n²», не влезает", WARN_C),
            ("2", "батчи", "рабочая память на весь батч — почти вся зря", WARN_C),
            ("3", "батч + куски", "рабочая память — по числу работающих потоков", OK_C),
            ("4", "+ балансировка", "варп больше не ждёт самую жёсткую ячейку", OK_C),
        ]

        rows = VGroup()
        for num, name, problem, color in rows_data:
            rows.add(VGroup(
                T(num, size=30, color=MUTED, weight=BOLD),
                T(name, size=25, color=FG),
                T(problem, size=24, color=color),
            ))

        # три колонки, выровненные по левому краю каждая; третья ещё и ужимается
        # по остатку ширины кадра, чтобы длинная строка не уехала за край
        for k, row in enumerate(rows):
            y = 1.55 - k * 1.05
            row[0].move_to([-6.5, y, 0])
            fit(row[1], 3.6).move_to([-6.0, y, 0], aligned_edge=LEFT)
            fit(row[2], 8.9).move_to([-2.2, y, 0], aligned_edge=LEFT)

        for row in rows:
            self.play(FadeIn(row, shift=RIGHT * 0.25), run_time=0.55)
        self.wait(0.2)
        self.next_slide()

        strike = VGroup(cross_out(rows[0][2]), cross_out(rows[1][2]))
        self.play(Create(strike[0]), run_time=0.4)
        self.play(Create(strike[1]), run_time=0.4)
        self.wait(0.2)
        self.next_slide()

        self.play(FadeOut(rows), FadeOut(strike), run_time=0.6)

        lines = [
            T("KODES", size=64, color=GPU_C, weight=BOLD),
            T("одна ячейка — один поток, тысячи ячеек — один запуск ядра",
              size=28, color=FG),
        ]
        if GPU_VS_CPU_SPEEDUP is not None:
            lines.append(T(f"ускорение против CPU — в {GPU_VS_CPU_SPEEDUP:g} раза",
                           size=27, color=OK_C))
            lines.append(T(GPU_VS_CPU_NOTE, size=20, color=MUTED))
        lines += [
            T("метод, балансировщик и размеры запуска — имена и числа в JSON",
              size=24, color=MUTED),
            T("встраивается как модель химии в reactingFoam", size=24, color=MUTED),
        ]

        final = VGroup(*lines).arrange(DOWN, buff=0.38)
        fit(final).move_to(ORIGIN)
        self.play(FadeIn(final, shift=UP * 0.25), run_time=1.2)
        self.wait(0.5)
