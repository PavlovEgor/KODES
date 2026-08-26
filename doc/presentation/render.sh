#!/usr/bin/env bash
#
# Рендер сцен manim и сборка их в одну .pptx.
#
#   ./render.sh              все сцены, качество 1080p30, затем kodes.pptx
#   ./render.sh -l           черновое качество (480p15) — для быстрой проверки
#   ./render.sh S09Warp      только одна сцена (pptx не собирается)
#
# Частоту кадров задаёт флаг качества, а не frame_rate из manim.cfg: -qh сам по
# себе это 1080p60, вдвое дольше и вдвое тяжелее при той же картинке на экране,
# поэтому 30 к/с задаются явным --fps.
#
set -euo pipefail
cd "$(dirname "$0")"

# shellcheck disable=SC1091
source venv/bin/activate

SCENES=(
    S01Title
    S02Splitting
    S03Independence
    S04CPU
    S05GPUIdea
    S06NaiveTransfer
    S07Batches
    S08Chunks
    S09Warp
    S10Balancer
    S11Summary
)

QUALITY=(-qh --fps 30)
if [[ "${1-}" == "-l" ]]; then
    QUALITY=(-ql)
    shift
fi

if [[ $# -gt 0 ]]; then
    for scene in "$@"; do
        manim render "${QUALITY[@]}" presentation.py "$scene"
    done
    exit 0
fi

# По одной сцене на процесс: manim-slides склеивает слайды в пуле процессов и
# на длинном списке сцен в одном запуске это иногда встаёт намертво.
for scene in "${SCENES[@]}"; do
    echo "── $scene"
    manim render "${QUALITY[@]}" presentation.py "$scene"
done

manim-slides convert "${SCENES[@]}" kodes.pptx --to pptx

echo
echo "Готово:  $(pwd)/kodes.pptx"
