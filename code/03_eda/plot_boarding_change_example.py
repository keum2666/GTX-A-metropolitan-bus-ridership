"""승차 변화량 크기 비교 예시: 원고용 고해상도 PNG와 벡터 SVG."""

from pathlib import Path
import hashlib
import json
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[2]
ASSET_NAME = "광역버스_승차변화량_크기비교_예시"
ASSET_FOLDERS = ("morning_paper_ppt_assets", "daily_paper_assets")


def make_figure():
    font_path = Path.home() / "Library/Fonts/KoPubWorld Dotum Bold.ttf"
    if font_path.is_file():
        font_manager.fontManager.addfont(str(font_path))
        font_name = font_manager.FontProperties(fname=font_path).get_name()
    else:
        font_name = "AppleGothic"

    plt.rcParams.update({
        "font.family": font_name,
        "font.weight": "bold",
        "font.size": 22,
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "figure.titleweight": "bold",
        "axes.unicode_minus": True,
        "svg.fonttype": "path",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    })

    fig, ax = plt.subplots(figsize=(10, 8.6), facecolor="white")
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.13, top=0.86)
    ax.set_xlim(-0.75, 2.15)
    ax.set_ylim(-66, 66)
    ax.axis("off")

    # Each rectangle shows a conceptual range, not an observed count or interval.
    # Its ends use the same signed values as the supplied presentation.
    examples = [
        (0.0, -10, 50, "#5096CE", "변화량 큼"),
        (1.4, -50, 10, "#E59978", "변화량 작음"),
    ]
    width = 0.70
    for x, low, high, color, label in examples:
        ax.add_patch(Rectangle(
            (x - width / 2, low), width, high - low,
            facecolor=color, edgecolor="none", zorder=2,
        ))
        ax.text(x, high + 4.5, f"+{high}", ha="center", va="bottom",
                fontsize=27, color=color)
        ax.text(x, low - 4.5, f"−{abs(low)}", ha="center", va="top",
                fontsize=27, color=color)
        ax.text(x, -65, label, ha="center", va="top", fontsize=23,
                color="#202020", clip_on=False)

    ax.plot([-0.62, 2.02], [0, 0], color="#555555", linewidth=1.5,
            zorder=3, solid_capstyle="butt")
    ax.text(-0.68, 0, "0", ha="right", va="center", fontsize=22,
            color="#333333")
    fig.suptitle("광역버스 승차건수 변화량 비교", x=0.525, y=0.955,
                 fontsize=27, color="#202020")
    return fig


def main():
    primary = ROOT / "results/paper" / ASSET_FOLDERS[0]
    primary.mkdir(parents=True, exist_ok=True)
    fig = make_figure()
    png = primary / f"{ASSET_NAME}.png"
    svg = primary / f"{ASSET_NAME}.svg"
    fig.savefig(png, dpi=600, facecolor="white", transparent=False)
    fig.savefig(svg, facecolor="white", transparent=False)
    plt.close(fig)

    for folder_name in ASSET_FOLDERS:
        folder = ROOT / "results/paper" / folder_name
        folder.mkdir(parents=True, exist_ok=True)
        if folder != primary:
            shutil.copyfile(png, folder / png.name)
            shutil.copyfile(svg, folder / svg.name)
        metadata = {
            "source": "기말발표_출근버전.pptx, slide 8의 개념도를 도형·글자로 재작성",
            "generator": "code/03_eda/plot_boarding_change_example.py",
            "sha256": hashlib.sha256((folder / png.name).read_bytes()).hexdigest(),
            "description": "부호를 살린 변화량 비교 예시. +50/−10, +10/−50. 실제 관측값 아님.",
            "format": "600 dpi PNG (6000×5160), 글꼴을 윤곽선으로 포함한 벡터 SVG",
            "font": "KoPubWorld Dotum Bold; 기존 기초통계·도넛·상관관계 그래프와 동일",
            "edit": "하단 예시 수치 설명 문구 없음. 원고의 기존 이미지 경로에 교체.",
        }
        (folder / "승차변화량_예시_자료출처.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        print(folder / png.name)
        print(folder / svg.name)


if __name__ == "__main__":
    main()
