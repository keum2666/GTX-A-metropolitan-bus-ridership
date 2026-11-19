"""PPT의 서울 접근시간 지도·강조 상자·설명 문구를 원고용 그림으로 결합."""

from pathlib import Path
from io import BytesIO
import hashlib
import json
import re
import shutil
import unicodedata
import xml.etree.ElementTree as ET
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


def source_pptx():
    folder = next(p for p in ROOT.parent.iterdir()
                  if unicodedata.normalize("NFC", p.name) == "캡스1")
    return next(p for p in folder.iterdir()
                if unicodedata.normalize("NFC", p.name) == "기말발표_출근버전.pptx")


def text_lines(shape):
    return ["".join(n.text or "" for n in p.findall(".//a:t", NS))
            for p in shape.findall(".//a:p", NS)]


def bounds(shape):
    transform = shape.find("p:spPr/a:xfrm", NS)
    offset = transform.find("a:off", NS)
    extent = transform.find("a:ext", NS)
    return tuple(int(offset.get(k)) for k in ("x", "y")) + tuple(
        int(extent.get(k)) for k in ("cx", "cy"))


def load_components(pptx):
    with zipfile.ZipFile(pptx) as archive:
        for member in archive.namelist():
            if not re.fullmatch(r"ppt/slides/slide\d+\.xml", member):
                continue
            slide = ET.fromstring(archive.read(member))
            shapes = list(slide.find("p:cSld/p:spTree", NS))
            annotation = next((s for s in shapes if
                               "서울 접근시간 단축폭이 가장" in "\n".join(text_lines(s))), None)
            if annotation is not None:
                break
        else:
            raise ValueError("서울 접근시간 비교 문구가 있는 슬라이드를 찾지 못했습니다.")

        rel_member = str(Path(member).parent / "_rels" / (Path(member).name + ".rels"))
        rels = {node.get("Id"): node.get("Target")
                for node in ET.fromstring(archive.read(rel_member))}

        def picture_bytes(shape):
            blip = shape.find(".//a:blip", NS)
            target = rels[blip.get("{" + NS["r"] + "}embed")]
            return archive.read("ppt/" + target.removeprefix("../"))

        pictures = [s for s in shapes if s.tag == "{" + NS["p"] + "}pic"]
        picture = max(pictures, key=lambda s: bounds(s)[2] * bounds(s)[3])
        data = picture_bytes(picture)
        image = Image.open(BytesIO(data)).convert("RGB")
        highlight = next(s for s in shapes
                         if s.find("p:spPr/a:ln/a:solidFill/a:srgbClr", NS) is not None
                         and s.find("p:spPr/a:ln/a:solidFill/a:srgbClr", NS).get("val") == "C12976")
        # The presentation uses a separate white picture to hide the map's source footer.
        source_masks = []
        for candidate in pictures:
            if candidate is picture:
                continue
            im = Image.open(BytesIO(picture_bytes(candidate))).convert("RGB")
            if im.getextrema() == ((255, 255), (255, 255), (255, 255)):
                source_masks.append(bounds(candidate))
        assert source_masks, "자료 표기를 가리는 흰색 도형을 찾지 못했습니다."
        return image, bounds(picture), highlight, annotation, source_masks, member


def make_figure(pptx):
    image, map_bounds, highlight, annotation, masks, member = load_components(pptx)
    font = Path.home() / "Library/Fonts/KoPubWorld Dotum Bold.ttf"
    font_manager.fontManager.addfont(str(font))
    family = font_manager.FontProperties(fname=font).get_name()
    plt.rcParams.update({"font.family": family, "font.weight": "bold",
                         "svg.fonttype": "path", "pdf.fonttype": 42})
    width, height = image.size
    x0, y0, cx, cy = map_bounds
    sx, sy = width / cx, height / cy
    figure_width = 10
    scale = figure_width / (cx / 914400)

    def local(box):
        x, y, w, h = box
        return (x - x0) * sx, (y - y0) * sy, w * sx, h * sy

    fig = plt.figure(figsize=(figure_width, figure_width * height / width), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(image, extent=(0, width, height, 0), interpolation="none", zorder=1)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")

    hx, hy, hw, hh = local(bounds(highlight))
    ax.add_patch(Rectangle((hx, hy), hw, hh, fill=False, edgecolor="#C12976",
                           linewidth=1.5 * scale, linestyle=(0, (2, 2)), zorder=2))

    tx, ty, tw, th = local(bounds(annotation))
    body = annotation.find("p:txBody/a:bodyPr", NS)
    top_inset = int(body.get("tIns", "45720")) * sy
    size = int(annotation.find(".//a:rPr", NS).get("sz")) / 100
    ax.text(tx + tw / 2, ty + top_inset, "\n".join(text_lines(annotation)),
            ha="center", va="top", fontsize=size * scale, fontweight="bold",
            linespacing=1.15, color="black", zorder=3)

    for mask in masks:
        mx, my, mw, mh = local(mask)
        ax.add_patch(Rectangle((mx, my), mw, mh, facecolor="white", edgecolor="none", zorder=4))

    return fig, member, "\n".join(text_lines(annotation)), masks


def main():
    pptx = source_pptx()
    fig, member, annotation, masks = make_figure(pptx)
    folder = ROOT / "results/paper/morning_paper_ppt_assets"
    png = folder / "PPT_서울역_접근시간_비교.png"
    svg = png.with_suffix(".svg")
    fig.savefig(png, dpi=600, facecolor="white", transparent=False)
    fig.savefig(svg, facecolor="white", transparent=False)
    preview = Path("/tmp/gtx_access_composite_20261117/서울접근시간_합성_미리보기.png")
    preview.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(preview, dpi=130, facecolor="white", transparent=False)
    plt.close(fig)
    daily = ROOT / "results/paper/daily_paper_assets/서울역_접근시간_비교.png"
    shutil.copyfile(png, daily)
    shutil.copyfile(svg, daily.with_suffix(".svg"))
    for asset in (png, daily):
        metadata = {"source": str(pptx), "slide": member,
                    "components": "PPT의 지도 원본, 점선 강조 상자, 설명 문구, 자료 표기 제거용 흰색 도형",
                    "annotation": annotation, "source_footer_hidden": True,
                    "generator": "code/03_eda/compose_seoul_access_ppt_figure.py",
                    "sha256": hashlib.sha256(asset.read_bytes()).hexdigest()}
        (asset.parent / "서울접근시간_지도합성_자료출처.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        print(asset)
    print(preview)


if __name__ == "__main__":
    main()
