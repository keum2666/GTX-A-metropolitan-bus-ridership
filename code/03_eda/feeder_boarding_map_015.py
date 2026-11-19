"""출근시간대 연계버스 승차건수 지도. 실제 126개 격자와 도보시간 사용."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import study_area_015 as area

LEVELS = [0, 1, 5, 10, 20, 40, np.inf]
LABELS = ['0건', '1–4건', '5–9건', '10–19건', '20–39건', '40건 이상']
COLORS = ['#E6E6E6', '#FEE5D9', '#FCAE91', '#FB6A4A', '#DE2D26', '#A50F15']
CYAN = '#00CFE8'


def load_data(root):
    root = Path(root)
    grid, stations, cities, _ = area.load_data(root)
    reference = pd.read_csv(root/'results/analysis/feeder_bus/tables/H11a_격자별_공급구조_GTX역행.csv')
    grid = grid.merge(reference[['셀 ID', 'GTX역행 2025 출근']], on='셀 ID', validate='one_to_one')
    grid['feeder_count'] = grid['GTX역행 2025 출근'].astype(int)
    grid['walk20'] = grid.walk_min.le(20)
    assert len(grid) == 126 and grid.feeder_count.sum() == 646
    assert grid.feeder_count.gt(0).sum() == 38 and grid.walk20.sum() == 11
    assert grid.groupby('nearest_station').feeder_count.sum().reindex(area.ORDER).tolist() == [488, 41, 117]
    assert grid.groupby('nearest_station').walk20.sum().reindex(area.ORDER).tolist() == [6, 4, 1]
    return grid, stations, cities


def make_figure(grid, stations, cities, root=None):
    fig, axes, bounds = area.make_figure(grid, stations, cities)
    for legend in list(fig.legends):
        legend.remove()
    # 기존 변화량용 하단 설명만 교체한다. 위치도·축척 등은 유지한다.
    for label in list(fig.texts):
        if '하루 전체 광역버스' in label.get_text():
            label.remove()
    context = None
    if root is not None:
        path = Path(root)/'data/processed/gtx_a/map_context/주변시설_철도노선.geojson'
        if path.exists():
            context = gpd.GeoDataFrame.from_features(json.loads(path.read_text())['features'], crs=4326).to_crs(grid.crs)
    for name in area.ORDER:
        ax = axes[name]
        part = grid.loc[grid.nearest_station.eq(name)]
        # make_figure의 첫 번째 컬렉션은 실제 격자, 두 번째는 역사이다.
        ax.collections[0].remove()
        if context is not None:
            rail = context.loc[context.kind.eq('rail') & context.geometry.intersects(__import__('shapely').geometry.box(*bounds[name]))]
            if not rail.empty:
                rail.plot(ax=ax, color='#CBCBCB', linewidth=1.0, zorder=1)
        bins = np.searchsorted([0, 4, 9, 19, 39], part.feeder_count.to_numpy(), side='left')
        part.plot(ax=ax, color=[COLORS[i] for i in bins], edgecolor='#444444', linewidth=.65, zorder=4)
        near = part.loc[part.walk20]
        if not near.empty:
            collection = near.boundary.plot(ax=ax, color=CYAN, linewidth=2.8, zorder=6).collections[-1]
            collection.set_path_effects([pe.Stroke(linewidth=4.2, foreground='#202020'), pe.Normal()])
        area._frame(ax, bounds[name])
    handles = [Patch(facecolor=color, edgecolor='#555555', label=label) for label, color in zip(LABELS, COLORS)]
    fig.legend(handles=handles, title='출근시간대 GTX역 연계버스 승차건수', title_fontsize=18,
               loc='lower center', bbox_to_anchor=(.5,.047), ncol=6, frameon=False,
               fontsize=17, handlelength=1.4, columnspacing=1.7)
    other = [Patch(facecolor='none', edgecolor=CYAN, linewidth=3,
                   label='GTX역까지 도보시간 20분 이하'),
             Line2D([], [], marker='*', color='none', markerfacecolor='#138A55',
                    markeredgecolor='white', markersize=18, label='GTX-A역')]
    fig.legend(handles=other, loc='lower center', bbox_to_anchor=(.5,.005),
               ncol=2, frameon=False, fontsize=17, columnspacing=3)
    fig.canvas.draw()
    return fig, axes, bounds


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = area.find_root()
    grid, stations, cities = load_data(root)
    fig, _, _ = make_figure(grid, stations, cities, root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=240, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(grid.groupby('nearest_station').agg(격자수=('셀 ID','size'),승차건수=('feeder_count','sum'),도보20분이하=('walk20','sum')).reindex(area.ORDER).to_string())
    print(args.output)
