"""사용자가 캡처한 연계버스 카카오맵 3장과 기존 중앙 위치도를 조합한다."""
from pathlib import Path
import importlib.util
import unicodedata
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from scipy import ndimage
from scipy.spatial import cKDTree
from shapely.geometry import Polygon


def _area(root):
    spec = importlib.util.spec_from_file_location('_feeder_capture_area_015', Path(root)/'code/03_eda/study_area_015.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def load_data(root, period='morning'):
    if period not in ('morning', 'day'):
        raise ValueError("period는 'morning' 또는 'day'입니다.")
    root = Path(root)
    grid, stations, cities, _ = _area(root).load_data(root)
    records = pd.read_csv(root/'results/analysis/feeder_bus/tables/H11a_격자별_공급구조_GTX역행.csv')
    column = 'GTX역행 2025 출근' if period == 'morning' else 'GTX역행 2025'
    grid = grid.merge(records[['셀 ID', column]], on='셀 ID', validate='one_to_one')
    grid['feeder_count'] = grid[column].astype(int)
    assert len(grid)==126 and grid.feeder_count.sum()==(646 if period == 'morning' else 1015)
    return grid, stations, cities


def _components(mask, min_pixels=150):
    labels, _ = ndimage.label(mask)
    result = []
    for i, obj in enumerate(ndimage.find_objects(labels), 1):
        if obj is not None and (labels[obj]==i).sum()>min_pixels:
            result.append([(obj[1].start+obj[1].stop-1)/2,
                           (obj[0].start+obj[0].stop-1)/2,
                           obj[1].stop-obj[1].start, obj[0].stop-obj[0].start])
    return np.array(result)


def _footprint(pixels, part, station, cell_limits=(60, 90), min_pixels=150):
    """캡처의 실제 격자 중심과 GTX역 별로 확대 범위를 검증한다."""
    import geopandas as gpd
    rgb = pixels[:, :, :3].astype(float)
    r, g, b = rgb.transpose(2, 0, 1)
    stars = _components((r < 65) & (g > 75) & (g > r * 1.7) & (g > b * 1.15))
    if len(stars) != 1:
        raise ValueError('캡처에서 GTX역 별을 하나로 식별하지 못했습니다.')
    # 실제 격자의 가는 회색 테두리와 빨간 내부에서 고립된 셀을 찾는다.
    # 사진을 수정하지 않으며, 격자의 좌표를 임의로 생성하지 않는다.
    border = ndimage.binary_closing(
        (abs(rgb - 51) < 5).all(axis=2), structure=np.ones((3, 3)))
    if cell_limits[1] <= 50:
        # 작은 캡처에서는 1px 격자선이 배경과 혼합되므로 긴 회색 직선을 추출한다.
        gray = (abs(r-g) < 7) & (abs(g-b) < 7) & (r < 195) & (r > 20)
        horizontal = ndimage.binary_opening(gray, structure=np.ones((1, 25)))
        vertical = ndimage.binary_opening(gray, structure=np.ones((25, 1)))
        border = ndimage.binary_closing(horizontal | vertical, structure=np.ones((3, 3)))
    red = (r > 140) & (r > g * 1.35) & (r > b * 1.3) & (g < 170) & (b < 170)
    observed = []
    for kind, mask in [('border', border), ('red', red)]:
        for u, v, w, h in _components(mask, min_pixels=min_pixels):
            if not (cell_limits[0] < w < cell_limits[1] and cell_limits[0] < h < cell_limits[1]):
                continue
            if kind == 'red':
                fill = mask[int(v-h/2)+1:int(v+h/2)+1,
                            int(u-w/2)+1:int(u+w/2)+1].mean()
                if fill < .65:
                    continue
            if any(np.linalg.norm([u-q[0], v-q[1]]) < 4 for q in observed):
                continue
            observed.append([u, v])
    observed = np.asarray(observed)
    if len(observed) < 4:
        raise ValueError('새 캡처에서 실제 격자의 대응점을 충분히 식별하지 못했습니다.')
    part = part.to_crs(3857)
    station = gpd.GeoSeries([station], crs=5179).to_crs(3857).iloc[0]
    centers = part.geometry.centroid
    relative = np.column_stack([centers.x-station.x, -(centers.y-station.y)])
    best = None
    for scale in np.linspace(.1, .4, 1501):
        distance, indices = cKDTree(relative * scale + stars[0, :2]).query(observed)
        score = np.mean(np.sort(distance ** 2)[:max(3, int(len(distance) * .8))])
        if best is None or score < best[0]:
            best = (score, distance, indices)
    _, distance, indices = best
    matched = distance < 12
    targets, points = observed[matched], relative[indices[matched]]
    if len(targets) < 4 or len(set(indices[matched])) != len(targets):
        raise ValueError('캡처 격자와 실제 GIS 격자의 대응을 확인하지 못했습니다.')
    pair_count = len(targets)
    for rejected in range(int(pair_count * .2) + 1):
        design = np.column_stack([points, np.ones(len(points))])
        if np.linalg.matrix_rank(design) != 3:
            raise ValueError('캡처 범위 검증에 필요한 공간적으로 분산된 대응점이 부족합니다.')
        if cell_limits[1] <= 50:
            # 저해상도 캡처는 역 별을 기준점으로 고정해 배경 시설의 사각형을 배제한다.
            linear = np.linalg.lstsq(points, targets-stars[0, :2], rcond=None)[0]
            affine = np.vstack([linear, stars[0, :2]])
        else:
            affine = np.linalg.lstsq(design, targets, rcond=None)[0]
        residual = np.linalg.norm(design @ affine - targets, axis=1)
        if residual.max() <= (3.5 if cell_limits[1] <= 50 else 2.5):
            break
        if rejected == int(pair_count * .2) or len(targets) <= 4:
            raise ValueError(f'새 캡처와 공간자료의 대조 오차가 큽니다: {residual.max():.2f}px')
        remove = residual.argmax()
        targets, points = np.delete(targets, remove, axis=0), np.delete(points, remove, axis=0)
    star_error = float(np.linalg.norm(affine[2] - stars[0, :2]))
    if star_error > 8:
        raise ValueError('캡처의 GTX역 별 위치와 실제 역사 좌표가 일치하지 않습니다.')
    h, w = pixels.shape[:2]
    corners = np.asarray([[0, 0], [w, 0], [w, h], [0, h]])
    world = ((corners-affine[2]) @ np.linalg.inv(affine[:2])) * [1, -1]
    world += [station.x, station.y]
    footprint3857 = Polygon(world)
    assert footprint3857.is_valid and footprint3857.covers(station)
    footprint5179 = gpd.GeoSeries([footprint3857], crs=3857).to_crs(5179).iloc[0]
    return footprint5179, {'대응_격자':len(targets), '최대_잔차_px':float(residual.max()),
                           '역_별_대조오차_px':star_error,
                           '가려진_대응점_제외':pair_count-len(targets)}


def make_figure(grid, stations, cities, root, period='morning'):
    if period not in ('morning', 'day'):
        raise ValueError("period는 'morning' 또는 'day'입니다.")
    root = Path(root); area = _area(root)
    filenames = {}
    for name in area.ORDER:
        filename = f'{name}{" 일일" if period == "day" else ""} 연계버스 카맵.png'
        matches = [p for p in root.glob('*.png') if unicodedata.normalize('NFC',p.name)==filename]
        if len(matches)!=1:
            raise FileNotFoundError(f'캡처 파일을 확인해 주세요: {filename}')
        filenames[name] = matches[0]
    def fit_capture(pixels, part, station):
        return _footprint(pixels, part, station,
                          cell_limits=(26, 50) if period == 'day' else (60, 90),
                          min_pixels=80 if period == 'day' else 150)
    fig, axes, validation = area.make_capture_figure(
        grid,stations,cities,root,capture_filenames=filenames,footprint_fitter=fit_capture)
    for legend in list(fig.legends):legend.remove()
    for text in list(fig.texts):
        if text.get_text().startswith(('하루 전체 광역버스 승차 변화','중앙 위치도:')):text.remove()
    colors = ['#E6E6E6','#FEE0D2','#FCAE91','#FB6A4A','#A50F15']
    edges = grid.loc[grid.feeder_count.gt(0),'feeder_count'].quantile([.25,.5,.75]).to_numpy()
    assert np.allclose(edges,[3,9.5,24.75] if period == 'morning' else [3,11,32.5])
    cutoffs = np.floor(edges).astype(int)
    labels = ['0건',f'1–{cutoffs[0]}건',f'{cutoffs[0]+1}–{cutoffs[1]}건',
              f'{cutoffs[1]+1}–{cutoffs[2]}건',f'{cutoffs[2]+1}건 이상']
    # 위 광역버스 지도와 같은, 색상 구간이 서로 붙어 있는 세로 막대 범례.
    legend_ax = fig.add_axes([.665,.04,.12,.22])
    legend_ax.set_xlim(0,2.1);legend_ax.set_ylim(0,5.8);legend_ax.set_axis_off()
    legend_ax.text(0,5.35,'연계버스 승차건수',fontsize=17,fontweight='bold',va='bottom')
    for i,(color,label) in enumerate(zip(colors,labels)):
        y = 4-i
        legend_ax.add_patch(Rectangle((0,y),.28,1,facecolor=color,edgecolor='none'))
        if i in (0,len(labels)-1):
            legend_ax.text(.43,y+.5,label,fontsize=16,fontweight='bold',va='center')
    extra = [Line2D([],[],marker='*',color='none',markerfacecolor='#138A55',
                    markeredgecolor='white',markersize=18,label='GTX-A역')]
    fig.legend(handles=extra,loc='lower left',bbox_to_anchor=(.795,.11),ncol=1,
               frameon=False,fontsize=16,handlelength=1.35,labelspacing=1.3)
    fig.canvas.draw()
    return fig, axes, validation
