"""015 출근시간대 발표 그림. 원자료·분석표·이미지 파일을 저장하지 않는다."""
from pathlib import Path
import re

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ORDER = ['운정중앙', '킨텍스', '대곡']
COLORS = {'운정중앙': '#C45A88', '킨텍스': '#888888', '대곡': '#8A6446'}
BEFORE_COLOR, AFTER_COLOR = '#B9B9B9', '#8064A2'
TIME_LABEL = '06:00 이상 10:00 미만'
BROAD_FIGSIZE = (14.5, 6.6)
BROAD_YTICKS = [0, 400, 800, 1200, 1600]


def font_context():
    family = 'DejaVu Sans'
    for path in [Path.home() / 'Library/Fonts/KoPubWorld Dotum Bold.ttf',
                 Path('/System/Library/Fonts/AppleSDGothicNeo.ttc'),
                 Path('C:/Windows/Fonts/malgunbd.ttf')]:
        if path.is_file():
            fm.fontManager.addfont(str(path))
            family = fm.FontProperties(fname=str(path)).get_name()
            break
    return plt.rc_context({'font.family': family, 'font.weight': 'bold',
                           'axes.labelweight': 'bold', 'axes.titleweight': 'bold',
                           'axes.unicode_minus': False, 'figure.dpi': 130})


def load_data(root):
    """126개 격자 유지. 광역버스 출근 건수는 원자료에서 다시 집계하여 H11a와 대조."""
    root = Path(root)
    base = pd.read_csv(root / 'data/final/gtx_a/S1_126격자_건수기준_기초표.csv',
                       encoding='utf-8-sig').set_index('셀 ID')
    cells = pd.read_csv(root / 'data/processed/gtx_a/16_새데이터_공급구조유지_250m격자.csv',
                        dtype=str, encoding='utf-8-sig')
    reference = pd.read_csv(root / 'results/analysis/feeder_bus/tables/H11a_격자별_공급구조_GTX역행.csv',
                            encoding='utf-8-sig').set_index('셀 ID').loc[base.index]
    assert len(base) == 126 and base.index.is_unique
    assert set(base.index) == set(cells['셀 ID'].astype(int))
    assert reference['권역'].eq(base.nearest_station).all()
    data = base[['nearest_station']].copy()
    for year, filename, date in [
        (2024, 'gtx_a_transport_card_20241017_raw_with_coords_api_filled.csv', '20241017'),
        (2025, 'gtx_a_transport_card_20251106_raw_with_coords.csv', '20251106')]:
        card = pd.read_csv(root / 'data/raw/gtx_a/transport_card' / filename,
                           dtype=str, encoding='utf-8-sig',
                           usecols=['query_date', 'opr_ymd', '승차정류장ID', 'ride_dt'])
        assert card.query_date.eq(date).all() and card.opr_ymd.eq(date).all()
        lookup = pd.DataFrame([(int(row['셀 ID']), sid) for _, row in cells.iterrows()
                               for sid in re.findall(r'\d{6,}', row[f'{year}년 정류장 ID'])],
                              columns=['셀 ID', '승차정류장ID'])
        assert not lookup['승차정류장ID'].duplicated().any()
        selected = card.merge(lookup, on='승차정류장ID', how='inner', validate='many_to_one')
        np.testing.assert_array_equal(selected.groupby('셀 ID').size().reindex(base.index, fill_value=0),
                                      base[f'board_{year}'])
        times = pd.to_datetime(selected.ride_dt, format='%Y%m%d%H%M%S', errors='raise')
        selected = selected[times.dt.hour.ge(6) & times.dt.hour.lt(10)]
        data[f'board_{year}'] = selected.groupby('셀 ID').size().reindex(base.index, fill_value=0)
    data['change'] = data.board_2025 - data.board_2024
    data['feeder'] = reference['GTX역행 2025 출근'].astype(int)
    np.testing.assert_array_equal(data.change, reference['Δ광역 출근'])
    assert data[['board_2024', 'board_2025', 'feeder']].sum().tolist() == [3098, 2286, 646]
    assert data.groupby('nearest_station').size().reindex(ORDER).tolist() == [63, 26, 37]
    assert data.feeder.gt(0).sum() == 38
    return data


def broad_summary(data):
    result = data.groupby('nearest_station').agg(
        격자수=('change', 'size'), 개통전=('board_2024', 'sum'),
        개통후=('board_2025', 'sum'), 변화량=('change', 'sum'),
        감소격자=('change', lambda s: int(s.lt(0).sum())),
        유지격자=('change', lambda s: int(s.eq(0).sum())),
        증가격자=('change', lambda s: int(s.gt(0).sum())),
        **{'40건이상감소': ('change', lambda s: int(s.le(-40).sum())),
           '30건이상감소': ('change', lambda s: int(s.le(-30).sum()))}).reindex(ORDER)
    result.loc['전체'] = result.sum()
    result['변화율(%)'] = 100 * result['변화량'] / result['개통전']
    result.index.name = '권역'
    assert result.loc[ORDER, '변화량'].tolist() == [-456, -395, 39]
    assert result.loc[ORDER, '40건이상감소'].tolist() == [2, 4, 0]
    assert result.loc[ORDER, '30건이상감소'].tolist() == [5, 5, 0]
    return result


def feeder_summary(data):
    result = data.groupby('nearest_station').agg(
        격자수=('feeder', 'size'), 승차건수=('feeder', 'sum'),
        이용격자=('feeder', lambda s: int(s.gt(0).sum()))).reindex(ORDER)
    result.loc['전체'] = result.sum()
    result['승차비중(%)'] = 100 * result['승차건수'] / result.loc['전체', '승차건수']
    result['이용격자비율(%)'] = 100 * result['이용격자'] / result['격자수']
    result.index.name = '권역'
    assert result.loc[ORDER, '승차건수'].tolist() == [488, 41, 117]
    return result


def correlation_summary(data):
    rows = []
    for name in [*ORDER, '전체']:
        part = data if name == '전체' else data[data.nearest_station.eq(name)]
        rho, p = spearmanr(part.change, part.feeder, alternative='two-sided')
        rows.append({'권역': name, '격자수': len(part), '승차건수': int(part.feeder.sum()),
                     'rho': float(rho), 'p': float(p)})
    result = pd.DataFrame(rows).set_index('권역')
    np.testing.assert_allclose(result.loc[ORDER, 'rho'], [-.399, -.151, -.327], atol=.0005)
    assert result.loc['전체', '격자수'] == 126
    return result


def make_broad_bar(data):
    """기존 일일 막대의 회색/보라색과 굵은 레이블 유지."""
    table = broad_summary(data)
    with font_context():
        fig, ax = plt.subplots(figsize=BROAD_FIGSIZE, facecolor='white')
        fig.subplots_adjust(left=.12, right=.98, bottom=.15, top=.96)
        x, width = np.arange(3), .34
        for offset, col, label, color in [(-width/2, '개통전', 'GTX-A 개통 전', BEFORE_COLOR),
                                          (width/2, '개통후', 'GTX-A 개통 후', AFTER_COLOR)]:
            values = table.loc[ORDER, col].to_numpy()
            bars = ax.bar(x + offset, values, width, color=color, label=label)
            ax.bar_label(bars, labels=[f'{v:,}' for v in values], padding=6,
                         fontsize=18, fontweight='bold')
        ax.set_xticks(x, [name + '역' for name in ORDER], fontsize=19)
        ax.set_xlabel('GTX-A 역 권역', fontsize=18, labelpad=12)
        ax.set_ylabel('승차건수(건)', fontsize=20, labelpad=14)
        ax.set_ylim(0, 1600)
        ax.set_yticks(BROAD_YTICKS)
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        ax.tick_params(axis='y', labelsize=17)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(loc='upper right', frameon=False, fontsize=16)
        for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
            label.set_fontweight('bold')
    return fig


def load_hourly_data(root, morning_data=None):
    """원고 그림 3의 24시간 집계표. 일일 합계 및 6~9시 격자 집계와 대조한다."""
    root = Path(root)
    table = pd.read_csv(root / 'results/paper/tables/3_1_2_시간대별_광역버스_승차변화.csv',
                        encoding='utf-8-sig').set_index(['권역', '시간']).sort_index()
    assert len(table) == 96 and table.index.is_unique
    base = pd.read_csv(root / 'data/final/gtx_a/S1_126격자_건수기준_기초표.csv',
                       encoding='utf-8-sig')
    morning_data = load_data(root) if morning_data is None else morning_data
    region_morning = broad_summary(morning_data)
    for year in [2024, 2025]:
        column = f'{year}년'
        values = table[column]
        assert values.notna().all() and values.ge(0).all()
        np.testing.assert_array_equal(values, values.round())
        whole = table.loc['전체', column]
        assert whole.index.tolist() == list(range(24))
        assert int(whole.sum()) == int(base[f'board_{year}'].sum())
        np.testing.assert_array_equal(whole, table.loc[ORDER, column].groupby('시간').sum())
        for name in [*ORDER, '전체']:
            expected = region_morning.loc[name, '개통전' if year == 2024 else '개통후']
            assert int(table.loc[name, column].loc[6:9].sum()) == int(expected)
    np.testing.assert_array_equal(table['변화량(건)'], table['2025년']-table['2024년'])
    return table.loc['전체'].copy()


def make_hourly_figure(hourly):
    """원고 그림 3과 같은 시간대별 선그래프. 5~23시 표시, 6~9시 배경 강조."""
    whole = hourly.loc[5:23]
    with font_context():
        fig, ax = plt.subplots(figsize=BROAD_FIGSIZE, facecolor='white')
        fig.subplots_adjust(left=.12, right=.98, bottom=.15, top=.96)
        ax.axvspan(5.5, 9.5, color='#F4DA9A', alpha=.35, zorder=0)
        for year, label, marker, color in [
            (2024, 'GTX-A 개통 전', 'o', BEFORE_COLOR),
            (2025, 'GTX-A 개통 후', 's', AFTER_COLOR)]:
            ax.plot(whole.index, whole[f'{year}년'], color=color, marker=marker,
                    markersize=6, linewidth=2.6, label=label, zorder=3)
        ax.set_xticks(range(5, 24), [f'{hour}시' for hour in range(5, 24)])
        ax.set_xlim(4.6, 23.4)
        ax.set_ylim(0, 1600)
        ax.set_yticks(BROAD_YTICKS)
        ax.set_xlabel('승차 시간대', fontsize=18, labelpad=12)
        ax.set_ylabel('승차건수(건)', fontsize=20, labelpad=14)
        ax.legend(frameon=False, fontsize=16, loc='upper right')
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_axisbelow(True); ax.grid(axis='y', alpha=.20, linewidth=.7)
        ax.tick_params(axis='both', labelsize=14, pad=7)
        ax.tick_params(axis='y', labelsize=17)
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
            label.set_fontweight('bold')
    return fig


def make_donut(data):
    """기존 도넛과 동일: 권역색·건수/비중·가운데 합계·아래 범례·투명 배경."""
    values = feeder_summary(data).loc[ORDER, '승차건수'].to_numpy()
    total = int(values.sum())
    with font_context():
        fig, ax = plt.subplots(figsize=(7.6, 8.2), facecolor='none')
        fig.subplots_adjust(left=.02, right=.98, bottom=.12, top=.98)
        _, _, texts = ax.pie(values, colors=[COLORS[name] for name in ORDER],
               startangle=90, counterclock=False,
               wedgeprops={'width': .44, 'edgecolor': 'white', 'linewidth': 3},
               autopct=lambda p: f'{round(p * total / 100):,}건\n{p:.1f}%', pctdistance=.78,
               textprops={'fontsize': 20, 'fontweight': 'bold', 'color': 'white', 'linespacing': 1.35})
        texts[1].set_fontsize(17)  # 작은 킨텍스 조각도 같은 위치에서 읽히도록 한다.
        ax.text(0, 0, f'{total:,}건', ha='center', va='center', fontsize=34, fontweight='bold', color='#111111')
        fig.legend(handles=[Patch(facecolor=COLORS[name], edgecolor='none', label=name+'역') for name in ORDER],
                   loc='lower center', bbox_to_anchor=(.5, .025), ncol=3, frameon=False,
                   fontsize=17, handlelength=1.15, handleheight=1.15, handletextpad=.5, columnspacing=1.3)
        ax.set_aspect('equal'); ax.set_facecolor('none')
    return fig


def make_scatter(data):
    """원래 집계의 출근 646건. 0건 포함 126개 전부, 권역별 X축/공통 Y축, 추세선 없음."""
    summary = correlation_summary(data)
    shown = data.copy()
    rng = np.random.default_rng(42)
    shown['plot_x'] = shown.change + rng.uniform(-.75, .75, len(shown))
    shown['plot_y'] = shown.feeder + rng.uniform(-.22, .22, len(shown))
    with font_context():
        fig, axes = plt.subplots(1, 3, figsize=(21, 7.8), sharey=True, facecolor='white')
        fig.subplots_adjust(left=.10, right=.985, bottom=.17, top=.78, wspace=.16)
        for ax, name in zip(axes, ORDER):
            part = shown[shown.nearest_station.eq(name)]
            row = summary.loc[name]
            ax.scatter(part.plot_x, part.plot_y, s=105, color=COLORS[name], alpha=.85,
                       edgecolors='white', linewidths=.75, zorder=3)
            xmin, xmax = part.change.min(), part.change.max()
            padding = max(1.2, (xmax-xmin)*.035)
            ax.set_xlim(xmin-padding, xmax+padding); ax.set_ylim(-3, 80)
            ax.set_yticks([0, 20, 40, 60, 80])
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
            ax.axvline(0, color='#BFBFBF', lw=.85, zorder=1)
            ax.axhline(0, color='#BFBFBF', lw=.85, zorder=1)
            ax.set_axisbelow(True); ax.grid(color='#EAEAEA', lw=.8)
            ax.spines[['top', 'right']].set_visible(False)
            ax.tick_params(labelsize=20, pad=8, colors='#111111')
            ptext = 'p < .001' if row['p'] < .001 else f'p = {row["p"]:.3f}'
            title_color = '#444444' if name == '킨텍스' else COLORS[name]
            ax.set_title(f'{name}역 ({len(part)}개 격자)\nρ = {row["rho"]:.3f}, {ptext}',
                         fontsize=24, color=title_color, fontweight='bold', pad=18, linespacing=1.4)
            ax.set_xlabel('광역버스 승차 변화량(건)', fontsize=21, labelpad=15)
            for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
                label.set_fontweight('bold')
            assert len(ax.collections[0].get_offsets()) == len(part)
            assert part.plot_x.between(*ax.get_xlim()).all() and part.plot_y.between(*ax.get_ylim()).all()
        axes[0].set_ylabel('GTX역 연계버스 승차건수(건)', fontsize=22, labelpad=17)
        assert sum(len(ax.collections[0].get_offsets()) for ax in axes) == 126
    return fig
