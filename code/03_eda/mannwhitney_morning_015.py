"""015 출근시간대 맨휘트니 비교. 원자료·그림 파일을 저장하지 않는다."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

REGIONS = ['전체', '운정중앙', '킨텍스', '대곡']
COLORS = {'운정중앙': '#C45A88', '킨텍스': '#888888', '대곡': '#8A6446'}
X, Y = 'Δ광역 출근', 'GTX역행 2025 출근'


def load_data(root):
    data = pd.read_csv(Path(root) / 'results/analysis/feeder_bus/tables/'
                       'H11a_격자별_공급구조_GTX역행.csv', encoding='utf-8-sig')
    assert len(data) == 126 and data['셀 ID'].is_unique
    assert data[[X, Y]].notna().all().all()
    assert np.isfinite(data[[X, Y]].to_numpy()).all()
    assert data[Y].ge(0).all() and data[Y].eq(data[Y].round()).all()
    assert data[Y].sum() == 646 and data[X].sum() == -812
    assert data[Y].gt(0).sum() == 38
    assert data.groupby('권역').size().reindex(REGIONS[1:]).tolist() == [63, 26, 37]
    return data.sort_values('셀 ID').copy()


def calculate_stats(data):
    rows = []
    for region in REGIONS:
        part = data if region == '전체' else data.loc[data['권역'].eq(region)]
        no = part.loc[part[Y].eq(0), X].to_numpy()
        yes = part.loc[part[Y].gt(0), X].to_numpy()
        # 하루 전체 발표자료의 U와 같은 입력 순서: 0건 집단, 1건 이상 집단.
        result = mannwhitneyu(no, yes, alternative='two-sided',
                              method='asymptotic', use_continuity=True)
        rows.append({'권역': region, '전체 격자': len(part),
                     '승차 없음': len(no), '승차 있음': len(yes),
                     '중앙값 없음': float(np.median(no)),
                     '중앙값 있음': float(np.median(yes)),
                     'U(0건 집단 먼저)': float(result.statistic),
                     'p-value': float(result.pvalue),
                     '변화량이 낮은 쌍의 비율': float(result.statistic / (len(no) * len(yes)))})
    return pd.DataFrame(rows).set_index('권역')


def slide_copy(data, stats):
    active = data.loc[data[Y].gt(0)]
    total_drop = -data[X].sum()
    active_drop = -active[X].sum()
    row = stats.loc['전체']
    return [
        '연계버스 승차 격자에서 광역버스 승차가 더 크게 감소하는 경향 '
        f'(U = {row["U(0건 집단 먼저)"]:.1f}, 격자 쌍의 '
        f'{row["변화량이 낮은 쌍의 비율"]:.0%}, p = {row["p-value"]:.3f})',
        '운정중앙은 통계적으로 유의, 킨텍스·대곡은 유의하지 않음',
        f'연계버스 승차 격자는 {len(active)}개({len(active)/len(data):.0%})지만 '
        f'전체 순감소량의 {active_drop/total_drop:.0%}({active_drop:,.0f}건)가 '
        '이 격자에서 나타남',
    ]


def make_figure(data, stats):
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import matplotlib.colors as mcolors

    family = 'DejaVu Sans'
    for name in ['KoPubWorld Dotum Medium.ttf', 'KoPubWorld Dotum Bold.ttf']:
        font = Path.home() / 'Library/Fonts' / name
        if font.is_file():
            fm.fontManager.addfont(str(font))
            family = fm.FontProperties(fname=str(font)).get_name()
    if family == 'DejaVu Sans':
        for font in [Path('/System/Library/Fonts/AppleSDGothicNeo.ttc'),
                     Path('C:/Windows/Fonts/malgun.ttf')]:
            if font.is_file():
                fm.fontManager.addfont(str(font))
                family = fm.FontProperties(fname=str(font)).get_name()
                break
    rc = {'font.family': family, 'axes.unicode_minus': False,
          'font.weight': 'normal', 'axes.labelweight': 'bold'}
    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, 4, figsize=(22, 7.4), sharey=True,
                                 facecolor='white')
        fig.subplots_adjust(left=.065, right=.99, bottom=.25, top=.77, wspace=.085)
        rng = np.random.default_rng(42)
        for ax, region in zip(axes, REGIONS):
            part = data if region == '전체' else data.loc[data['권역'].eq(region)]
            arrays = [part.loc[part[Y].eq(0), X].to_numpy(),
                      part.loc[part[Y].gt(0), X].to_numpy()]
            boxes = ax.boxplot(arrays, positions=[0, 1], widths=.70,
                               whis=1.5, patch_artist=True, showfliers=False,
                               medianprops={'color': '#111111', 'linewidth': 3},
                               whiskerprops={'color': '#888888', 'linewidth': 1.5},
                               capprops={'color': '#888888', 'linewidth': 1.5})
            color = COLORS.get(region, '#888888')
            for i, box in enumerate(boxes['boxes']):
                box.set_facecolor('#DDDDDD' if i == 0 else
                                  '#FFFFFF' if region == '전체' else
                                  mcolors.to_rgba(color, .26))
                box.set_edgecolor('#CCCCCC')
                box.set_linewidth(1.3)
            for i, array in enumerate(arrays):
                group = part.loc[part[Y].eq(0) if i == 0 else part[Y].gt(0)]
                point_colors = group['권역'].map(COLORS) if region == '전체' else color
                ax.scatter(i + rng.uniform(-.16, .16, len(array)), array,
                           c=point_colors, s=43, alpha=.83,
                           edgecolors='white', linewidths=.55, zorder=3)
            row = stats.loc[region]
            title = '전체 (126개)' if region == '전체' else f'{region}역'
            ptext = 'p < .001' if row['p-value'] < .001 else f'p = {row["p-value"]:.3f}'
            ax.set_title(f'{title}\nU = {row["U(0건 집단 먼저)"]:.1f}, {ptext}',
                         fontsize=24, fontweight='bold', pad=46, linespacing=1.3)
            for i, array in enumerate(arrays):
                ax.text(i, 1.025, f'중앙값 {np.median(array):g}',
                        transform=ax.get_xaxis_transform(), ha='center',
                        va='bottom', fontsize=19, color='#333333')
            ax.axhline(0, color='#B0B0B0', linewidth=.9, zorder=0)
            ax.set_axisbelow(True)
            ax.grid(axis='y', color='#E7E7E7', linewidth=.8)
            ax.set_xlim(-.6, 1.6)
            ax.set_ylim(-105, 48)
            ax.set_yticks([-100, -75, -50, -25, 0, 25])
            ax.set_xticks([0, 1], [f'0건\n({len(arrays[0])}개)',
                                   f'1건 이상\n({len(arrays[1])}개)'])
            ax.tick_params(axis='both', labelsize=20, pad=8, colors='#111111')
            ax.spines[['top', 'right']].set_visible(False)
            for spine in ['left', 'bottom']:ax.spines[spine].set_linewidth(1)
            assert sum(len(c.get_offsets()) for c in ax.collections) == len(part)
            assert part[X].between(*ax.get_ylim()).all()
        axes[0].set_ylabel('광역버스 승차 변화량(건)', fontsize=23, labelpad=14)
        fig.text(.53, .055, 'GTX역 연계버스 승차건수',
                 ha='center', fontsize=23, fontweight='bold')
        fig.text(.99, .055, '점 하나 = 격자 하나',
                 ha='right', fontsize=16, color='#666666')
    return fig
