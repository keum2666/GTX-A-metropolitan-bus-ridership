"""015 일일·출근시간대 GTX역 연계버스 승차건수 카카오맵. 지도 파일 저장 없음."""
import html
import importlib.util
import json
from pathlib import Path
import sys
import time
import types
from urllib.parse import quote

import numpy as np
import pandas as pd

ORDER = ['운정중앙', '킨텍스', '대곡']
VIEW_ID = 'feeder_morning_counts_015'
COLORS = ['#E6E6E6', '#FEE0D2', '#FCAE91', '#FB6A4A', '#A50F15']
PERIODS = {
    'morning': {'column': 'GTX역행 2025 출근', 'label': '출근시간대',
                'time': '06:00 이상 10:00 미만', 'total': 646, 'positive': 38,
                'totals': [488, 41, 117], 'view_id': VIEW_ID},
    'day': {'column': 'GTX역행 2025', 'label': '일일',
            'time': '일일 전체', 'total': 1015, 'positive': 43,
            'totals': [672, 113, 230], 'view_id': 'feeder_daily_counts_015'},
}


def _period(grid):
    return PERIODS[grid.attrs.get('feeder_period', 'morning')]


def _helper(root):
    path = Path(root)/'code/03_eda/kakao_day_capture_015.py'
    spec = importlib.util.spec_from_file_location('_feeder_capture_helpers_015', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_data(root, period='morning'):
    if period not in PERIODS:
        raise ValueError("period는 'morning'(출근시간대) 또는 'day'(일일)입니다.")
    settings = PERIODS[period]
    root = Path(root)
    grid, stations, _ = _helper(root).load_map_data(root)
    reference = pd.read_csv(root/'results/analysis/feeder_bus/tables/H11a_격자별_공급구조_GTX역행.csv')
    grid = grid.merge(reference[['셀 ID', settings['column']]], on='셀 ID', validate='one_to_one')
    grid['feeder_count'] = grid[settings['column']].astype(int)
    grid.attrs['feeder_period'] = period
    assert len(grid) == 126 and grid['셀 ID'].is_unique
    assert grid.feeder_count.sum() == settings['total']
    assert grid.feeder_count.gt(0).sum() == settings['positive']
    assert grid.groupby('nearest_station').size().reindex(ORDER).tolist() == [63, 26, 37]
    assert grid.groupby('nearest_station').feeder_count.sum().reindex(ORDER).tolist() == settings['totals']
    return grid, stations


def make_payload(grid, stations, root):
    settings = _period(grid)
    # 기존 캡처 지도와 같은 실제 경계, 역사 좌표, 전체/주요지역 보기 범위를 재사용한다.
    source = _helper(root).make_payload(grid, stations)
    lookup = grid.set_index('셀 ID')
    # 0건은 별도 회색으로 표시하고, 세 권역의 양수 관측값을 합쳐 사분위수를 산출한다.
    # 같은 건수는 반드시 같은 색으로 유지한다.
    positive = grid.loc[grid.feeder_count.gt(0), 'feeder_count']
    edges = positive.quantile([.25, .5, .75]).to_numpy(dtype=float)
    assert np.all(np.diff(edges) > 0)
    cutoffs = np.floor(edges).astype(int)
    labels = ['0건', f'1–{cutoffs[0]}건',
              f'{cutoffs[0]+1}–{cutoffs[1]}건',
              f'{cutoffs[1]+1}–{cutoffs[2]}건', f'{cutoffs[2]+1}건 이상']
    result = {}
    for name in ORDER:
        item = source[name]
        features = []
        for original in item['features']:
            row = lookup.loc[original['id']]
            count = int(row.feeder_count)
            color_index = 0 if count == 0 else 1 + int(np.searchsorted(edges, count, side='left'))
            features.append({'id': original['id'], 'paths': original['paths'],
                             'count': count, 'color': COLORS[color_index]})
        result[name] = {'station': name, 'point': item['point'], 'bounds': item['bounds'],
                        'features': features, 'count': len(features),
                        'total': sum(f['count'] for f in features),
                        'quartileEdges': edges.tolist(), 'legendLabels': labels,
                        'periodLabel': settings['label'], 'timeLabel': settings['time']}
    assert sum(len(p['features']) for p in result.values()) == 126
    assert sum(p['total'] for p in result.values()) == settings['total']
    return result


def make_page(item, key):
    template = r'''<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>__NAME__역 연계버스 승차건수</title>
<style>*{box-sizing:border-box}body{margin:0;color:#111;background:white;font-family:Arial,'Apple SD Gothic Neo',sans-serif;font-weight:700}
.head{padding:12px 16px 5px;font-size:23px}.toolbar{padding:6px 16px 10px;display:flex;align-items:center;gap:9px;flex-wrap:wrap}
button{font-size:15px;padding:6px 11px;cursor:pointer;background:white;border:1px solid #888;border-radius:4px;font-weight:700}
#frame{height:calc(100vh - 140px);min-height:530px;position:relative}#map{height:100%;width:100%;background:#eee}
#status{position:absolute;top:12px;left:12px;background:white;padding:8px 12px;z-index:20;font-size:14px}
.legend{position:absolute;right:16px;top:16px;z-index:15;background:rgba(255,255,255,.96);padding:13px 17px;border:1px solid #555;box-shadow:0 1px 4px #999;font-size:17px;line-height:1.6}
.legend-title{font-size:18px;margin-bottom:7px}.item{display:flex;gap:12px;align-items:center}.swatch{width:24px;height:28px;border:1px solid #444;display:inline-block;flex-shrink:0}
.legend-scale .item{height:30px;line-height:30px}.legend-scale .swatch{height:30px;border:0}
.station-star{font-size:43px;color:#00865A;line-height:1;text-shadow:-2px -2px white,2px -2px white,-2px 2px white,2px 2px white}.green{color:#00865A;font-size:29px;width:24px;text-align:center}
.info{padding:10px 13px;white-space:nowrap;font-size:15px;line-height:1.7;font-weight:700}
.note{padding:9px 16px;color:#444;font-size:13px}#view-label{font-size:14px;color:#444}
body.capture .head,body.capture .toolbar,body.capture .note{display:none}body.capture #frame{height:100vh;min-height:0}
body.hide-legend .legend{display:none}
</style></head><body>
<div class="head">__NAME__역 권역 · __PERIOD__ GTX역 연계버스 승차건수</div>
<div class="toolbar"><button onclick="chooseView('all')">전체 격자 보기</button><button onclick="chooseView('focus')">주요지역 확대</button>
<button onclick="toggleCapture()">캡처 화면 (C)</button><button onclick="toggleLegend()">범례 숨기기/표시 (L)</button><span id="view-label">전체 __COUNT__개 · 승차 __TOTAL__건</span></div>
<div id="frame"><div id="map"></div><div id="status">카카오맵 불러오는 중…</div>
<div class="legend"><div class="legend-title">연계버스 승차건수</div>__LEGEND__
<div class="item"><i class="green">★</i>GTX-A역</div></div></div>
<div class="note">2025.11.06 · __TIME__ · 250 m 격자 · 0건 별도 표시 / 세 권역의 1건 이상 격자에 공통 사분위수 적용 · C: 캡처 화면 · L: 범례 숨기기/표시</div>
<script>
const data=__DATA__;let map=null,info=null;
function fail(msg){const s=document.getElementById('status');s.textContent=msg;s.style.display='block';s.style.color='#A22';}
function chooseView(view){if(!map)return;const b=data.bounds[view];map.setBounds(new kakao.maps.LatLngBounds(new kakao.maps.LatLng(b[1],b[0]),new kakao.maps.LatLng(b[3],b[2])),35,35,35,35);document.getElementById('view-label').textContent=(view==='all'?'전체 '+data.count+'개':'주요지역 확대 · 화면 밖에도 분석 격자 있음')+' · 승차 '+data.total+'건';if(info)info.close();}
function toggleCapture(){if(!map)return;const center=map.getCenter(),level=map.getLevel();document.body.classList.toggle('capture');info.close();requestAnimationFrame(()=>{map.relayout();map.setLevel(level);map.setCenter(center);});}
function toggleLegend(){document.body.classList.toggle('hide-legend');}
window.addEventListener('keydown',e=>{if(e.ctrlKey||e.metaKey||e.altKey)return;if(e.key.toLowerCase()==='c')toggleCapture();if(e.key.toLowerCase()==='l')toggleLegend();});
function initMap(){
 if(!window.kakao||!kakao.maps){fail('카카오 지도 SDK 연결을 확인해 주세요.');return;}
 map=new kakao.maps.Map(document.getElementById('map'),{center:new kakao.maps.LatLng(data.point[0],data.point[1]),level:6,draggable:true});
 info=new kakao.maps.InfoWindow({removable:true});
 for(const cell of data.features)for(const rings of cell.paths){
  const path=rings.map(ring=>ring.map(p=>new kakao.maps.LatLng(p[1],p[0])));
  const polygon=new kakao.maps.Polygon({map,path,strokeColor:'#333333',strokeWeight:1,strokeOpacity:1,fillColor:cell.color,fillOpacity:cell.count===0?.5:.82,zIndex:4});
  kakao.maps.event.addListener(polygon,'click',event=>{info.setContent('<div class="info"><b>격자 '+cell.id+'</b><br>'+data.periodLabel+' 연계버스 승차 '+cell.count+'건</div>');info.setPosition(event.latLng);info.open(map);});
 }
 new kakao.maps.CustomOverlay({map,position:new kakao.maps.LatLng(data.point[0],data.point[1]),content:'<span class="station-star" title="GTX-A '+data.station+'역">★</span>',xAnchor:.5,yAnchor:.5,zIndex:8});
 kakao.maps.event.addListener(map,'tilesloaded',()=>{document.getElementById('status').style.display='none';});
 map.setZoomable(true);chooseView('all');
}
window.addEventListener('resize',()=>{if(map){const center=map.getCenter();map.relayout();map.setCenter(center);}});
</script>
<script src="https://dapi.kakao.com/v2/maps/sdk.js?appkey=__KEY__&autoload=false" onload="kakao.maps.load(initMap)" onerror="fail('카카오 SDK 연결 실패: JavaScript 키와 SDK 도메인 '+location.origin+' 등록 여부를 확인해 주세요.')"></script>
</body></html>'''
    legend = '<div class="legend-scale">' + ''.join(
        f'<span class="item"><i class="swatch" style="background:{c}"></i>{label if i in (0,len(COLORS)-1) else ""}</span>'
        for i,(c,label) in enumerate(zip(COLORS,item['legendLabels']))) + '</div>'
    for marker,value in {'__NAME__':html.escape(item['station']), '__COUNT__':str(item['count']),
                         '__PERIOD__':html.escape(item['periodLabel']), '__TIME__':html.escape(item['timeLabel']),
                         '__TOTAL__':str(item['total']), '__LEGEND__':legend,
                         '__DATA__':json.dumps(item,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c'),
                         '__KEY__':quote(key,safe='')}.items():
        template = template.replace(marker,value)
    return template


def show_maps(grid, stations, root, key):
    from IPython.display import HTML, IFrame, display
    if not key.strip():
        raise ValueError('카카오 JavaScript 키를 입력해 주세요.')
    backend = sys.modules.get('kakao_grid_overlay')
    if backend is None:
        path = Path(root)/'code/03_eda/kakao_grid_overlay.py'
        backend = types.ModuleType('kakao_grid_overlay');backend.__file__=str(path)
        exec(compile(path.read_text(encoding='utf-8'),str(path),'exec'),backend.__dict__)
        sys.modules['kakao_grid_overlay'] = backend
    geo = stations.to_crs(4326).set_index('역명')
    coordinates = {name:(geo.loc[name].geometry.y,geo.loc[name].geometry.x) for name in ORDER}
    helper = _helper(root)
    view_id = _period(grid)['view_id']
    helper._start_available_server(backend, grid, coordinates, key, view_id=view_id)
    payload = make_payload(grid, stations, root)
    backend._pages[view_id] = {name:make_page(item,key) for name,item in payload.items()}
    urls = {name:f'http://localhost:{backend.PORT}/map?station={quote(name)}&view={view_id}&v={time.time_ns()}' for name in ORDER}
    for name,url in urls.items():
        display(HTML(f'<p><b>{name}역 권역</b> · <a href="{html.escape(url)}" target="_blank" rel="noopener">새 탭에서 크게 열기 · 직접 캡처</a></p>'))
        display(IFrame(url,width='100%',height=850))
    return urls
