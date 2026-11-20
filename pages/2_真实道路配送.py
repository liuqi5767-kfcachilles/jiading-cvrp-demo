"""原 app_v2.py 侧栏中的道路配送算例。"""

from dataclasses import asdict
import json

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from cvrptw import EPS
from road_cases import DEMO_SEED, ROAD_WINDOW_LABELS, compare_road_scenario, load_road_data, make_road_scenario
from road_plots import plot_road_plan
from timewindow_plots import clock, plot_schedule


st.set_page_config(page_title='真实道路配送', page_icon='🛣️', layout='wide')
st.title('真实道路配送')
st.caption('上海 · 同济大学嘉定校区周边 · 1 个模拟仓库，12 位模拟客户')
st.markdown('[© OpenStreetMap contributors](https://www.openstreetmap.org/copyright) · '
            '[ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/)')


@st.cache_data
def load_dataset():
    return load_road_data()


with st.sidebar:
    st.header('道路算例设置')
    with st.form('road_settings'):
        width = st.selectbox('收货时窗', ('wide', 'tight'), format_func=ROAD_WINDOW_LABELS.get,
                             key='road_width')
        seed = int(st.number_input('算法种子', 0, 999999, DEMO_SEED, key='road_seed'))
        budget = st.slider('候选检查上限', 500, 10000, 3000, 500, key='road_budget')
        submitted = st.form_submit_button('运行道路对照', type='primary')

data = load_dataset()
if submitted or 'road_run' not in st.session_state:
    try:
        scenario = make_road_scenario(data, width)
        with st.spinner('正在比较直线排程与有向道路排程…'):
            result = compare_road_scenario(scenario, budget, seed)
        st.session_state['road_run'] = (scenario, result, {'width': width, 'seed': seed, 'budget': budget})
        st.session_state.pop('road_vehicle', None)
    except (ValueError, RuntimeError) as exc:
        st.session_state.pop('road_run', None)
        st.error(str(exc))
        st.stop()

scenario, result, settings = st.session_state['road_run']
road = scenario.road
method_routes = {
    '直线排程（道路复核）': (result.direct_solution.routes, result.direct_road_check),
    '道路贪心插入': (result.road_initial.routes, result.road_initial.check),
    '道路邻近退火': (result.road_solution.routes, result.road_solution.check),
}
st.caption(f'{ROAD_WINDOW_LABELS[settings["width"]]} · 每车容量 {road.capacity} · '
           f'08:00 发车，10:00 前返仓 · 算法种子 {settings["seed"]}')
cards = st.columns(4)
cards[0].metric('直线排程按道路复核时迟到', f'{result.direct_road_check.late_customers} 位')
cards[1].metric('道路退火方案距离', f'{result.road_solution.check.distance:.2f} km')
cards[2].metric('道路退火方案车辆', f'{len(result.road_solution.routes)} 辆')
cards[3].metric('道路退火方案迟到', f'{result.road_solution.check.late_customers} 位')
st.success('道路退火方案通过客户覆盖、车载容量、收货窗口和返仓校验。')

st.dataframe(pd.DataFrame([
    {'方案': name, '车辆数': len(routes), '道路里程（km）': round(check.distance, 3),
     '迟到客户': check.late_customers, '晚返车辆': check.late_returns,
     '等待分钟': round(check.waiting, 2), '校验': '通过' if check.feasible else '未通过'}
    for name, (routes, check) in method_routes.items()
]), hide_index=True, width='stretch')
st.caption(f'直线排程在其自身模型下估计距离为 {result.direct_solution.check.distance:.2f} km。'
           '表中全部方案按同一组道路里程和行驶时间复核。')

map_tab, timeline_tab, matrix_tab = st.tabs(('道路路线对照', '服务时间轴', '矩阵与导出'))
with map_tab:
    panels = st.columns(2)
    for panel, name in zip(panels, ('直线排程（道路复核）', '道路邻近退火'), strict=True):
        with panel:
            st.markdown(f'**{name}**')
            fig = plot_road_plan(data, *method_routes[name])
            st.pyplot(fig, width='stretch')
            plt.close(fig)
    st.caption('浅蓝区域为嘉定校区，星标为模拟仓库，编号为客户，红色节点表示超时。')

with timeline_tab:
    first, second = st.columns(2)
    with first:
        method = st.selectbox('查看方案', ('道路邻近退火', '直线排程（道路复核）', '道路贪心插入'),
                              key='road_method')
    routes, checked = method_routes[method]
    with second:
        vehicle = st.selectbox('查看车辆', list(range(1, len(routes) + 1)),
                               format_func=lambda i: f'车辆 {i}', key='road_vehicle')
    trip = checked.vehicles[vehicle - 1]
    st.caption(f'访问顺序 {" → ".join(map(str, routes[vehicle - 1]))} · '
               f'载重 {trip.load}/{road.capacity} · 返仓 {clock(trip.return_time)}')
    fig = plot_schedule(road, trip)
    st.pyplot(fig, width='stretch')
    plt.close(fig)
    st.dataframe(pd.DataFrame([
        {'客户': s.customer, '最早': clock(road.windows[s.customer][0]),
         '最晚': clock(road.windows[s.customer][1]), '到达': clock(s.arrival),
         '等待（分）': round(s.waiting, 2), '开始服务': clock(s.service_start),
         '服务结束': clock(s.departure), '检查': '超时' if s.late_by > EPS else '通过'}
        for s in trip.stops
    ]), hide_index=True, width='stretch')

with matrix_tab:
    st.caption('行是出发节点，列是到达节点，0 表示仓库。里程和时间取自同一条估计最快的有向道路路径。')
    st.markdown('**道路距离（km）**')
    st.dataframe(pd.DataFrame(road.distances).round(3), width='stretch')
    st.markdown('**估计行驶时间（分钟）**')
    st.dataframe(pd.DataFrame(road.travel_minutes).round(2), width='stretch')
    with st.expander('算例来源与时间估计'):
        st.write(f'校区周边约 4.5 km × 5.0 km，包含 {len(data["nodes"])} 个道路节点、'
                 f'{len(data["edges"])} 条有向弧。行驶时间由路段速度静态估算。')
        snapshot = data['source']['snapshot_utc']
        st.markdown(f'数据截止时刻 `{snapshot}` · [Geofabrik 上海 OSM 快照]({data["source"]["url"]})')
    download = {'configuration': settings, 'source': data['source'],
                'road_problem': asdict(road), 'straight_problem': asdict(scenario.direct),
                'results': asdict(result)}
    st.download_button('下载道路实验与逐站结果 JSON',
                       json.dumps(download, ensure_ascii=False, indent=2),
                       file_name='road_delivery_experiment.json', mime='application/json')
