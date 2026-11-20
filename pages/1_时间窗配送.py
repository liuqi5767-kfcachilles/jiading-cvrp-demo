"""通过 app_v2.py 的导航进入的课程项目拓展页。"""

from dataclasses import asdict
import json

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from cvrptw import EPS
from plots import plot_routes
from timewindow_cases import METHOD_LABELS, WINDOW_LABELS, compare, hand_example, simulated_case, stop_rows
from timewindow_plots import clock, plot_schedule


st.set_page_config(page_title='时间窗配送', page_icon='🕒', layout='wide')
st.title('带收货时间窗的配送规划')
st.caption('单仓库 · 车辆有载重上限 · 客户在指定时段开始收货 · 车辆按时返仓')

with st.sidebar:
    st.header('时间窗设置')
    source = st.selectbox('算例', ('手算例（4 位客户）', '模拟客户'), key='tw_source')
    with st.form('tw_settings'):
        if source == '模拟客户':
            n = st.slider('客户数', 5, 60, 20, 5, key='tw_n')
            capacity = st.slider('车辆容量', 10, 60, 30, 5, key='tw_capacity')
            width = st.selectbox('收货时间窗', list(WINDOW_LABELS), index=1,
                                 format_func=WINDOW_LABELS.get, key='tw_width')
            data_seed = int(st.number_input('客户数据种子', 0, 999999, 42, key='tw_data_seed'))
        else:
            st.caption('08:00 发车。客户 2 只在 08:06–08:08 开始收货。')
        algorithm_seed = int(st.number_input('算法种子', 0, 999999, 2024, key='tw_algorithm_seed'))
        budget = st.slider('候选检查上限', 500, 10000, 3000, 500, key='tw_budget')
        submitted = st.form_submit_button('运行时间窗对照', type='primary')

if submitted or 'tw_run' not in st.session_state:
    try:
        if source == '模拟客户':
            problem = simulated_case(n, capacity, data_seed, width).problem
            title = f'模拟客户 · {WINDOW_LABELS[width]}'
            case_settings = {'scenario': 'simulation', 'data_seed': data_seed, 'window': width}
        else:
            problem = hand_example()
            title = '手算例 · 紧急收货客户'
            case_settings = {'scenario': 'hand_example'}
        with st.spinner('正在构造可行路线并逐站检查时间窗…'):
            results = compare(problem, budget, algorithm_seed)
        st.session_state['tw_run'] = (problem, results, title, {
            **case_settings, 'budget': budget, 'algorithm_seed': algorithm_seed,
            'customers': problem.instance.n, 'capacity': problem.capacity,
        })
        st.session_state.pop('tw_vehicle', None)
    except (ValueError, RuntimeError) as exc:
        st.session_state.pop('tw_run', None)
        st.error(str(exc))
        st.stop()

problem, results, title, case_settings = st.session_state['tw_run']
ordinary, initial, best = (results[key] for key in ('capacity', 'insertion', 'annealing'))
depot_start, depot_end = problem.windows[0]
st.subheader(title)
st.caption(f'客户 {problem.instance.n} 位 · 每车容量 {problem.capacity} · '
           f'{clock(depot_start)[:5]} 发车，{clock(depot_end)[:5]} 前返仓 · 时间和里程为模拟数据')

metrics = st.columns(4)
metrics[0].metric('容量版超时客户', f'{ordinary.check.late_customers} 位')
metrics[1].metric('时间窗方案距离', f'{best.check.distance:.2f}')
metrics[2].metric('时间窗方案车辆', f'{len(best.routes)} 辆')
metrics[3].metric('时间窗方案累计等待', f'{best.check.waiting:.1f} 分钟')
st.success('时间窗方案已通过客户覆盖、车辆容量、收货时间和返仓校验。')

st.dataframe(pd.DataFrame([
    {'方案': METHOD_LABELS[method], '总距离': round(result.check.distance, 2),
     '车辆数': len(result.routes), '超时客户': result.check.late_customers,
     '超时返仓车辆': result.check.late_returns, '等待分钟': round(result.check.waiting, 1),
     '可行性': '通过' if result.check.feasible else '未通过'}
    for method, result in results.items()
]), hide_index=True, width='stretch')
st.caption(f'时间窗退火检查了 {best.evaluations} 个候选，其中 {best.rejected_for_time} 个未通过时间检查。'
           '容量版路线保留原行驶顺序，超时项来自将其套入同一批时间窗后的复核。')

routes_tab, clock_tab, data_tab = st.tabs(('路线对照', '服务时间轴', '客户数据与导出'))
with routes_tab:
    left, right = st.columns(2)
    for panel, label, result, heading in (
        (left, '容量版第二版', ordinary, 'Capacity-only plan'),
        (right, '时间窗邻近退火', best, 'Time-window plan'),
    ):
        with panel:
            st.markdown(f'**{label}**')
            fig = plot_routes(problem.instance, result.routes, problem.distances, heading,
                              show_labels=problem.instance.n <= 25)
            st.pyplot(fig, width='stretch')
            plt.close(fig)
    if not ordinary.check.feasible:
        st.info('容量版的距离可作参照。含超时客户或超时返仓的路线无法按当前时窗执行。')

with clock_tab:
    a, b = st.columns(2)
    with a:
        method = st.selectbox('查看方案', ('annealing', 'insertion', 'capacity'),
                              format_func=METHOD_LABELS.get, key='tw_method')
    selected = results[method]
    with b:
        vehicle = st.selectbox('查看车辆', list(range(1, len(selected.routes) + 1)),
                               format_func=lambda i: f'车辆 {i}', key='tw_vehicle')
    schedule = selected.check.vehicles[vehicle - 1]
    st.caption(f'访问顺序 {" → ".join(map(str, selected.routes[vehicle - 1]))} · '
               f'载重 {schedule.load}/{problem.capacity} · 返仓 {clock(schedule.return_time)}')
    if schedule.return_late_by > EPS:
        st.warning(f'该车晚于返仓截止时刻 {schedule.return_late_by:.1f} 分钟。')
    fig = plot_schedule(problem, schedule)
    st.pyplot(fig, width='stretch')
    plt.close(fig)
    st.caption('浅蓝区域为允许开始服务的时间窗。圆点是到达时刻，橙色为等待，绿色为服务，红色为迟到后的服务。')
    st.dataframe(pd.DataFrame([
        {'客户': stop.customer, '收货开始': clock(problem.windows[stop.customer][0]),
         '收货截止': clock(problem.windows[stop.customer][1]), '到达': clock(stop.arrival),
         '等待（分）': round(stop.waiting, 2), '开始服务': clock(stop.service_start),
         '服务结束': clock(stop.departure),
         '检查': '超时' if stop.late_by > EPS else '通过'}
        for stop in schedule.stops
    ]), hide_index=True, width='stretch')

with data_tab:
    st.dataframe(pd.DataFrame([
        {'节点': customer, 'X': round(x, 3), 'Y': round(y, 3),
         '需求': problem.instance.demands[customer],
         '最早': clock(problem.windows[customer][0]), '最晚': clock(problem.windows[customer][1]),
         '服务时长（分）': problem.service_minutes[customer]}
        for customer, (x, y) in enumerate(problem.instance.coordinates)
    ]), hide_index=True, width='stretch')
    st.download_button('下载时间窗方案的逐站明细 CSV',
                       pd.DataFrame(stop_rows(best)).to_csv(index=False).encode('utf-8-sig'),
                       file_name='time_window_schedule.csv', mime='text/csv')
    payload = {'settings': case_settings, 'problem': asdict(problem),
               'plans': {key: asdict(result) for key, result in results.items()}}
    st.download_button('下载完整算例与结果 JSON', json.dumps(payload, ensure_ascii=False, indent=2),
                       file_name='time_window_experiment.json', mime='application/json')
