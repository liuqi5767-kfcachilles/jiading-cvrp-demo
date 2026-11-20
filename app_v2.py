"""本地运行：python -m streamlit run app_v2.py"""
import io
import json
from dataclasses import asdict

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from cvrp_core import (LABELS, METHODS, distance_matrix, generate_instance,
                       instance_csv, read_instance_csv, result_record, route_load,
                       solve, solve_exact, total_distance, validate_routes)
from plots import plot_routes
from stage3_common import (METHODS, NAMES as LABELS, DISTRIBUTIONS,
                           load_settings, make_instance, run_method, size_band)

settings=load_settings()
ENGLISH={'greedy':'Greedy', 'greedy_2opt':'Greedy + 2-opt',
         'basic_tuned':'Tuned basic SA', 'improved_v1':'Improved SA v1',
         'stage2_selected':'Selected v1', 'guided_v2':'Guided SA v2'}

def plot_history(results):
    fig,ax=plt.subplots(figsize=(9,3.8))
    for method in ['basic_tuned','improved_v1','stage2_selected','guided_v2']:
        r=results[method]
        ax.plot(range(len(r.history)+1),[r.initial_distance]+r.history,
                label=ENGLISH[method],lw=1.6)
    ax.set_xlabel('Candidate evaluations'); ax.set_ylabel('Best distance')
    ax.legend(frameon=False,ncols=2); ax.grid(alpha=.2);fig.tight_layout()
    return fig

st.set_page_config(page_title='车辆路径规划实验系统',page_icon='🚚',layout='wide')
st.title('车辆路径规划实验系统 · 第二版')
st.caption('容量约束车辆路径规划 · 最近邻贪心与模拟退火 · 距离以模拟单位计量')

with st.sidebar:
    st.header('实验设置')
    with st.form('experiment_settings'):
        source=st.selectbox('数据来源',['随机生成','导入CSV'],key='data_source')
        distribution=st.selectbox('客户分布',list(DISTRIBUTIONS),format_func=lambda x:DISTRIBUTIONS[x],key='distribution')
        n=st.slider('客户数量',5,100,20,5,key='customer_count')
        capacity=st.slider('车辆容量',10,100,30,5,key='vehicle_capacity')
        data_seed=st.number_input('数据随机种子',0,999999,42,key='data_seed')
        algorithm_seed=st.number_input('算法随机种子',0,999999,2024,key='algorithm_seed')
        budget=st.slider('候选方案评价上限',1000,30000,8000,1000,key='budget')
        uploaded=st.file_uploader('客户CSV  导入模式使用',type=['csv'],key='customer_csv')
        st.caption('修改参数后点击运行。改变容量不会改变随机客户的坐标与需求。')
        submitted=st.form_submit_button('生成数据并运行六种方案',type='primary')

if submitted or 'experiment_v2' not in st.session_state:
    try:
        if source=='导入CSV':
            if uploaded is None:
                raise ValueError('请先选择客户CSV文件。')
            instance=read_instance_csv(uploaded.getvalue().decode('utf-8-sig'),uploaded.name)
            if instance.n>100:
                raise ValueError('交互演示最多支持100个客户。')
        else:
            instance=make_instance(int(n),int(data_seed),distribution)
        instance.validate(int(capacity))
        cfg={'source':source,'customers':instance.n,'capacity':int(capacity),
             'data_seed':int(data_seed) if source=='随机生成' else None,
             'algorithm_seed':int(algorithm_seed),'budget':int(budget),
             'distribution':distribution if source=='随机生成' else 'CSV',
             'frozen_settings':settings,'parameter_size_band':size_band(instance.n),
             'data_fingerprint':instance.fingerprint}
        with st.spinner('正在求解并检查所有路线…'):
            results={m:run_method(instance,cfg['capacity'],m,settings,budget=cfg['budget'],
                                  seed=cfg['algorithm_seed']) for m in METHODS}
        st.session_state.experiment_v2={'instance':instance,'config':cfg,'results':results}
        st.session_state.pop('exact_reference',None)
    except (ValueError,UnicodeDecodeError,RuntimeError) as exc:
        st.error(str(exc))

if 'experiment_v2' not in st.session_state:
    st.info('请设置数据后开始运行。'); st.stop()

experiment=st.session_state.experiment_v2
instance,cfg,results=experiment['instance'],experiment['config'],experiment['results']
cap=cfg['capacity']  # All displayed values use this successful run snapshot.
d=distance_matrix(instance)
st.caption(f"本次结果：分布{DISTRIBUTIONS.get(cfg['distribution'],cfg['distribution'])} · 客户{instance.n}个 · 容量{cap} · 数据种子{cfg['data_seed']} · "
           f"算法种子{cfg['algorithm_seed']} · 评价上限{cfg['budget']} · 数据编号{instance.fingerprint}")
selected=st.selectbox('查看方案',list(METHODS),index=5,format_func=lambda key:LABELS[key],key='display_method')
result=results[selected]
columns=st.columns(4)
columns[0].metric('贪心初始距离',f'{result.initial_distance:.2f}')
columns[1].metric('当前方案总距离',f'{result.distance:.2f}')
columns[2].metric('相对贪心改善率',f'{result.improvement_pct:.2f}%')
columns[3].metric('使用车辆数',len(result.routes))
errors=validate_routes(result.routes,instance,cap)
if errors:
    st.error('校验未通过：'+'；'.join(errors))
else:
    st.success('已重新校验：每个客户恰好访问一次，所有路线从仓库出发并返回，所有车辆均不超载。')
st.info(f'求解耗时 {result.elapsed_seconds:.3f} 秒 · 候选评价 {result.evaluations} 次 · '
        f'平均装载率 {sum(instance.demands)/(cap*len(result.routes)):.1%}')

summary=pd.DataFrame([{'算法':LABELS[m],'总距离':r.distance,'改善率百分比':r.improvement_pct,
                        '车辆数':len(r.routes),'求解时间秒':r.elapsed_seconds,'候选评价次数':r.evaluations,
                        '约束校验':'通过' if not validate_routes(r.routes,instance,cap) else '未通过'}
                       for m,r in results.items()])
st.subheader('六种方案在相同客户数据上的结果')
st.dataframe(summary.round(4),hide_index=True,width='stretch')
route_tab,export_tab,exact_tab,notes_tab=st.tabs(['路线与搜索过程','车辆明细与导出','小规模最优校验','算法说明'])
with route_tab:
    labels=st.checkbox('显示客户编号',value=instance.n<=30)
    left,right=st.columns(2)
    with left:
        fig=plot_routes(instance,results['greedy'].routes,d,'Greedy baseline',labels)
        st.pyplot(fig,width='stretch'); plt.close(fig)
    with right:
        fig=plot_routes(instance,result.routes,d,ENGLISH[selected],labels)
        st.pyplot(fig,width='stretch'); plt.close(fig)
    fig=plot_history(results)
    st.pyplot(fig,width='stretch'); plt.close(fig)
    st.caption('曲线为历史最好距离；变平表示本次搜索暂未找到更好方案，不能据此证明全局最优。')

with export_tab:
    records=[]
    for k,route in enumerate(result.routes,1):
        load=route_load(route,instance.demands)
        records.append({'车辆':k,'访问顺序':' → '.join(map(str,route)),
                        '配送量':load,'车辆容量':cap,'装载率百分比':100*load/cap,
                        '行驶距离':total_distance([route],d),'是否超载':load>cap})
    route_df=pd.DataFrame(records)
    st.dataframe(route_df.round(3),hide_index=True,width='stretch')
    customer_df=pd.DataFrame([{'id':i,'x':xy[0],'y':xy[1],'demand':q}
                             for i,(xy,q) in enumerate(zip(instance.coordinates,instance.demands))])
    st.dataframe(customer_df,hide_index=True,width='stretch')
    st.download_button('下载客户数据 CSV',instance_csv(instance).encode('utf-8-sig'),'customers.csv','text/csv')
    st.download_button('下载算法对比 CSV',summary.to_csv(index=False).encode('utf-8-sig'),'algorithm_comparison.csv','text/csv')
    st.download_button('下载当前车辆明细 CSV',route_df.to_csv(index=False).encode('utf-8-sig'),'routes.csv','text/csv')
    payload={'configuration':cfg,'instance':asdict(instance),
             'results':{m:asdict(r) for m,r in results.items()}}
    st.download_button('下载完整实验 JSON',json.dumps(payload,ensure_ascii=False,indent=2),
                       'experiment_v2.json','application/json')

with exact_tab:
    st.write('不超过12个客户时，可通过子集动态规划计算同一模型的精确最优解。')
    if st.button('计算当前实例的精确最优解',disabled=instance.n>12):
        with st.spinner('正在进行精确校验…'):
            st.session_state.exact_reference=solve_exact(instance,cap)
    if 'exact_reference' in st.session_state:
        exact=st.session_state.exact_reference
        st.metric('精确最优距离',f'{exact.distance:.4f}')
        st.dataframe(pd.DataFrame([{'算法':LABELS[m],
                                   '距最优差距百分比':max(0.0,100*(r.distance-exact.distance)/exact.distance) if exact.distance else 0.0}
                                  for m,r in results.items()]).round(4),hide_index=True,width='stretch')
    elif instance.n>12:
        st.caption('将客户数设置为5或10并运行后，可以启用精确校验。')

with notes_tab:
    st.write('最近邻贪心和贪心加2-opt提供可解释的起点。基础退火采用第二阶段已选定的温度参数。')
    st.write('第一版保留原改进方法；第一版筛选设置按第二阶段开发集结果打开或关闭局部搜索、路线拆分。')
    st.write('第二版优先在附近客户之间尝试移动、交换和两条路线的尾段交换，结合局部搜索与共享预算的退火周期。')
    st.write('相同候选评价上限不代表相同耗时，表中同时展示距离、车辆数和实际求解时间。当前目标只最小化总距离。')
    band=str(cfg['parameter_size_band'])
    basic=cfg['frozen_settings']['basic_by_size'][band]
    st.caption(f"基础退火使用{band}客户档参数：初温{basic['initial_temperature']:g}，降温系数{basic['cooling_rate']:g}。参数在容量30的20、50、100客户开发数据上选定，其他客户数使用最近的规模档。")
    with st.expander('查看本次第二版的搜索设置与评价统计'):
        st.json(results['guided_v2'].metadata)
    st.caption('CSV列为id、x、y、demand，包含0号仓库及连续客户编号；坐标距离为模拟单位。')
