"""第二版实验的实例、冻结参数与方法分派。"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from cvrp_core import Instance, generate_instance, solve
from cvrp_enhanced import EnhancedConfig, solve_enhanced

METHODS=('greedy','greedy_2opt','basic_tuned','improved_v1','stage2_selected','guided_v2')
NAMES={'greedy':'最近邻贪心','greedy_2opt':'贪心加2-opt',
       'basic_tuned':'基础退火（已调参）','improved_v1':'改进退火第一版',
       'stage2_selected':'第一版筛选设置','guided_v2':'邻近搜索退火第二版'}
DISTRIBUTIONS={'uniform':'均匀分布','clustered':'聚类分布','mixed':'混合分布'}


def make_instance(n, seed, distribution='uniform'):
    if distribution=='uniform':
        return generate_instance(n,seed)
    if distribution not in DISTRIBUTIONS:
        raise ValueError('未知客户分布。')
    if not isinstance(n,int) or isinstance(n,bool) or not 1<=n<=1000:
        raise ValueError('客户数必须为1至1000的整数。')
    if not isinstance(seed,int) or isinstance(seed,bool) or seed<0:
        raise ValueError('数据种子必须为非负整数。')
    rng=np.random.default_rng(seed)
    coordinates=[(50.,50.)];demands=[0]
    centers=np.array([(22.,22.),(78.,22.),(22.,78.),(78.,78.)])
    for i in range(n):
        if distribution=='mixed' and i%2==0:
            xy=rng.uniform(5,95,size=2)
        else:
            center=centers[int(rng.integers(0,4))]
            xy=np.clip(rng.normal(center,8,size=2),5,95)
        coordinates.append((float(xy[0]),float(xy[1])))
        demands.append(int(rng.integers(1,11)))
    instance=Instance(tuple(coordinates),tuple(demands),f'{distribution}_n{n}_seed{seed}')
    instance.validate(10)
    return instance


def load_settings(path=None):
    path=Path(path) if path is not None else Path(__file__).with_name('stage3_settings.json')
    settings=json.loads(path.read_text(encoding='utf-8'))
    EnhancedConfig(**settings['enhanced_config']).validate()
    return settings


def size_band(n):
    return min([20,50,100],key=lambda size:(abs(size-n),size))


def run_method(instance, capacity, method, settings, budget=8000, seed=2024):
    band=str(size_band(instance.n))
    if method=='guided_v2':
        return solve_enhanced(instance,capacity,budget,seed,EnhancedConfig(**settings['enhanced_config']))
    if method=='basic_tuned':
        result=solve(instance,capacity,'sa_original',budget=budget,seed=seed,**settings['basic_by_size'][band])
    elif method=='improved_v1':
        result=solve(instance,capacity,'sa_improved',budget=budget,seed=seed)
    elif method=='stage2_selected':
        result=solve(instance,capacity,'sa_improved',budget=budget,seed=seed,**settings['stage2_by_size'][band])
    elif method in ('greedy','greedy_2opt'):
        result=solve(instance,capacity,method,budget=budget,seed=seed)
    else:
        raise ValueError('未知方法。')
    result.method=method
    result.metadata['parameter_size_band']=int(band)
    return result
