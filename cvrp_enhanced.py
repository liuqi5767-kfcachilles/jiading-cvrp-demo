"""容量可行的邻近搜索、路线尾段交换和分段退火。保留第一版求解器。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
import time

from cvrp_core import (Instance, Result, Search, distance_matrix, greedy_solution,
                       route_load, total_distance, validate_routes)


@dataclass(frozen=True)
class EnhancedConfig:
    neighbor_count: int = 12
    cycles: int = 3
    temperature_scale: float = .1
    initial_descent_fraction: float = .2
    final_descent_fraction: float = .2
    use_nearby: bool = True
    use_tail_exchange: bool = True
    use_descent: bool = True

    def validate(self):
        if not isinstance(self.neighbor_count, int) or isinstance(self.neighbor_count, bool) or self.neighbor_count < 1:
            raise ValueError('neighbor_count必须为正整数。')
        if not isinstance(self.cycles, int) or isinstance(self.cycles, bool) or self.cycles < 1:
            raise ValueError('cycles必须为正整数。')
        if not math.isfinite(self.temperature_scale) or self.temperature_scale <= 0:
            raise ValueError('temperature_scale必须为有限正数。')
        if not (0 <= self.initial_descent_fraction < 1 and 0 <= self.final_descent_fraction < 1
                and self.initial_descent_fraction + self.final_descent_fraction < 1):
            raise ValueError('局部搜索预算比例必须非负，且两者之和小于1。')


def route_index(routes, demands):
    positions = {}
    loads = []
    for a, route in enumerate(routes):
        loads.append(route_load(route, demands))
        for p, node in enumerate(route[1:-1], 1):
            positions[node] = (a, p)
    return positions, loads


def move(routes, positions, loads, demands, capacity, i, j, kind):
    """构造一个容量可行方案；无效或完全不变时返回None。不调用目标函数。"""
    if i == j or i not in positions or j not in positions:
        return None
    a, pa = positions[i]; b, pb = positions[j]
    c = [r[:] for r in routes]
    if kind in ('relocate_before', 'relocate_after'):
        if a != b and loads[b] + demands[i] > capacity:
            return None
        c[a].pop(pa)
        target = pb - (1 if a == b and pa < pb else 0)
        c[b].insert(target + int(kind == 'relocate_after'), i)
    elif kind == 'swap':
        if a != b and (loads[a]-demands[i]+demands[j] > capacity or loads[b]-demands[j]+demands[i] > capacity):
            return None
        c[a][pa], c[b][pb] = j, i
    elif kind == 'two_opt':
        if a != b:
            return None
        lo, hi = sorted((pa, pb))
        c[a][lo:hi+1] = reversed(c[a][lo:hi+1])
    elif kind in ('tail_after', 'tail_before', 'tail_i_before', 'tail_j_before'):
        if a == b:
            return None
        ca = pa-1 if kind in ('tail_before','tail_i_before') else pa
        cb = pb-1 if kind in ('tail_before','tail_j_before') else pb
        r1 = routes[a][:ca+1] + routes[b][cb+1:]
        r2 = routes[b][:cb+1] + routes[a][ca+1:]
        if route_load(r1, demands) > capacity or route_load(r2, demands) > capacity:
            return None
        c[a], c[b] = r1, r2
    else:
        raise ValueError('未知邻域操作。')
    c = [r for r in c if len(r) > 2]
    return c if c != routes else None


def near_lists(distances, count):
    n = len(distances)-1
    return {i: sorted((j for j in range(1,n+1) if j != i),
                      key=lambda j: (distances[i][j],j))[:count] for i in range(1,n+1)}


def kinds_for_pair(same_route, tail):
    if same_route:
        return ('two_opt','relocate_before','relocate_after','swap')
    return ('relocate_before','relocate_after','swap','tail_after','tail_before','tail_i_before','tail_j_before') if tail else ('relocate_before','relocate_after','swap')


def neighbors(routes, demands, capacity, near, rng, config):
    positions, loads = route_index(routes, demands)
    order = list(positions); rng.shuffle(order)
    for i in order:
        targets = near[i][:] if config.use_nearby else [j for j in positions if j != i]
        # 随机化遍历顺序，避免每次重启都耗在相同客户上。
        rng.shuffle(targets)
        for j in targets:
            kinds = list(kinds_for_pair(positions[i][0] == positions[j][0], config.use_tail_exchange))
            rng.shuffle(kinds)
            for kind in kinds:
                candidate = move(routes,positions,loads,demands,capacity,i,j,kind)
                if candidate is not None:
                    yield candidate, kind


def local_search(routes, search, demands, capacity, near, rng, config, stop, stats):
    current = [r[:] for r in routes]
    cost = total_distance(current,search.distances)
    while search.evaluations < stop:
        improved = False
        for candidate, kind in neighbors(current,demands,capacity,near,rng,config):
            if search.evaluations >= stop:
                return current
            value = search.evaluate(candidate)
            stats['local_evaluations'] += 1
            stats['evaluated_by_kind'][kind] = stats['evaluated_by_kind'].get(kind,0)+1
            if value < cost-1e-9:
                current, cost, improved = candidate, value, True
                stats['local_improvements'] += 1
                break
        if not improved:
            return current
    return current


def propose(routes, demands, capacity, near, rng, config, stats):
    positions, loads = route_index(routes,demands)
    nodes = list(positions)
    if len(nodes) < 2:
        return [r[:] for r in routes], 'unchanged'
    # 预筛仅检查可行性，不比较距离；每轮最多12次尝试，避免死循环。
    for _ in range(12):
        stats['proposal_attempts'] += 1
        i = rng.choice(nodes)
        nearby = config.use_nearby and rng.random() < .85
        targets = near[i] if nearby else [j for j in nodes if j != i]
        j = rng.choice(targets)
        kinds = kinds_for_pair(positions[i][0] == positions[j][0],config.use_tail_exchange)
        kind = rng.choice(kinds)
        candidate = move(routes,positions,loads,demands,capacity,i,j,kind)
        if candidate is not None:
            return candidate, kind
        stats['constraint_or_identity_rejections'] += 1
    return [r[:] for r in routes], 'unchanged'


def solve_enhanced(instance: Instance, capacity: int, budget=8000, seed=2024,
                   config: EnhancedConfig | None = None):
    config = config or EnhancedConfig()
    config.validate(); instance.validate(capacity)
    if not isinstance(budget,int) or isinstance(budget,bool) or budget < 1:
        raise ValueError('评价预算必须为正整数。')
    start = time.perf_counter()
    rng = random.Random(seed)
    distances = distance_matrix(instance)
    initial = greedy_solution(instance,capacity,distances)
    initial_cost = total_distance(initial,distances)
    search = Search(distances,initial,budget)
    near = near_lists(distances,config.neighbor_count)
    stats = {'proposal_attempts':0,'constraint_or_identity_rejections':0,
             'unchanged_evaluations':0,'local_evaluations':0,'local_improvements':0,
             'annealing_evaluations':0,'accepted_uphill':0,'evaluated_by_kind':{}}
    stage_boundaries = []
    if config.use_descent:
        stop = int(budget*config.initial_descent_fraction)
        local_search(initial,search,instance.demands,capacity,near,rng,config,stop,stats)
    stage_boundaries.append({'phase':'initial_local','evaluations':search.evaluations})
    remaining = budget-search.evaluations
    final_reserved = int(budget*config.final_descent_fraction) if config.use_descent else 0
    annealing_steps = max(0,remaining-final_reserved)
    # 温度随问题距离尺度变化；无需额外试探性候选评价。
    temperature = max(1e-8, config.temperature_scale*initial_cost/max(1,instance.n))
    for cycle in range(config.cycles):
        steps = annealing_steps//config.cycles + int(cycle < annealing_steps%config.cycles)
        current = [r[:] for r in search.best]; cost = search.best_cost
        for step in range(steps):
            t = temperature * .01**(step/max(1,steps-1))
            candidate, kind = propose(current,instance.demands,capacity,near,rng,config,stats)
            value = search.evaluate(candidate)
            stats['annealing_evaluations'] += 1
            stats['evaluated_by_kind'][kind] = stats['evaluated_by_kind'].get(kind,0)+1
            stats['unchanged_evaluations'] += int(kind=='unchanged')
            delta = value-cost
            if delta <= 0 or rng.random() < math.exp(-delta/t):
                stats['accepted_uphill'] += int(delta>1e-9)
                current,cost=candidate,value
        stage_boundaries.append({'phase':f'annealing_cycle_{cycle+1}','evaluations':search.evaluations})
    if config.use_descent and search.evaluations < budget:
        local_search(search.best,search,instance.demands,capacity,near,rng,config,budget,stats)
    stage_boundaries.append({'phase':'final_local','evaluations':search.evaluations})
    errors = validate_routes(search.best,instance,capacity)
    if errors:
        raise RuntimeError('第二版路线校验失败：'+'；'.join(errors))
    metadata={'budget':budget,'data_fingerprint':instance.fingerprint,'config':asdict(config),
              'initial_temperature':temperature,'end_temperature_ratio':.01,
              'stage_boundaries':stage_boundaries,'stats':stats,
              'budget_definition':'所有完整候选距离评价合计，包括局部搜索、所有退火周期和不变候选；容量预筛不比较目标，所有耗时均计入。'}
    return Result('sa_guided_v2',search.best,search.best_cost,initial_cost,
                  time.perf_counter()-start,search.evaluations,search.history,True,seed,metadata)
