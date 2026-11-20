"""容量约束车辆路径规划：数据、校验、四种启发式与小规模精确求解。"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import random
import time
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

Routes = list[list[int]]
METHODS = ('greedy', 'greedy_2opt', 'sa_original', 'sa_improved')
LABELS = {
    'greedy': '最近邻贪心', 'greedy_2opt': '贪心加2-opt',
    'sa_original': '基础模拟退火', 'sa_improved': '改进模拟退火',
}


@dataclass(frozen=True)
class Instance:
    coordinates: tuple[tuple[float, float], ...]
    demands: tuple[int, ...]
    name: str = 'instance'
    edge_weight_type: str = 'EUCLIDEAN'

    @property
    def n(self) -> int:
        return len(self.demands) - 1

    @property
    def fingerprint(self) -> str:
        # Capacity is intentionally excluded so capacity experiments share data.
        data = [self.coordinates, self.demands]
        if self.edge_weight_type != 'EUCLIDEAN':
            data.append(self.edge_weight_type)
        raw = json.dumps(data, separators=(',', ':'))
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def validate(self, capacity: int) -> None:
        if self.edge_weight_type not in ('EUCLIDEAN', 'EUC_2D'):
            raise ValueError('不支持该距离计量方式。')
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity <= 0:
            raise ValueError('车辆容量必须为正整数。')
        if self.n < 1 or len(self.coordinates) != len(self.demands):
            raise ValueError('至少需要一个客户，且坐标与需求数量必须一致。')
        if self.demands[0] != 0:
            raise ValueError('0号仓库的需求必须为0。')
        for point in self.coordinates:
            if len(point) != 2 or not all(math.isfinite(x) for x in point):
                raise ValueError('所有坐标必须是有限数值。')
        if any(not isinstance(q, int) or isinstance(q, bool) or q <= 0 for q in self.demands[1:]):
            raise ValueError('客户需求必须为正整数。')
        if max(self.demands) > capacity:
            raise ValueError('存在客户需求超过车辆容量；本模型不允许拆分客户需求。')


def generate_instance(n: int = 20, seed: int = 42) -> Instance:
    """需求固定在1至10，与车辆容量无关；前缀与原程序容量>=11时一致。"""
    if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= 1000:
        raise ValueError('客户数量必须是1至1000之间的整数。')
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError('数据种子必须为非负整数。')
    rng = np.random.default_rng(seed)
    xy = [(50.0, 50.0)]
    demands = [0]
    for _ in range(n):
        xy.append((float(rng.uniform(5, 95)), float(rng.uniform(5, 95))))
        demands.append(int(rng.integers(1, 11)))
    return Instance(tuple(xy), tuple(demands), f'n{n}_seed{seed}')


def distance_matrix(instance: Instance) -> list[list[float]]:
    distances = [[math.hypot(a[0]-b[0], a[1]-b[1]) for b in instance.coordinates]
                 for a in instance.coordinates]
    if instance.edge_weight_type == 'EUC_2D':
        return [[math.floor(distance + .5) for distance in row] for row in distances]
    return distances


def total_distance(routes: Routes, distances: list[list[float]]) -> float:
    return sum(distances[a][b] for route in routes for a, b in zip(route, route[1:]))


def route_load(route: list[int], demands: tuple[int, ...]) -> int:
    return sum(demands[i] for i in route[1:-1])


def validate_routes(routes: Routes, instance: Instance, capacity: int) -> list[str]:
    """返回全部已发现的问题，空列表表示通过。"""
    instance.validate(capacity)
    errors, seen = [], []
    if not routes:
        return ['没有配送路线。']
    for k, route in enumerate(routes, 1):
        if len(route) < 3 or route[0] != 0 or route[-1] != 0:
            errors.append(f'路线{k}必须从仓库出发并返回，且至少服务一个客户。')
            continue
        if any(not isinstance(i, int) or isinstance(i, bool) or not 1 <= i <= instance.n for i in route[1:-1]):
            errors.append(f'路线{k}存在非法客户编号或中途仓库节点。')
            continue
        seen.extend(route[1:-1])
        if route_load(route, instance.demands) > capacity:
            errors.append(f'路线{k}超载。')
    if sorted(seen) != list(range(1, instance.n+1)):
        errors.append('客户存在遗漏或重复访问。')
    return errors


def greedy_solution(instance: Instance, capacity: int, distances=None) -> Routes:
    instance.validate(capacity)
    d = distances if distances is not None else distance_matrix(instance)
    remaining = set(range(1, instance.n+1))
    routes = []
    while remaining:
        route, load = [0], 0
        while True:
            candidates = [i for i in remaining if load + instance.demands[i] <= capacity]
            if not candidates:
                break
            nxt = min(candidates, key=lambda i: d[route[-1]][i])
            route.append(nxt)
            remaining.remove(nxt)
            load += instance.demands[nxt]
        routes.append(route + [0])
    return routes


def random_neighbor(routes: Routes, demands, capacity: int, rng: random.Random,
                    allow_split: bool = False) -> Routes:
    """三种基础邻域保持原实现的随机抽取顺序；改进版另允许路线拆分。"""
    ops = ['two_opt', 'relocate', 'swap']
    if allow_split:
        ops = ops * 2 + ['split']
    op = rng.choice(ops)
    c = [r[:] for r in routes]
    if op == 'two_opt':
        valid = [k for k, r in enumerate(c) if len(r) >= 4]
        if valid:
            k = rng.choice(valid)
            left, right = sorted(rng.sample(range(1, len(c[k])-1), 2))
            c[k][left:right+1] = reversed(c[k][left:right+1])
    elif op == 'relocate':
        valid = [k for k, r in enumerate(c) if len(r) > 2]
        if valid:
            a = rng.choice(valid)
            pos = rng.randint(1, len(c[a])-2)
            customer = c[a].pop(pos)
            targets = list(range(len(c)))
            if len(targets) > 1:
                targets.remove(a)
            b = rng.choice(targets)
            insert = rng.randint(1, len(c[b])-1)
            if route_load(c[b], demands) + demands[customer] <= capacity:
                c[b].insert(insert, customer)
            else:
                c[a].insert(pos, customer)
            c = [r for r in c if len(r) > 2]
    elif op == 'swap':
        valid = [k for k, r in enumerate(c) if len(r) > 2]
        if len(valid) >= 2:
            a, b = rng.sample(valid, 2)
            pa, pb = rng.randint(1, len(c[a])-2), rng.randint(1, len(c[b])-2)
            ia, ib = c[a][pa], c[b][pb]
            if (route_load(c[a], demands)-demands[ia]+demands[ib] <= capacity
                    and route_load(c[b], demands)-demands[ib]+demands[ia] <= capacity):
                c[a][pa], c[b][pb] = ib, ia
    else:
        valid = [k for k, r in enumerate(c) if len(r) >= 4]
        if valid:
            k = rng.choice(valid)
            nodes = c[k][1:-1]
            cut = rng.randint(1, len(nodes)-1)
            c[k] = [0] + nodes[:cut] + [0]
            c.append([0] + nodes[cut:] + [0])
    return c


@dataclass
class Search:
    distances: list[list[float]]
    best: Routes
    limit: int
    evaluations: int = 0
    best_cost: float = field(init=False)
    history: list[float] = field(default_factory=list)

    def __post_init__(self):
        self.best = [r[:] for r in self.best]
        self.best_cost = total_distance(self.best, self.distances)

    def evaluate(self, routes: Routes) -> float:
        if self.evaluations >= self.limit:
            raise RuntimeError('候选评价预算已耗尽。')
        cost = total_distance(routes, self.distances)
        self.evaluations += 1
        if cost < self.best_cost - 1e-10:
            self.best, self.best_cost = [r[:] for r in routes], cost
        self.history.append(self.best_cost)
        return cost


def local_neighbors(routes: Routes, demands, capacity: int,
                    kinds: tuple[str, ...], rng=None) -> Iterator[Routes]:
    order = list(range(len(routes)))
    if rng:
        rng.shuffle(order)
    loads = [route_load(r, demands) for r in routes]
    for kind in kinds:
        if kind == 'two_opt':
            for a in order:
                for left in range(1, len(routes[a])-2):
                    for right in range(left+1, len(routes[a])-1):
                        c = [r[:] for r in routes]
                        c[a][left:right+1] = reversed(c[a][left:right+1])
                        yield c
        elif kind == 'relocate':
            for a in order:
                for pa in range(1, len(routes[a])-1):
                    customer = routes[a][pa]
                    for b in order:
                        if a == b or loads[b] + demands[customer] > capacity:
                            continue
                        for pb in range(1, len(routes[b])):
                            c = [r[:] for r in routes]
                            c[a].pop(pa)
                            c[b].insert(pb, customer)
                            yield [r for r in c if len(r) > 2]
        elif kind == 'swap':
            for ai, a in enumerate(order):
                for b in order[ai+1:]:
                    for pa in range(1, len(routes[a])-1):
                        ia = routes[a][pa]
                        for pb in range(1, len(routes[b])-1):
                            ib = routes[b][pb]
                            if (loads[a]-demands[ia]+demands[ib] <= capacity
                                    and loads[b]-demands[ib]+demands[ia] <= capacity):
                                c = [r[:] for r in routes]
                                c[a][pa], c[b][pb] = ib, ia
                                yield c


def descent(routes: Routes, search: Search, demands, capacity: int, stop: int,
            kinds=('two_opt', 'relocate', 'swap'), rng=None) -> Routes:
    current = [r[:] for r in routes]
    cost = total_distance(current, search.distances)
    while search.evaluations < min(stop, search.limit):
        improved = False
        for candidate in local_neighbors(current, demands, capacity, kinds, rng):
            if search.evaluations >= min(stop, search.limit):
                return current
            value = search.evaluate(candidate)
            if value < cost - 1e-9:
                current, cost, improved = candidate, value, True
                break
        if not improved:
            break
    return current


@dataclass
class Result:
    method: str
    routes: Routes
    distance: float
    initial_distance: float
    elapsed_seconds: float
    evaluations: int
    history: list[float]
    feasible: bool
    seed: int
    metadata: dict = field(default_factory=dict)

    @property
    def improvement_pct(self):
        return 100*(self.initial_distance-self.distance)/self.initial_distance if self.initial_distance else 0.0


def solve(instance: Instance, capacity: int, method='sa_improved', budget=8000,
          seed=2024, cooling_rate=.995, initial_temperature=100.0,
          minimum_temperature=.01, enable_descent=True, allow_split=True) -> Result:
    if method not in METHODS:
        raise ValueError('未知算法。')
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1:
        raise ValueError('评价预算必须为正整数。')
    if not 0 < cooling_rate < 1:
        raise ValueError('降温系数必须在0和1之间。')
    if not 0 < minimum_temperature < initial_temperature or not math.isfinite(initial_temperature):
        raise ValueError('温度必须满足0 < 最低温度 < 初始温度。')
    instance.validate(capacity)
    start = time.perf_counter()
    d = distance_matrix(instance)
    initial = greedy_solution(instance, capacity, d)
    initial_cost = total_distance(initial, d)
    sc = Search(d, initial, budget)
    rng = random.Random(seed)
    metadata = {'budget': budget, 'data_fingerprint': instance.fingerprint}
    if method == 'greedy_2opt':
        descent(initial, sc, instance.demands, capacity, budget, kinds=('two_opt',))
    elif method == 'sa_original':
        current, cost, temperature = [r[:] for r in initial], initial_cost, initial_temperature
        while temperature > minimum_temperature and sc.evaluations < budget:
            for _ in range(80):
                if sc.evaluations >= budget:
                    break
                candidate = random_neighbor(current, instance.demands, capacity, rng)
                value = sc.evaluate(candidate)
                delta = value - cost
                if delta <= 0 or rng.random() < math.exp(-delta/temperature):
                    current, cost = candidate, value
            temperature *= cooling_rate
        metadata.update(cooling_rate=cooling_rate, final_temperature=temperature,
                        initial_temperature=initial_temperature, minimum_temperature=minimum_temperature)
    elif method == 'sa_improved':
        # Local search consumes the same evaluation budget as annealing.
        if enable_descent:
            descent(initial, sc, instance.demands, capacity, max(1, budget//5), rng=rng)
        current = [r[:] for r in sc.best]
        cost = sc.best_cost
        # Calibrate a distance-scale-dependent initial temperature using budgeted probes.
        probes = min(48, max(1, budget//40), budget-sc.evaluations)
        uphill = []
        for _ in range(probes):
            candidate = random_neighbor(current, instance.demands, capacity, rng, allow_split)
            delta = sc.evaluate(candidate)-cost
            if delta > 1e-9:
                uphill.append(delta)
        t0 = float(np.median(uphill))/math.log(2) if uphill else max(1.0, initial_cost/instance.n)
        t0 = max(t0, 1e-6)
        current, cost = [r[:] for r in sc.best], sc.best_cost
        end_sa = max(sc.evaluations, int(budget*.8)) if enable_descent else budget
        steps = end_sa-sc.evaluations
        for step in range(steps):
            temperature = t0 * (.001 ** (step/max(1, steps-1)))
            candidate = random_neighbor(current, instance.demands, capacity, rng, allow_split)
            value = sc.evaluate(candidate)
            delta = value-cost
            if delta <= 0 or rng.random() < math.exp(-delta/temperature):
                current, cost = candidate, value
        if enable_descent and sc.evaluations < budget:
            descent(sc.best, sc, instance.demands, capacity, budget, rng=rng)
        metadata.update(initial_temperature=t0, end_temperature_ratio=.001,
                        enable_descent=enable_descent, allow_split=allow_split,
                        calibration_evaluations=probes)
    errors = validate_routes(sc.best, instance, capacity)
    if errors:
        raise RuntimeError('求解结果校验失败：'+'；'.join(errors))
    return Result(method, sc.best, sc.best_cost, initial_cost, time.perf_counter()-start,
                  sc.evaluations, sc.history, True, seed, metadata)


def solve_exact(instance: Instance, capacity: int, max_customers=12) -> Result:
    """Held-Karp子集路线DP + 可行子集划分DP。仅用于小规模最优性校验。"""
    instance.validate(capacity)
    if instance.n > max_customers:
        raise ValueError(f'精确求解最多支持{max_customers}个客户。')
    start = time.perf_counter()
    n, size = instance.n, 1 << instance.n
    d = distance_matrix(instance)
    loads = [0]*size
    dp = [[math.inf]*n for _ in range(size)]
    pred = [[-1]*n for _ in range(size)]
    route_cost, last = [math.inf]*size, [-1]*size
    route_cost[0] = 0.0
    for mask in range(1, size):
        bit = mask & -mask
        j = bit.bit_length()-1
        loads[mask] = loads[mask ^ bit] + instance.demands[j+1]
        if loads[mask] > capacity:
            continue
        for j in range(n):
            if not mask & (1 << j):
                continue
            prev = mask ^ (1 << j)
            if not prev:
                dp[mask][j] = d[0][j+1]
            else:
                for k in range(n):
                    if prev & (1 << k):
                        value = dp[prev][k] + d[k+1][j+1]
                        if value < dp[mask][j]:
                            dp[mask][j], pred[mask][j] = value, k
            value = dp[mask][j] + d[j+1][0]
            if value < route_cost[mask]:
                route_cost[mask], last[mask] = value, j
    optimum, partition = [math.inf]*size, [0]*size
    optimum[0] = 0.0
    for mask in range(1, size):
        anchor, subset = mask & -mask, mask
        while subset:
            if subset & anchor and math.isfinite(route_cost[subset]):
                value = route_cost[subset]+optimum[mask ^ subset]
                if value < optimum[mask]:
                    optimum[mask], partition[mask] = value, subset
            subset = (subset-1) & mask
    routes, mask = [], size-1
    while mask:
        subset = partition[mask]
        nodes, current, endpoint = [], subset, last[subset]
        while current:
            nodes.append(endpoint+1)
            previous = pred[current][endpoint]
            current ^= 1 << endpoint
            endpoint = previous
        routes.append([0]+nodes[::-1]+[0])
        mask ^= subset
    assert not validate_routes(routes, instance, capacity)
    actual = total_distance(routes, d)
    if not math.isclose(actual, optimum[-1], abs_tol=1e-8):
        raise RuntimeError('精确求解回溯距离不一致。')
    initial_cost = total_distance(greedy_solution(instance, capacity, d), d)
    return Result('exact', routes, actual, initial_cost, time.perf_counter()-start, 0, [], True, 0,
                  {'optimality_proven': True, 'data_fingerprint': instance.fingerprint})


def instance_csv(instance: Instance) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator='\n')
    writer.writerow(['id', 'x', 'y', 'demand'])
    for i, ((x,y), q) in enumerate(zip(instance.coordinates, instance.demands)):
        writer.writerow([i, repr(x), repr(y), q])
    return buffer.getvalue()


def read_instance_csv(text: str, name='uploaded') -> Instance:
    reader = csv.DictReader(io.StringIO(text.lstrip('\ufeff')))
    aliases = {'节点编号':'id','X坐标':'x','Y坐标':'y','需求量':'demand'}
    values = {}
    try:
        for row in reader:
            row = {aliases.get(k, k):v for k,v in row.items()}
            i, q = int(row['id']), int(row['demand'])
            if i in values:
                raise ValueError('客户编号重复。')
            values[i] = ((float(row['x']), float(row['y'])), q)
    except (KeyError, TypeError) as exc:
        raise ValueError('CSV需要id、x、y、demand四列，并包含0号仓库。') from exc
    if not values or sorted(values) != list(range(len(values))):
        raise ValueError('节点编号必须从0开始连续编号。')
    instance = Instance(tuple(values[i][0] for i in range(len(values))),
                        tuple(values[i][1] for i in range(len(values))), name)
    # Validate intrinsic data even before capacity is selected.
    instance.validate(max(1, max(instance.demands)))
    return instance


def result_record(result: Result, instance: Instance, capacity: int) -> dict:
    return {
        'method':result.method, 'customers':instance.n, 'capacity':capacity,
        'algorithm_seed':result.seed, 'distance':result.distance,
        'initial_distance':result.initial_distance, 'improvement_pct':result.improvement_pct,
        'vehicles':len(result.routes), 'load_rate_pct':100*sum(instance.demands)/(capacity*len(result.routes)),
        'elapsed_seconds':result.elapsed_seconds, 'evaluations':result.evaluations,
        'feasible':result.feasible, 'data_fingerprint':instance.fingerprint,
    }
