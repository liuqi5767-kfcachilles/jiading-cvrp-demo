"""容量及硬收货时间窗约束下的路线构造、退火搜索与独立复核。"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from time import perf_counter

from cvrp_core import Instance, Routes, route_load, total_distance, validate_routes
from cvrp_enhanced import EnhancedConfig, near_lists, neighbors, propose


EPS = 1e-8
Matrix = tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class TimeWindowProblem:
    """分钟自当天零点计，0号节点的窗口是发车起点与最晚返仓时刻。"""

    instance: Instance
    capacity: int
    distances: Matrix
    travel_minutes: Matrix
    windows: tuple[tuple[float, float], ...]
    service_minutes: tuple[float, ...]

    def __post_init__(self):
        self.instance.validate(self.capacity)
        size = self.instance.n + 1
        for name, matrix in (('距离', self.distances), ('行驶时间', self.travel_minutes)):
            if len(matrix) != size or any(len(row) != size for row in matrix):
                raise ValueError(f'{name}矩阵必须包含仓库和所有客户。')
            if any(not math.isfinite(value) or value < 0 for row in matrix for value in row):
                raise ValueError(f'{name}矩阵必须由非负有限数构成。')
            if any(matrix[i][i] != 0 for i in range(size)):
                raise ValueError(f'{name}矩阵对角线必须为零。')
        if len(self.windows) != size or len(self.service_minutes) != size:
            raise ValueError('仓库和每位客户均需提供时间窗与服务时长。')
        for earliest, latest in self.windows:
            if not (math.isfinite(earliest) and math.isfinite(latest) and 0 <= earliest <= latest):
                raise ValueError('时间窗必须为非负有限分钟数，最早时刻不得晚于最晚时刻。')
        if any(not math.isfinite(value) or value < 0 for value in self.service_minutes):
            raise ValueError('服务时长必须为非负有限分钟数。')
        if self.service_minutes[0] != 0:
            raise ValueError('仓库服务时长应为零。')


@dataclass(frozen=True)
class StopSchedule:
    customer: int
    arrival: float
    service_start: float
    departure: float
    waiting: float
    late_by: float


@dataclass(frozen=True)
class RouteSchedule:
    stops: tuple[StopSchedule, ...]
    load: int
    distance: float
    start: float
    return_time: float
    return_late_by: float

    @property
    def waiting(self) -> float:
        return sum(stop.waiting for stop in self.stops)


@dataclass(frozen=True)
class PlanCheck:
    vehicles: tuple[RouteSchedule, ...]
    structural_errors: tuple[str, ...] = ()

    @property
    def late_customers(self) -> int:
        return sum(stop.late_by > EPS for route in self.vehicles for stop in route.stops)

    @property
    def late_returns(self) -> int:
        return sum(route.return_late_by > EPS for route in self.vehicles)

    @property
    def feasible(self) -> bool:
        return not self.structural_errors and self.late_customers == self.late_returns == 0

    @property
    def distance(self) -> float:
        return sum(route.distance for route in self.vehicles)

    @property
    def waiting(self) -> float:
        return sum(route.waiting for route in self.vehicles)


@dataclass(frozen=True)
class TimeWindowResult:
    routes: Routes
    check: PlanCheck
    elapsed_seconds: float
    evaluations: int = 0
    rejected_for_time: int = 0
    history: tuple[float, ...] = ()


class InitialSolutionNotFound(ValueError):
    """可行插入未能得到覆盖所有客户的初始解。"""


def _visit(problem: TimeWindowProblem, time: float, previous: int, customer: int):
    arrival = time + problem.travel_minutes[previous][customer]
    service_start = max(arrival, problem.windows[customer][0])
    departure = service_start + problem.service_minutes[customer]
    return arrival, service_start, departure


def _fits_time(problem: TimeWindowProblem, route: list[int]) -> bool:
    """输入是由构造或邻域生成的闭合路线。"""
    time, close = problem.windows[0]
    for previous, customer in zip(route, route[1:-1]):
        _, service_start, time = _visit(problem, time, previous, customer)
        if service_start > problem.windows[customer][1] + EPS:
            return False
    return time + problem.travel_minutes[route[-2]][0] <= close + EPS


def check_plan(problem: TimeWindowProblem, routes: Routes) -> PlanCheck:
    """复核覆盖、仓库、容量与时间窗，给出到达、等候、服务及返仓明细。"""
    errors = validate_routes(routes, problem.instance, problem.capacity)
    if errors:
        return PlanCheck((), tuple(errors))
    start, close = problem.windows[0]
    schedules = []
    for route in routes:
        time = start
        stops = []
        for previous, customer in zip(route, route[1:-1]):
            arrival, service_start, time = _visit(problem, time, previous, customer)
            stops.append(StopSchedule(customer, arrival, service_start, time,
                                      service_start - arrival,
                                      max(0.0, service_start - problem.windows[customer][1])))
        returned = time + problem.travel_minutes[route[-2]][0]
        schedules.append(RouteSchedule(
            tuple(stops), route_load(route, problem.instance.demands),
            total_distance([route], problem.distances), start, returned,
            max(0.0, returned - close),
        ))
    return PlanCheck(tuple(schedules))


def greedy_time_insertion(problem: TimeWindowProblem) -> TimeWindowResult:
    """先处理最早截止的客户，插入已有路线中增加距离最少的可行位置。"""
    start = perf_counter()
    distances = problem.distances
    demands = problem.instance.demands
    order = sorted(range(1, problem.instance.n + 1), key=lambda i: (problem.windows[i][1], i))
    routes: Routes = []
    loads: list[int] = []
    for customer in order:
        best = None
        for route_index, route in enumerate(routes):
            if loads[route_index] + demands[customer] > problem.capacity:
                continue
            for position in range(1, len(route)):
                a, b = route[position - 1], route[position]
                increase = distances[a][customer] + distances[customer][b] - distances[a][b]
                key = (increase, route_index, position)
                if best is not None and key >= best:
                    continue
                candidate = route[:position] + [customer] + route[position:]
                if _fits_time(problem, candidate):
                    best = key
        if best is None:
            candidate = [0, customer, 0]
            if not _fits_time(problem, candidate):
                raise InitialSolutionNotFound(f'插入法未找到满足客户 {customer} 时间窗的初始路线。')
            routes.append(candidate)
            loads.append(demands[customer])
        else:
            _, route_index, position = best
            routes[route_index].insert(position, customer)
            loads[route_index] += demands[customer]
    checked = check_plan(problem, routes)
    if not checked.feasible:
        raise RuntimeError('时间窗初始方案复核失败。')
    return TimeWindowResult(routes, checked, perf_counter() - start)


@dataclass
class _WindowSearch:
    problem: TimeWindowProblem
    best: Routes
    best_distance: float
    evaluations: int = 0
    rejected_for_time: int = 0
    history: list[float] = field(default_factory=list)

    def consider(self, candidate: Routes) -> float | None:
        self.evaluations += 1
        value = None
        if all(_fits_time(self.problem, route) for route in candidate):
            value = total_distance(candidate, self.problem.distances)
            if value < self.best_distance - EPS:
                self.best = candidate
                self.best_distance = value
        else:
            self.rejected_for_time += 1
        self.history.append(self.best_distance)
        return value


def _improve(search: _WindowSearch, nearby, rng, config, stop: int):
    problem = search.problem
    while search.evaluations < stop:
        current = search.best
        cost = search.best_distance
        for candidate, _ in neighbors(current, problem.instance.demands, problem.capacity, nearby, rng, config):
            if search.evaluations >= stop:
                return
            value = search.consider(candidate)
            if value is not None and value < cost - EPS:
                break
        else:
            return


def solve_time_windows(problem: TimeWindowProblem, initial: TimeWindowResult, budget: int = 3000,
                       seed: int = 2024, config: EnhancedConfig | None = None) -> TimeWindowResult:
    """用第二版邻域与分段退火搜索，候选通过时间检查后才参与比较。"""
    if type(budget) is not int or budget < 1:
        raise ValueError('候选检查上限须为正整数。')
    config = config or EnhancedConfig(neighbor_count=8, temperature_scale=.2)
    config.validate()
    start = perf_counter()
    routes = [route[:] for route in initial.routes]
    checked = check_plan(problem, routes)
    if not checked.feasible:
        raise ValueError('退火需要覆盖所有客户的可行初始方案。')
    search = _WindowSearch(problem, routes, checked.distance)
    nearby = near_lists(problem.distances, config.neighbor_count)
    rng = random.Random(seed)
    if config.use_descent:
        _improve(search, nearby, rng, config, int(budget * config.initial_descent_fraction))
    reserved = int(budget * config.final_descent_fraction) if config.use_descent else 0
    annealing_steps = budget - search.evaluations - reserved
    base, extra = divmod(annealing_steps, config.cycles)
    temperature = max(1e-8, config.temperature_scale * checked.distance / problem.instance.n)
    proposal_stats = {'proposal_attempts': 0, 'constraint_or_identity_rejections': 0}
    for cycle in range(config.cycles):
        count = base + int(cycle < extra)
        current = search.best
        cost = search.best_distance
        for step in range(count):
            candidate, _ = propose(current, problem.instance.demands, problem.capacity,
                                   nearby, rng, config, proposal_stats)
            value = search.consider(candidate)
            if value is None:
                continue
            delta = value - cost
            t = temperature * .01 ** (step / max(1, count - 1))
            if delta <= 0 or rng.random() < math.exp(-delta / t):
                current, cost = candidate, value
    if config.use_descent:
        _improve(search, nearby, rng, config, budget)
    final = check_plan(problem, search.best)
    if not final.feasible or not math.isclose(final.distance, search.best_distance, abs_tol=EPS):
        raise RuntimeError('时间窗退火结果复核失败。')
    return TimeWindowResult(search.best, final, initial.elapsed_seconds + perf_counter() - start,
                            search.evaluations, search.rejected_for_time, tuple(search.history))
