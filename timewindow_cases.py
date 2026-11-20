"""可手算和可重复生成的时间窗算例，以及三种方案的同实例对照。"""

from dataclasses import dataclass

from cvrp_core import Instance, distance_matrix, generate_instance, greedy_solution
from cvrp_enhanced import EnhancedConfig
from cvrptw import TimeWindowProblem, TimeWindowResult, check_plan, greedy_time_insertion, solve_time_windows
from stage3_common import load_settings, run_method


METHOD_LABELS = {
    'capacity': '容量版第二版（时间窗复核）',
    'insertion': '时间窗贪心插入',
    'annealing': '时间窗邻近退火',
}
WINDOW_SLACK = {'wide': 45, 'tight': 12}
WINDOW_LABELS = {'wide': '宽时窗（±45 分钟）', 'tight': '紧时窗（±12 分钟）'}


@dataclass(frozen=True)
class SimulatedCase:
    problem: TimeWindowProblem
    witness: list[list[int]]


def hand_example() -> TimeWindowProblem:
    instance = Instance(
        ((0., 0.), (2., 0.), (5., 0.), (1., 4.), (5., 4.)),
        (0, 3, 3, 3, 3), 'hand_4',
    )
    distances = tuple(map(tuple, distance_matrix(instance)))
    return TimeWindowProblem(
        instance, 12, distances, distances,
        ((480., 540.), (480., 535.), (486., 488.), (505., 530.), (480., 532.)),
        (0., 6., 4., 5., 3.),
    )


def simulated_case(customers: int, capacity: int, data_seed: int, width: str) -> SimulatedCase:
    """围绕同一条已知可行的配送方案生成嵌套的宽/紧时间窗。"""
    slack = WINDOW_SLACK.get(width)
    if slack is None:
        raise ValueError('请选择宽时窗或紧时窗。')
    instance = generate_instance(customers, data_seed)
    distances = tuple(map(tuple, distance_matrix(instance)))
    travel = tuple(tuple(distance / 2 for distance in row) for row in distances)
    service = (0.,) + tuple(float(3 + (i * 7 + data_seed) % 5) for i in range(1, customers + 1))
    witness = greedy_solution(instance, capacity, distances)
    start = 480.
    arrivals = [start] * (customers + 1)
    latest_return = start
    for route in witness:
        minute = start
        for previous, customer in zip(route, route[1:-1]):
            minute += travel[previous][customer]
            arrivals[customer] = minute
            minute += service[customer]
        latest_return = max(latest_return, minute + travel[route[-2]][0])
    end = max(720., latest_return + 60.)
    windows = ((start, end),) + tuple(
        (max(start, minute - slack), minute + slack) for minute in arrivals[1:]
    )
    problem = TimeWindowProblem(instance, capacity, distances, travel, windows, service)
    if not check_plan(problem, witness).feasible:
        raise RuntimeError('模拟数据的已知可行方案校验失败。')
    return SimulatedCase(problem, witness)


def capacity_reference(problem: TimeWindowProblem, budget: int, seed: int, settings: dict) -> TimeWindowResult:
    """以容量版第二版为对照，再用输入的时间窗复核其路线。"""
    ordinary = run_method(problem.instance, problem.capacity, 'guided_v2', settings, budget, seed)
    checked = check_plan(problem, ordinary.routes)
    if checked.structural_errors or abs(checked.distance - ordinary.distance) > 1e-7:
        raise RuntimeError('容量版对照的路线或距离不一致。')
    return TimeWindowResult(ordinary.routes, checked, ordinary.elapsed_seconds, ordinary.evaluations,
                            history=(ordinary.initial_distance, *ordinary.history))


def compare(problem: TimeWindowProblem, budget: int = 3000, seed: int = 2024) -> dict[str, TimeWindowResult]:
    settings = load_settings()
    initial = greedy_time_insertion(problem)
    improved = solve_time_windows(problem, initial, budget, seed, EnhancedConfig(**settings['enhanced_config']))
    reference = capacity_reference(problem, budget, seed, settings)
    return {'capacity': reference, 'insertion': initial, 'annealing': improved}


def stop_rows(result: TimeWindowResult) -> list[dict]:
    """为逐站明细保留未格式化的分钟数，便于独立复核。"""
    return [
        {'vehicle': vehicle_id, 'customer': stop.customer, 'arrival': stop.arrival,
         'service_start': stop.service_start, 'departure': stop.departure,
         'waiting_minutes': stop.waiting, 'late_minutes': stop.late_by,
         'route_return': route.return_time}
        for vehicle_id, route in enumerate(result.check.vehicles, 1)
        for stop in route.stops
    ]
