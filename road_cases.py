"""同一批配送任务的直线时间模型与有向道路时间模型。"""

from dataclasses import dataclass
import gzip
import json
from pathlib import Path

from cvrp_core import Instance, greedy_solution
from cvrp_enhanced import EnhancedConfig
from cvrptw import PlanCheck, TimeWindowProblem, TimeWindowResult, check_plan, greedy_time_insertion, solve_time_windows
from road_network import geographic_distance, local_coordinates
from stage3_common import load_settings


DATASET = Path(__file__).with_name('datasets') / 'jiading_osm' / 'road_case.json.gz'
DEMO_SEED = 2026
ROAD_WINDOWS = {'tight': 2, 'wide': 12}
ROAD_WINDOW_LABELS = {'tight': '紧时窗（±2 分钟）', 'wide': '宽时窗（±12 分钟）'}
DEMANDS = (0, 4, 5, 6, 3, 4, 7, 3, 4, 5, 6, 3, 4)
SERVICE = (0., 6., 5., 7., 4., 6., 8., 4., 5., 6., 7., 5., 4.)
CAPACITY = 20
DIRECT_SPEED_KMH = 30.


@dataclass(frozen=True)
class RoadScenario:
    road: TimeWindowProblem
    direct: TimeWindowProblem
    witness: list[list[int]]


@dataclass(frozen=True)
class RoadComparison:
    direct_solution: TimeWindowResult
    direct_road_check: PlanCheck
    road_initial: TimeWindowResult
    road_solution: TimeWindowResult


def load_road_data(path: Path = DATASET) -> dict:
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return json.load(stream)


def make_road_scenario(data: dict, width: str) -> RoadScenario:
    if width not in ROAD_WINDOWS:
        raise ValueError('请选择紧时窗或宽时窗。')
    lat_lon = {node: (lat, lon) for node, lat, lon in data['nodes']}
    locations = [lat_lon[node] for node in data['stops']]
    points = tuple(local_coordinates(point, locations[0]) for point in locations)
    instance = Instance(points, DEMANDS, data['name'])
    road_distance = tuple(map(tuple, data['distance_km']))
    road_minutes = tuple(map(tuple, data['travel_minutes']))
    direct_distance = tuple(tuple(geographic_distance(a, b) for b in locations) for a in locations)
    direct_minutes = tuple(tuple(60 * km / DIRECT_SPEED_KMH for km in row) for row in direct_distance)

    witness = greedy_solution(instance, CAPACITY, road_distance)
    depot = (480., 600.)
    wide_model = TimeWindowProblem(instance, CAPACITY, road_distance, road_minutes,
                                  (depot,) * len(locations), SERVICE)
    checked_witness = check_plan(wide_model, witness)
    if not checked_witness.feasible:
        raise RuntimeError('设置收货窗所用的见证路线未通过时间校验。')
    slack = ROAD_WINDOWS[width]
    customer_windows = [depot] * len(locations)
    for vehicle in checked_witness.vehicles:
        for stop in vehicle.stops:
            customer_windows[stop.customer] = (max(depot[0], stop.service_start - slack),
                                               stop.service_start + slack)
    windows = tuple(customer_windows)
    real = TimeWindowProblem(instance, CAPACITY, road_distance, road_minutes, windows, SERVICE)
    direct = TimeWindowProblem(instance, CAPACITY, direct_distance, direct_minutes, windows, SERVICE)
    return RoadScenario(real, direct, witness)


def compare_road_scenario(scenario: RoadScenario, budget: int = 3000, seed: int = DEMO_SEED) -> RoadComparison:
    config = EnhancedConfig(**load_settings()['enhanced_config'])
    direct_initial = greedy_time_insertion(scenario.direct)
    direct_solution = solve_time_windows(scenario.direct, direct_initial, budget, seed, config)
    direct_road_check = check_plan(scenario.road, direct_solution.routes)
    road_initial = greedy_time_insertion(scenario.road)
    road_solution = solve_time_windows(scenario.road, road_initial, budget, seed, config)
    if direct_road_check.structural_errors:
        raise RuntimeError('直线方案的客户覆盖或容量校验出错。')
    return RoadComparison(direct_solution, direct_road_check, road_initial, road_solution)
