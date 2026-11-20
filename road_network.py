"""从固定 OSM 道路样本构造有向图，并按估计行驶时间计算路径和矩阵。"""

from __future__ import annotations

import gzip
from hashlib import sha256
from heapq import heappop, heappush
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET


EARTH_KM = 6371.0088
ESTIMATED_SPEED = {
    'motorway': 70., 'motorway_link': 35., 'trunk': 55., 'trunk_link': 30.,
    'primary': 40., 'primary_link': 25., 'secondary': 35., 'secondary_link': 25.,
    'tertiary': 30., 'tertiary_link': 25., 'unclassified': 25.,
    'residential': 25., 'living_street': 10., 'service': 15.,
}
RESTRICTED = {'no', 'private', 'permit', 'agricultural', 'forestry', 'customers', 'delivery', 'destination'}
ACCESS_KEYS = ('motorcar', 'motor_vehicle', 'vehicle', 'access')
# WGS84，0 为校区南门附近的模拟仓库，其余为周边模拟客户的位置。
STOP_TARGETS = (
    (31.2836, 121.2075), (31.2764, 121.1950), (31.2708, 121.2017),
    (31.2705, 121.2130), (31.2784, 121.2242), (31.2839, 121.2300),
    (31.2892, 121.2225), (31.2943, 121.2245), (31.2995, 121.2163),
    (31.3011, 121.2055), (31.2956, 121.1956), (31.2899, 121.1965),
    (31.2826, 121.2009),
)
JIADING_BOUNDS = (121.188, 31.265, 121.235, 31.310)  # 西、南、东、北
CAMPUS_WAY_ID = 384531657
SHANGHAI_SOURCE = 'https://download.geofabrik.de/asia/china/shanghai-261118.osm.pbf'
SHANGHAI_SHA256 = '9345c6cbafcf0ea06981ccfaaaaa2927b291aa3afa072e6c2356119e287e046a'
SHANGHAI_TIMESTAMP = '2026-11-18T21:21:02Z'
JIADING_EXTRACT_SHA256 = 'b1fcf287e890f54757915cd5a030042d0a2f1394988dc3a8302f4793df4f8957'


def geographic_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    """经纬度为纬度在前，返回球面距离（km）。"""
    lat_a, lon_a = map(math.radians, a)
    lat_b, lon_b = map(math.radians, b)
    h = math.sin((lat_b - lat_a) / 2) ** 2
    h += math.cos(lat_a) * math.cos(lat_b) * math.sin((lon_b - lon_a) / 2) ** 2
    return 2 * EARTH_KM * math.asin(math.sqrt(min(1., h)))


def local_coordinates(lat_lon: tuple[float, float], origin: tuple[float, float]) -> tuple[float, float]:
    lat, lon = lat_lon
    lat0, lon0 = origin
    x = EARTH_KM * math.radians(lon - lon0) * math.cos(math.radians(lat0))
    y = EARTH_KM * math.radians(lat - lat0)
    return (x, y)


def _permitted(tags: dict[str, str], direction: str = '') -> bool:
    for key in ACCESS_KEYS:
        val = tags.get(f'{key}:{direction}') if direction else None
        val = val or tags.get(key)
        if val is not None:
            return val not in RESTRICTED
    return True


def _speed(tags, direction: str) -> float:
    estimate = ESTIMATED_SPEED[tags['highway']]
    raw = tags.get(f'maxspeed:{direction}', tags.get('maxspeed', '')).strip().lower()
    match = re.fullmatch(r'(\d+(?:\.\d+)?)\s*(km/h|mph)?', raw)
    if match:
        limit = float(match[1]) * (1.609344 if match[2] == 'mph' else 1)
        if limit > 0:
            estimate = min(estimate, limit)
    return estimate


def _directions(tags: dict[str, str]) -> tuple[str, ...]:
    value = tags.get('oneway:motorcar', tags.get('oneway:motor_vehicle', tags.get('oneway', '')))
    if value in ('yes', 'true', '1'):
        options = ('forward',)
    elif value == '-1':
        options = ('backward',)
    elif value in ('reversible', 'alternating'):
        return ()
    elif value not in ('no', 'false', '0') and (
        tags.get('junction') == 'roundabout' or tags.get('highway') == 'motorway'
    ):
        options = ('forward',)
    else:
        options = ('forward', 'backward')
    return tuple(option for option in options if _permitted(tags, option))


def read_osm(source: Path, bounds: tuple[float, float, float, float] | None = None):
    """读取坐标、有向道路和校区边界，弧权是（分钟、公里）。"""
    opener = gzip.open if source.suffix == '.gz' else open
    with opener(source, 'rb') as stream:
        root = ET.parse(stream).getroot()
    if root.tag != 'osm':
        raise ValueError('需要 OSM XML 道路数据。')
    points = {int(node.attrib['id']): (float(node.attrib['lat']), float(node.attrib['lon']))
              for node in root.findall('node')}
    campus = root.find(f"./way[@id='{CAMPUS_WAY_ID}']")
    outline = [] if campus is None else [points[int(node.attrib['ref'])] for node in campus.findall('nd')]
    if bounds is not None:
        west, south, east, north = bounds
        points = {node: (lat, lon) for node, (lat, lon) in points.items()
                  if south <= lat <= north and west <= lon <= east}
    graph: dict[int, dict[int, tuple[float, float]]] = {}
    for way in root.findall('way'):
        tags = {tag.attrib['k']: tag.attrib['v'] for tag in way.findall('tag')}
        if tags.get('highway') not in ESTIMATED_SPEED or tags.get('area') == 'yes':
            continue
        directions = _directions(tags)
        speeds = {direction: _speed(tags, direction) for direction in directions}
        ids = [int(node.attrib['ref']) for node in way.findall('nd')]
        for a, b in zip(ids, ids[1:]):
            if a not in points or b not in points or a == b:
                continue
            km = geographic_distance(points[a], points[b])
            for direction, speed in speeds.items():
                u, v = (a, b) if direction == 'forward' else (b, a)
                row = graph.setdefault(u, {})
                graph.setdefault(v, {})
                edge = (60 * km / speed, km)
                if v not in row or edge < row[v]:
                    row[v] = edge
    return points, graph, outline


def _reachable(graph, start):
    reached = {start}
    pending = [start]
    while pending:
        for other in graph[pending.pop()]:
            if other not in reached:
                reached.add(other)
                pending.append(other)
    return reached


def _returning_nodes(graph, depot: int) -> set[int]:
    reverse = {node: set() for node in graph}
    for a, row in graph.items():
        for b in row:
            reverse[b].add(a)
    return _reachable(graph, depot) & _reachable(reverse, depot)


def dijkstra(graph, source: int, targets: set[int]):
    """优先最短估计行驶时间，同时间时取较短里程。"""
    pending = set(targets)
    best = {source: (0., 0.)}
    parent = {}
    queue = [(0., 0., source)]
    while queue and pending:
        minutes, distance, node = heappop(queue)
        if (minutes, distance) != best[node]:
            continue
        pending.discard(node)
        for neighbor, (edge_minutes, edge_km) in graph[node].items():
            weight = (minutes + edge_minutes, distance + edge_km)
            if neighbor not in best or weight < best[neighbor]:
                best[neighbor] = weight
                parent[neighbor] = node
                heappush(queue, (*weight, neighbor))
    if pending:
        raise ValueError('存在无法通过有向道路到达的配送节点。')
    paths = {}
    for target in sorted(targets):
        trail = [target]
        while trail[-1] != source:
            trail.append(parent[trail[-1]])
        paths[target] = (best[target], list(reversed(trail)))
    return paths


def build_dataset(source: Path) -> dict:
    content = source.read_bytes()
    if source.suffix == '.gz':
        content = gzip.decompress(content)
    digest = sha256(content).hexdigest()
    if digest != JIADING_EXTRACT_SHA256:
        raise ValueError('请使用算例随附的嘉定道路 OSM 样本。')
    points, graph, outline = read_osm(source, JIADING_BOUNDS)
    depot = min(graph, key=lambda node: (geographic_distance(STOP_TARGETS[0], points[node]), node))
    retained = _returning_nodes(graph, depot)
    selected = [depot]
    for target in STOP_TARGETS[1:]:
        node = min(retained - set(selected),
                   key=lambda item: (geographic_distance(target, points[item]), item))
        selected.append(node)
    graph = {
        a: {b: edge for b, edge in sorted(graph[a].items()) if b in retained}
        for a in sorted(retained)
    }
    routes = [dijkstra(graph, start, set(selected)) for start in selected]
    return {
        'name': 'shanghai_jiading_12',
        'source': {'url': SHANGHAI_SOURCE, 'sha256': SHANGHAI_SHA256, 'snapshot_utc': SHANGHAI_TIMESTAMP,
                   'bounds_wsen': JIADING_BOUNDS, 'extract_xml_sha256': digest,
                   'attribution': '© OpenStreetMap contributors', 'license': 'ODbL 1.0',
                   'license_url': 'https://opendatacommons.org/licenses/odbl/1-0/'},
        'campus_outline': outline,
        'nodes': [[node, *points[node]] for node in sorted(retained)],
        'edges': [[a, b, value[0], value[1]] for a, row in graph.items() for b, value in row.items()],
        'stops': selected,
        'distance_km': [[record[j][0][1] for j in selected] for record in routes],
        'travel_minutes': [[record[j][0][0] for j in selected] for record in routes],
        'paths': [[record[j][1] for j in selected] for record in routes],
    }


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description=__doc__)
    parser.add_argument('osm_file', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    dataset = build_dataset(args.osm_file)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(dataset, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    args.output.write_bytes(gzip.compress(raw, mtime=0))
    print(f"{len(dataset['nodes'])} 道路节点，{len(dataset['edges'])} 有向弧，{len(dataset['stops'])} 站点。")
