"""显示被采用的道路折线及时间窗复核中的迟到位置。"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon

from cvrptw import EPS, PlanCheck
from road_network import local_coordinates


def plot_road_plan(data: dict, routes: list[list[int]], checked: PlanCheck):
    stops = data['stops']
    lat_lon = {node: (lat, lon) for node, lat, lon in data['nodes']}
    points = {node: local_coordinates(point, lat_lon[stops[0]]) for node, point in lat_lon.items()}
    lines = {(min(a, b), max(a, b)) for a, b, _, _ in data['edges']}
    fig, ax = plt.subplots(figsize=(6.2, 5.1), layout='constrained')
    outline = [local_coordinates(point, lat_lon[stops[0]]) for point in data['campus_outline']]
    ax.add_patch(Polygon(outline, closed=True, facecolor='#e1edf8', edgecolor='#7c98b4',
                         linewidth=.8, zorder=0, label='Tongji Jiading'))
    ax.add_collection(LineCollection([(points[a], points[b]) for a, b in sorted(lines)],
                                    colors='#d8dfe7', linewidths=.65, zorder=1))
    colors = ('#08776c', '#4265aa', '#b56b22', '#904f91', '#4a8899')
    for i, route in enumerate(routes, 1):
        trail = [stops[0]]
        for a, b in zip(route, route[1:]):
            trail.extend(data['paths'][a][b][1:])
        x, y = zip(*(points[node] for node in trail), strict=True)
        ax.plot(x, y, color=colors[(i - 1) % len(colors)], alpha=.86, lw=1.9,
                label=f'Vehicle {i}', zorder=2)
    late = {s.customer for trip in checked.vehicles for s in trip.stops if s.late_by > EPS}
    for customer, node in enumerate(stops[1:], 1):
        xy = points[node]
        fill = '#efb0ad' if customer in late else '#ffffff'
        border = '#a73232' if customer in late else '#233b52'
        ax.scatter(*xy, s=56, color=fill, edgecolors=border, linewidths=.95, zorder=4)
        ax.annotate(str(customer), xy, color=border, ha='center', va='center', weight='bold',
                    fontsize=7, zorder=5)
    ax.scatter(*points[stops[0]], marker='*', color='#233b52', s=153, zorder=5, label='Depot')
    if late:
        ax.scatter([], [], s=54, facecolor='#efb0ad', edgecolor='#a73232', label='Late')
    ax.autoscale()
    ax.margins(.06)
    ax.set_aspect('equal')
    ax.set_xlabel('East of depot (km)')
    ax.set_ylabel('North of depot (km)')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, 1.01), ncol=3, frameon=False, fontsize=8.4)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.text(.995, .01, '© OpenStreetMap contributors', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=7.2, color='#4b5968',
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .87, 'pad': 2})
    return fig
