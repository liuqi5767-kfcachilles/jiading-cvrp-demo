"""网页和实验图共用的交付时间轴。"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator

from cvrptw import EPS, RouteSchedule, TimeWindowProblem


def clock(minutes: float) -> str:
    seconds = round(minutes * 60)
    hour, rest = divmod(seconds, 3600)
    minute, second = divmod(rest, 60)
    return f'{hour:02d}:{minute:02d}:{second:02d}'


def plot_schedule(problem: TimeWindowProblem, route: RouteSchedule):
    fig, ax = plt.subplots(figsize=(9, max(3.0, 1.55 + .43 * len(route.stops))), layout='constrained')
    window_color, waiting_color, service_color = '#dbe5f1', '#d39a3b', '#007d76'
    late_color, arrival_color = '#b84441', '#253a57'
    for y, stop in enumerate(route.stops):
        earliest, latest = problem.windows[stop.customer]
        ax.barh(y, latest - earliest, left=earliest, height=.58, color=window_color)
        if stop.waiting > EPS:
            ax.barh(y, stop.waiting, left=stop.arrival, height=.24, color=waiting_color)
        color = late_color if stop.late_by > EPS else service_color
        ax.barh(y, stop.departure - stop.service_start, left=stop.service_start, height=.24, color=color)
        ax.scatter(stop.arrival, y, s=15, color=arrival_color, zorder=3)
    keys = [Patch(facecolor=window_color, label='Allowed service start'),
            Line2D([], [], marker='o', lw=0, ms=4, color=arrival_color, label='Arrival'),
            Patch(facecolor=waiting_color, label='Waiting'),
            Patch(facecolor=service_color, label='Service')]
    if any(stop.late_by > EPS for stop in route.stops):
        keys.append(Patch(facecolor=late_color, label='Late service'))
    ax.legend(handles=keys, loc='lower center', bbox_to_anchor=(.5, 1.0), ncol=len(keys),
              fontsize=9, frameon=False)
    ax.set_yticks(range(len(route.stops)), [f'Customer {stop.customer}' for stop in route.stops])
    ax.invert_yaxis()
    earliest = min(route.start, *(problem.windows[stop.customer][0] for stop in route.stops))
    latest = max(route.return_time, *(problem.windows[stop.customer][1] for stop in route.stops))
    margin = max(1., (latest - earliest) * .025)
    ax.set_xlim(max(0., earliest - margin), latest + margin)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda minute, _: clock(minute)[:5]))
    ax.set_xlabel('Time of day')
    ax.grid(axis='x', color='#e4e9ef', lw=.75)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    return fig
