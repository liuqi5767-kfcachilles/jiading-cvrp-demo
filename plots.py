"""可供网页和报告共用的精确路线图与历史最好距离图。"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cvrp_core import total_distance


def plot_routes(instance, routes, distances, title='', show_labels=True):
    fig, ax = plt.subplots(figsize=(7,5.4))
    colors = plt.get_cmap('tab20')
    for k, route in enumerate(routes):
        points = [instance.coordinates[i] for i in route]
        ax.plot([p[0] for p in points],[p[1] for p in points],'-o',
                color=colors(k%20),lw=1.6,markersize=3.5,alpha=.85)
    x,y = instance.coordinates[0]
    ax.scatter([x],[y],marker='*',s=230,c='black',zorder=6)
    ax.annotate('Depot',(x,y),xytext=(6,6),textcoords='offset points',fontsize=9)
    if show_labels:
        for i,(x,y) in enumerate(instance.coordinates[1:],1):
            ax.annotate(str(i),(x,y),xytext=(4,4),textcoords='offset points',fontsize=7)
    ax.set_title(f'{title}\nDistance {total_distance(routes,distances):.2f} | Vehicles {len(routes)}',fontsize=11)
    ax.set_xlabel('X coordinate'); ax.set_ylabel('Y coordinate')
    ax.set_aspect('equal',adjustable='datalim'); ax.margins(.12)
    ax.grid(alpha=.18); fig.tight_layout()
    return fig


def plot_history(results):
    fig,ax=plt.subplots(figsize=(9,3.8))
    names={'greedy_2opt':'Greedy + 2-opt','sa_original':'Basic SA','sa_improved':'Improved SA'}
    for method,result in results.items():
        if result.history:
            ax.plot(range(len(result.history)+1),[result.initial_distance]+result.history,
                    label=names.get(method,method),lw=1.6)
    ax.set_xlabel('Candidate solution evaluations')
    ax.set_ylabel('Best distance'); ax.grid(alpha=.2); ax.legend()
    fig.tight_layout()
    return fig
