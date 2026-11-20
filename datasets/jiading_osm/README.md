# 上海同济大学嘉定校区周边道路算例

道路与边界数据 © OpenStreetMap contributors。随包所含 `jiading_roads.osm.gz` 和 `road_case.json.gz` 按 [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/) 提供。署名见 [OpenStreetMap Copyright](https://www.openstreetmap.org/copyright)。

原始文件为 [Geofabrik 的 shanghai-261118.osm.pbf](https://download.geofabrik.de/asia/china/shanghai-261118.osm.pbf)。PBF 中的数据截止时间为 `2026-11-18T21:21:02Z`，上海本地时间 2026 年 11 月 19 日 05:21。原文件 SHA-256 为 `9345c6cbafcf0ea06981ccfaaaaa2927b291aa3afa072e6c2356119e287e046a`。上海数据下载页见 [Geofabrik 上海](https://download.geofabrik.de/asia/china/shanghai.html)。

## 研究范围和模型

范围为东经 121.188° 至 121.235°、北纬 31.265° 至 31.310°，约 4.5 km × 5.0 km。摘录保留区域内的机动车候选道路，以及 OSM 对象 `way/384531657` 对应的同济大学嘉定校区轮廓。建图时保留两端均在范围内的路段，再取与仓库双向可达的部分，共 3,291 个节点、5,848 条有向弧。站点 0 是校区南门外曹安公路与绿苑路附近的模拟仓库，其余 12 个模拟客户分布于周边。

弧的方向和通行性依据 OSM 路段标签。弧长按节点的球面距离求得。弧的估计速度取道路等级默认速度与可读数值限速的较小值。以弧的行驶时间运行 Dijkstra，同时间时选择里程更短的路径。两张 13×13 矩阵记录**同一条估计最快路径**上的用时和里程。模型范围是所截取的路网，通行条件限于路段标签，行驶用时采用固定速度。时段交通、校门现场通行条件和复杂转向约束属于更细的导航模型。

直线和道路对照采用相同的模拟客户、需求、服务时长和收货窗口。容量为 20，仓库 08:00 开放、10:00 截止返仓。宽窗和紧窗分别以同一条道路可行方案的开始服务时刻为中心，设置 ±12 分钟和 ±2 分钟，窗口下界至少为 08:00。

## 离线复现

`jiading_roads.osm.gz` 是从上海快照保留的原始道路摘录。展开该文件得到的 XML 的 SHA-256 为 `b1fcf287e890f54757915cd5a030042d0a2f1394988dc3a8302f4793df4f8957`。直接运行网页时会读取已经计算好的有向图和矩阵。从项目根目录运行以下命令可离线重建并复跑。

```bat
python road_network.py datasets\jiading_osm\jiading_roads.osm.gz datasets\jiading_osm\road_case.json.gz
python road_benchmark.py --output results_jiading_local --budget 3000
```

若要核对摘录步骤，从上述链接下载固定 PBF 后，执行以下命令。此处的 `osmium` 仅用于读取 PBF。

```bat
python -m pip install osmium==4.2.0
python extract_jiading_roads.py path\to\shanghai-261118.osm.pbf datasets\jiading_osm\jiading_roads.osm.gz
```

| 字段 | 含义 |
|---|---|
| `source` | 原始来源、校验值、时间戳、裁剪范围和许可 |
| `nodes`、`edges` | 节点的编号和经纬度。有向弧记录起点、终点、分钟和公里 |
| `campus_outline` | 嘉定校区边界经纬度 |
| `stops` | 仓库及客户对应的道路节点编号 |
| `travel_minutes` | 各对站点的最短估计行驶时间 |
| `distance_km` | 同一条估计最快路径上的道路里程 |
| `paths` | 各对站点对应的道路节点序列 |
