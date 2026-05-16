"""
FastAPI Backend — Pasig City Health Facility Shortest Path
Uses the BMSSP (Breaking the Sorting Barrier) Algorithm
"""

import math
import heapq
import pandas as pd
import osmnx as ox
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ============================================================================
# YOUR BMSSP ALGORITHM (copied directly from ssp_algo.py — do not modify)
# ============================================================================

@dataclass
class Edge:
    to: int
    weight: float


class Graph:
    def __init__(self, semi_graph):
        self.semi_graph = semi_graph
        self.vertices = semi_graph.number_of_vertices()
        self._vertex_to_index = {}
        self._index_to_vertex = {}
        self.adj = []
        self._build_adjacency()

    def _build_adjacency(self):
        for i, vertex in enumerate(self.semi_graph.vertices):
            self._vertex_to_index[vertex] = i
            self._index_to_vertex[i] = vertex

        self.adj = [[] for _ in range(self.vertices)]

        for (source, target), weight in self.semi_graph.directed_edges.items():
            src_idx = self._vertex_to_index[source]
            tgt_idx = self._vertex_to_index[target]
            self.adj[src_idx].append(Edge(tgt_idx, weight))

        for (v1, v2), weight in self.semi_graph.undirected_edges.items():
            idx1 = self._vertex_to_index[v1]
            idx2 = self._vertex_to_index[v2]
            self.adj[idx1].append(Edge(idx2, weight))
            self.adj[idx2].append(Edge(idx1, weight))

    def get_vertex_name(self, index: int) -> str:
        return self._index_to_vertex[index]

    def get_vertex_index(self, name: str) -> int:
        return self._vertex_to_index[name]


class EfficientDataStructure:
    def __init__(self, block_size: int, bound: float):
        self.block_size = block_size
        self.bound = bound
        self.buckets = []
        self.prepend_list = []

    def insert(self, vertex: int, distance: float):
        if distance >= self.bound:
            return
        bucket_idx = int(distance / self.block_size) if self.block_size > 0 else 0
        while bucket_idx >= len(self.buckets):
            self.buckets.append((bucket_idx * self.block_size, set()))
        self.buckets[bucket_idx][1].add(vertex)

    def batch_prepend(self, vertices: List[Tuple[int, float]]):
        for vertex, dist in vertices:
            self.prepend_list.append((vertex, dist))

    def pull(self) -> Tuple[float, List[int]]:
        if self.prepend_list:
            self.prepend_list.sort(key=lambda x: x[1])
            vertices = [v for v, _ in self.prepend_list]
            max_dist = self.prepend_list[-1][1] if self.prepend_list else 0
            self.prepend_list = []
            return max_dist, vertices

        for i, (range_start, bucket_set) in enumerate(self.buckets):
            if bucket_set:
                vertices = list(bucket_set)
                self.buckets[i] = (range_start, set())
                return range_start + self.block_size, vertices
        return float('inf'), []

    def is_empty(self) -> bool:
        if self.prepend_list:
            return False
        return all(len(bucket) == 0 for _, bucket in self.buckets)


class BmsspSolver:
    INFINITY = float('inf')

    def __init__(self, graph: Graph):
        self.graph = graph
        self.n = graph.vertices
        self.k = int(math.log2(self.n) ** (1 / 3) * 2) if self.n > 1 else 1
        self.t = int(math.log2(self.n) ** (2 / 3)) if self.n > 1 else 1
        self.k = max(self.k, 3)
        self.t = max(self.t, 2)
        self.distances = [self.INFINITY] * self.n
        self.predecessors = [None] * self.n
        self.complete = [False] * self.n
        self.best_goal = self.INFINITY

    def solve(self, source: int, goal: int) -> Optional[Tuple[float, List[int]]]:
        self.distances = [self.INFINITY] * self.n
        self.predecessors = [None] * self.n
        self.complete = [False] * self.n
        self.best_goal = self.INFINITY
        self.distances[source] = 0.0

        max_level = math.ceil(math.log2(self.n) / self.t) if self.n > 1 else 0
        self._bmssp(max_level, self.INFINITY, [source], goal)

        if self.distances[goal] == self.INFINITY:
            return None

        path = self._reconstruct_path(source, goal)
        return self.distances[goal], path

    def _reconstruct_path(self, source: int, goal: int) -> List[int]:
        path = []
        curr = goal
        while curr is not None:
            path.append(curr)
            if curr == source:
                break
            curr = self.predecessors[curr]
        return path[::-1]

    def _bmssp(self, level: int, bound: float, pivots: List[int], goal: Optional[int]) -> List[int]:
        if not pivots or (goal is not None and self.complete[goal]):
            return []

        if level == 0:
            return self._bounded_dijkstra(bound, pivots, goal)

        pivots, _ = self._find_pivots(bound, pivots)
        block_size = 2 ** max(0, (level - 1) * self.t)
        ds = EfficientDataStructure(block_size, bound)

        for pivot in pivots:
            if not self.complete[pivot] and self.distances[pivot] < bound:
                ds.insert(pivot, self.distances[pivot])

        result_set = []

        while not ds.is_empty():
            if goal is not None and self.complete[goal]:
                break

            subset_bound, subset = ds.pull()
            if not subset:
                continue

            sub_result = self._bmssp(level - 1, subset_bound, subset, goal)
            result_set.extend(sub_result)
            self._edge_relaxation(sub_result, subset_bound, bound, ds)

        return result_set

    def _bounded_dijkstra(self, bound: float, frontier: List[int], goal: Optional[int]) -> List[int]:
        pq = []
        for start_node in frontier:
            if not self.complete[start_node] and self.distances[start_node] < bound:
                heapq.heappush(pq, (self.distances[start_node], start_node))

        completed_nodes = []

        while pq:
            dist, u = heapq.heappop(pq)

            if self.complete[u] or dist > self.distances[u]:
                continue

            self.complete[u] = True
            completed_nodes.append(u)

            if u == goal:
                if dist < self.best_goal:
                    self.best_goal = dist
                break

            for edge in self.graph.adj[u]:
                new_dist = dist + edge.weight

                if (not self.complete[edge.to] and
                        new_dist <= self.distances[edge.to] and
                        new_dist < bound and
                        new_dist < self.best_goal):
                    self.distances[edge.to] = new_dist
                    self.predecessors[edge.to] = u
                    heapq.heappush(pq, (new_dist, edge.to))

        return completed_nodes

    def _find_pivots(self, bound: float, frontier: List[int]) -> Tuple[List[int], List[int]]:
        working_set = set(frontier)
        current_layer = {node for node in frontier if not self.complete[node]}

        for _ in range(self.k):
            next_layer = set()
            for u in current_layer:
                if self.distances[u] >= bound:
                    continue
                for edge in self.graph.adj[u]:
                    v = edge.to
                    if self.complete[v]:
                        continue
                    new_dist = self.distances[u] + edge.weight
                    if (new_dist <= self.distances[v] and
                            new_dist < bound and
                            new_dist < self.best_goal):
                        self.distances[v] = new_dist
                        self.predecessors[v] = u
                        if v not in working_set:
                            next_layer.add(v)

            if not next_layer:
                break

            working_set.update(next_layer)
            current_layer = next_layer

            if len(working_set) > self.k * len(frontier):
                return frontier, list(working_set)

        children = {node: [] for node in working_set}
        for node in working_set:
            pred = self.predecessors[node]
            if pred is not None and pred in working_set:
                children.setdefault(pred, []).append(node)

        subtree_sizes = {node: len(ch) for node, ch in children.items()}
        pivots = [root for root in frontier if subtree_sizes.get(root, 0) >= self.k]

        if not pivots:
            return frontier, list(working_set)

        return pivots, list(working_set)

    def _edge_relaxation(self, completed_vertices: List[int], lower_bound: float,
                         upper_bound: float, ds: EfficientDataStructure):
        batch_prepend_list = []

        for u in completed_vertices:
            for edge in self.graph.adj[u]:
                v = edge.to
                if self.complete[v]:
                    continue
                new_dist = self.distances[u] + edge.weight
                if new_dist <= self.distances[v] and new_dist < self.best_goal:
                    self.distances[v] = new_dist
                    self.predecessors[v] = u
                    if new_dist < lower_bound:
                        batch_prepend_list.append((v, new_dist))
                    elif new_dist < upper_bound:
                        ds.insert(v, new_dist)

        if batch_prepend_list:
            ds.batch_prepend(batch_prepend_list)


def _normalize_vertex_id(vertex_id):
    if isinstance(vertex_id, (int, float)) and vertex_id == int(vertex_id):
        return str(int(vertex_id))
    return str(vertex_id)


# ============================================================================
# SEMI-DIRECTED GRAPH (your original class, unchanged)
# ============================================================================

class SemiDirectedGraph:
    def __init__(self):
        self._vertices = set()
        self._directed_edges = {}
        self._undirected_edges = {}
        self._adjacency_directed = defaultdict(dict)
        self._adjacency_undirected = defaultdict(dict)

    def add_vertex(self, vertex):
        self._vertices.add(vertex)

    def add_directed_edge(self, source, target, weight=1.0):
        if source not in self._vertices:
            self.add_vertex(source)
        if target not in self._vertices:
            self.add_vertex(target)
        self._directed_edges[(source, target)] = weight
        self._adjacency_directed[source][target] = weight

    def add_undirected_edge(self, vertex1, vertex2, weight=1.0):
        if vertex1 not in self._vertices:
            self.add_vertex(vertex1)
        if vertex2 not in self._vertices:
            self.add_vertex(vertex2)
        edge = (vertex1, vertex2) if vertex1 < vertex2 else (vertex2, vertex1)
        self._undirected_edges[edge] = weight
        self._adjacency_undirected[vertex1][vertex2] = weight
        self._adjacency_undirected[vertex2][vertex1] = weight

    @property
    def vertices(self):
        return self._vertices.copy()

    @property
    def directed_edges(self):
        return self._directed_edges.copy()

    @property
    def undirected_edges(self):
        return self._undirected_edges.copy()

    def number_of_vertices(self):
        return len(self._vertices)

    def number_of_edges(self):
        return len(self._directed_edges), len(self._undirected_edges)

    def shortest_path_bmssp(self, source: str, destination: str) -> Optional[Tuple[float, List[str]]]:
        indexed_graph = Graph(self)

        if source not in indexed_graph._vertex_to_index:
            return None
        if destination not in indexed_graph._vertex_to_index:
            return None

        src_idx = indexed_graph.get_vertex_index(source)
        dst_idx = indexed_graph.get_vertex_index(destination)

        solver = BmsspSolver(indexed_graph)
        result = solver.solve(src_idx, dst_idx)

        if result is None:
            return None

        distance, path_indices = result
        path = [indexed_graph.get_vertex_name(idx) for idx in path_indices]
        return distance, path


# ============================================================================
# GRAPH LOADER — from your Excel file
# ============================================================================

def load_graph_from_excel(filepath: str) -> Optional[SemiDirectedGraph]:
    """Load the SemiDirectedGraph from your Excel edge list."""
    graph = SemiDirectedGraph()
    try:
        df_directed = pd.read_excel(filepath, sheet_name='directed_edges')
        df_directed.columns = df_directed.columns.str.strip()
        for _, row in df_directed.iterrows():
            source = _normalize_vertex_id(row['source'])
            target = _normalize_vertex_id(row['target'])
            weight = float(row['weight']) if 'weight' in df_directed.columns and pd.notna(row['weight']) else 1.0
            graph.add_directed_edge(source, target, weight)

        df_undirected = pd.read_excel(filepath, sheet_name='undirected_edges')
        df_undirected.columns = df_undirected.columns.str.strip()
        for _, row in df_undirected.iterrows():
            v1 = _normalize_vertex_id(row['vertex1'])
            v2 = _normalize_vertex_id(row['vertex2'])
            weight = float(row['weight']) if 'weight' in df_undirected.columns and pd.notna(row['weight']) else 1.0
            graph.add_undirected_edge(v1, v2, weight)

        print(f"✓ Graph loaded: {graph.number_of_vertices()} vertices")
        return graph
    except Exception as e:
        print(f"✗ Failed to load graph: {e}")
        return None


# ============================================================================
# NODE COORDINATE LOOKUP — uses your nodes_with_coordinates.xlsx
# Coordinates are in EPSG:3857 (Web Mercator) projected format
# and are converted to lat/lng (EPSG:4326) on load.
# ============================================================================

import math

# Global node coordinate lookup: { "1": {"lat": 14.57, "lng": 121.08}, ... }
node_coords_map: dict = {}


def _projected_to_latlng(x: float, y: float):
    """
    Convert projected coordinates to lat/lng (EPSG:4326).
    The coordinates use UTM Zone 51N with large false offsets:
      x_offset = 9503472.38, y_offset = 14503837.33
    Subtracting these gives standard UTM Zone 51N easting/northing.
    """
    X_OFFSET = 9503472.38
    Y_OFFSET = 14503837.33
    k0 = 0.9996
    a = 6378137.0
    e2 = 0.00669438
    lon0 = math.radians(123.0)  # UTM Zone 51N central meridian

    easting = x - X_OFFSET
    northing = y - Y_OFFSET

    e1 = (1 - math.sqrt(1 - e2)) / (1 + math.sqrt(1 - e2))
    M = northing / k0
    mu = M / (a * (1 - e2/4 - 3*e2**2/64 - 5*e2**3/256))

    phi1 = mu + (3*e1/2 - 27*e1**3/32)*math.sin(2*mu)
    phi1 += (21*e1**2/16 - 55*e1**4/32)*math.sin(4*mu)
    phi1 += (151*e1**3/96)*math.sin(6*mu)

    N1 = a / math.sqrt(1 - e2*math.sin(phi1)**2)
    T1 = math.tan(phi1)**2
    C1 = e2*math.cos(phi1)**2 / (1 - e2)
    R1 = a*(1 - e2) / (1 - e2*math.sin(phi1)**2)**1.5
    D = (easting - 500000) / (N1 * k0)

    lat = phi1 - (N1*math.tan(phi1)/R1)*(D**2/2 - (5+3*T1+10*C1-4*C1**2-9*e2)*D**4/24)
    lon = lon0 + (D - (1+2*T1+C1)*D**3/6) / math.cos(phi1)

    return math.degrees(lat), math.degrees(lon)


def load_node_coordinates(filepath: str) -> dict:
    """Load node coordinates from Excel and return {node_id_str: {lat, lng}}."""
    coords = {}
    try:
        df = pd.read_excel(filepath)
        df.columns = df.columns.str.strip()
        for _, row in df.iterrows():
            node_id = str(int(row['node_id']))
            x = float(row['x'])
            y = float(row['y'])
            lat, lng = _projected_to_latlng(x, y)
            coords[node_id] = {"lat": lat, "lng": lng}
        print(f"✓ Node coordinates loaded: {len(coords)} nodes")
    except Exception as e:
        print(f"✗ Failed to load node coordinates: {e}")
    return coords


def get_nearest_node(lat: float, lng: float) -> str:
    """Find the nearest node ID (string) to a given lat/lng using Euclidean distance."""
    best_node = None
    best_dist = float('inf')
    for node_id, coords in node_coords_map.items():
        # Simple Euclidean distance on lat/lng (sufficient for nearby points)
        d = (coords['lat'] - lat) ** 2 + (coords['lng'] - lng) ** 2
        if d < best_dist:
            best_dist = d
            best_node = node_id
    return best_node


def get_node_coords(node_id_str: str) -> dict:
    """Returns lat/lng for a node ID string."""
    return node_coords_map.get(node_id_str, {"lat": 0.0, "lng": 0.0})


# ============================================================================
# HEALTH FACILITIES — add your Pasig City facilities here
# ============================================================================

HEALTH_FACILITIES = [
    {"name": "Pasig City General Hospital",          "lat": 14.5721974,  "lng": 121.0991899},
    {"name": "Child's HOPE (Pasig Children's Hospital)", "lat": 14.5617792,  "lng": 121.0743292},
    {"name": "Rizal Medical Center",                 "lat": 14.5632873,  "lng": 121.0660398},
    {"name": "Alfonso Specialist Hospital",           "lat": 14.5901372,  "lng": 121.0846938},
    {"name": "Holylife Hospital",                    "lat": 14.5920681,  "lng": 121.0957120},
    {"name": "Mission Hospital",                     "lat": 14.5898229,  "lng": 121.0975159},
    {"name": "Family Healthcare Hospital",            "lat": 14.5832360,  "lng": 121.0851870},
    {"name": "Pasig Doctors Medical Center",          "lat": 14.6007047,  "lng": 121.0920985},
    {"name": "Salve Regina General Hospital",         "lat": 14.6199475,  "lng": 121.0963324},
    {"name": "St. Camillus Medical Center",           "lat": 14.6121254,  "lng": 121.0918480},
    {"name": "St. Christiana Hospital",               "lat": 14.6039035,  "lng": 121.1028590},
    {"name": "Tri-city Medical Center",               "lat": 14.5759284,  "lng": 121.0851341},
    {"name": "The Medical City",                     "lat": 14.5894424,  "lng": 121.0695690},
    {"name": "Prime Hospital and Medical Center",     "lat": 14.5590693,  "lng": 121.0857886},
]


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(title="Pasig Health Facility Router (BMSSP)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://pasig-health-finder.netlify.app",
        "http://localhost:8000",
        "http://localhost:*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global state: loaded once at startup ---
road_graph: Optional[SemiDirectedGraph] = None


@app.on_event("startup")
def startup_event():
    global road_graph, node_coords_map

    # 1. Load your Excel-based road graph
    EXCEL_PATH = "final_edge_layer.xlsx"
    road_graph = load_graph_from_excel(EXCEL_PATH)

    # 2. Load node coordinates from your nodes Excel file
    NODES_PATH = "nodes_with_coordinates.xlsx"
    node_coords_map = load_node_coordinates(NODES_PATH)
    print("Startup complete.")


# --- Request / Response Models ---

class LocationRequest(BaseModel):
    lat: float
    lng: float


class RouteToFacilityRequest(BaseModel):
    user_lat: float
    user_lng: float
    facility_lat: float
    facility_lng: float
    facility_name: str


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/health")
def health_check():
    return {"status": "ok", "graph_loaded": road_graph is not None}


@app.get("/facilities")
def list_facilities():
    """Return all health facilities."""
    return HEALTH_FACILITIES


@app.post("/nearest-facility")
def nearest_facility(req: LocationRequest):
    """
    Find the nearest health facility using BMSSP algorithm.
    Accepts user lat/lng, returns the shortest path and destination.
    """
    if road_graph is None:
        raise HTTPException(status_code=503, detail="Road graph not loaded")

    user_node = get_nearest_node(req.lat, req.lng)

    best_result = None
    best_distance = float('inf')

    for facility in HEALTH_FACILITIES:
        facility_node = get_nearest_node(facility["lat"], facility["lng"])

        result = road_graph.shortest_path_bmssp(user_node, facility_node)
        if result is None:
            continue

        distance, path_node_ids = result

        if distance < best_distance:
            best_distance = distance

            # Convert node IDs to lat/lng coordinates
            path_coords = []
            for node_id_str in path_node_ids:
                try:
                    coords = get_node_coords(node_id_str)
                    path_coords.append(coords)
                except Exception:
                    continue

            best_result = {
                "facility": facility["name"],
                "distance": round(distance, 4),   # unit = whatever your edge weights are (meters/seconds)
                "path": path_coords,               # list of {"lat": ..., "lng": ...}
                "destination": {
                    "lat": facility["lat"],
                    "lng": facility["lng"],
                },
            }

    if best_result is None:
        raise HTTPException(status_code=404, detail="No reachable facility found")

    return best_result


@app.post("/route-to-facility")
def route_to_specific_facility(req: RouteToFacilityRequest):
    """
    Route to a specific facility chosen by the user.
    """
    if road_graph is None:
        raise HTTPException(status_code=503, detail="Road graph not loaded")

    user_node = get_nearest_node(req.user_lat, req.user_lng)
    facility_node = get_nearest_node(req.facility_lat, req.facility_lng)

    result = road_graph.shortest_path_bmssp(user_node, facility_node)

    if result is None:
        raise HTTPException(status_code=404, detail=f"No path found to {req.facility_name}")

    distance, path_node_ids = result

    path_coords = []
    for node_id_str in path_node_ids:
        try:
            coords = get_node_coords(node_id_str)
            path_coords.append(coords)
        except Exception:
            continue

    return {
        "facility": req.facility_name,
        "distance": round(distance, 4),
        "path": path_coords,
        "destination": {
            "lat": req.facility_lat,
            "lng": req.facility_lng,
        },
    }
