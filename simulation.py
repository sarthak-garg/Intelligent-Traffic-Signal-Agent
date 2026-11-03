from dataclasses import dataclass, field
from collections import deque
import heapq
import random
import pandas as pd

DIRECTIONS = ("North", "East", "South", "West")
GREEN = "GREEN"
RED = "RED"

@dataclass
class Vehicle:
    vehicle_id: int
    direction: str
    arrival_time: int
    is_emergency: bool = False
    destination: str = ""

@dataclass
class Intersection:
    name: str
    queues: dict = field(default_factory=lambda: {d: deque() for d in DIRECTIONS})
    current_green: str = "North"
    green_elapsed: int = 0
    total_wait: int = 0
    vehicles_served: int = 0
    emergency_served: int = 0

    def queue_lengths(self):
        return {d: len(self.queues[d]) for d in DIRECTIONS}

class TrafficSimulation:
    """Discrete-time traffic simulation comparing fixed-time and intelligent control."""

    def __init__(self, mode="intelligent", traffic_level="Normal", seed=7,
                 intersections=3, duration=240, emergency_rate=0.012):
        self.mode = mode
        self.traffic_level = traffic_level
        self.rng = random.Random(seed)
        self.duration = duration
        self.emergency_rate = emergency_rate
        self.nodes = [Intersection(f"Junction {i+1}") for i in range(intersections)]
        self.tick = 0
        self.next_vehicle_id = 1
        self.history = []
        self.fixed_order = list(DIRECTIONS)
        self.fixed_green_duration = 12
        self.min_green = 5
        self.max_green = 25
        self.last_decision = "Initial signal"
        self.emergency_events = []

    def _arrival_probability(self):
        base = 0.10 if self.traffic_level == "Normal" else 0.24
        return base

    def _spawn_vehicles(self, node):
        p = self._arrival_probability()
        for direction in DIRECTIONS:
            # Independent arrivals on each approach.
            if self.rng.random() < p:
                emergency = self.rng.random() < self.emergency_rate
                vehicle = Vehicle(self.next_vehicle_id, direction, self.tick, emergency)
                self.next_vehicle_id += 1
                node.queues[direction].append(vehicle)
                if emergency:
                    self.emergency_events.append((self.tick, node.name, direction, vehicle.vehicle_id))

    def _score_direction(self, node, direction):
        queue = node.queues[direction]
        if not queue:
            return 0.0
        waiting = sum(self.tick - v.arrival_time for v in queue)
        emergency_bonus = 1000 if any(v.is_emergency for v in queue) else 0
        # Utility rewards clearing long queues and reducing accumulated waiting.
        return len(queue) * 8 + waiting * 0.35 + emergency_bonus

    def _best_next_signal(self, node):
        """One-step search over candidate green directions using a utility score."""
        candidates = []
        for direction in DIRECTIONS:
            queue = node.queues[direction]
            emergency_wait = any(v.is_emergency for v in queue)
            score = self._score_direction(node, direction)
            # Avoid needless switching when the current lane remains productive.
            if direction == node.current_green:
                score += 3
            candidates.append((score, direction, emergency_wait))
        candidates.sort(reverse=True)
        best_score, best_direction, emergency_wait = candidates[0]
        return best_direction, best_score

    def _choose_signal(self, node):
        if self.mode == "fixed":
            if node.green_elapsed >= self.fixed_green_duration:
                idx = self.fixed_order.index(node.current_green)
                node.current_green = self.fixed_order[(idx + 1) % len(self.fixed_order)]
                node.green_elapsed = 0
                self.last_decision = f"Fixed cycle → {node.current_green}"
        else:
            current_q = node.queues[node.current_green]
            emergency_in_current = any(v.is_emergency for v in current_q)
            # Minimum green avoids rapid oscillation, unless an emergency is waiting elsewhere.
            emergency_elsewhere = any(
                any(v.is_emergency for v in node.queues[d])
                for d in DIRECTIONS if d != node.current_green
            )
            if node.green_elapsed >= self.min_green:
                best_direction, score = self._best_next_signal(node)
                should_switch = (
                    best_direction != node.current_green and
                    (node.green_elapsed >= self.max_green or
                     emergency_elsewhere or
                     self._score_direction(node, best_direction) > self._score_direction(node, node.current_green) + 8)
                )
                if should_switch:
                    node.current_green = best_direction
                    node.green_elapsed = 0
                    self.last_decision = f"Search selected {best_direction} (utility {score:.1f})"
            if emergency_in_current and node.green_elapsed < self.max_green:
                self.last_decision = f"Emergency vehicle clearing {node.current_green}"

    def _serve_vehicle(self, node):
        queue = node.queues[node.current_green]
        if queue:
            # Emergency vehicles at the front get served first.
            emergency_index = next((i for i, v in enumerate(queue) if v.is_emergency), None)
            if emergency_index is not None:
                vehicle = queue[emergency_index]
                del queue[emergency_index]
            else:
                vehicle = queue.popleft()
            wait = self.tick - vehicle.arrival_time
            node.total_wait += wait
            node.vehicles_served += 1
            if vehicle.is_emergency:
                node.emergency_served += 1

    def step(self):
        if self.tick >= self.duration:
            return False
        for node in self.nodes:
            self._spawn_vehicles(node)
            self._choose_signal(node)
            self._serve_vehicle(node)
            node.green_elapsed += 1
            row = {
                "time": self.tick,
                "junction": node.name,
                "mode": self.mode,
                "green": node.current_green,
                "queue_length": sum(node.queue_lengths().values()),
                "waiting_total": sum(self.tick - v.arrival_time for q in node.queues.values() for v in q),
                "vehicles_served": node.vehicles_served,
                "emergency_waiting": sum(1 for q in node.queues.values() for v in q if v.is_emergency),
            }
            row.update({f"queue_{d}": node.queue_lengths()[d] for d in DIRECTIONS})
            self.history.append(row)
        self.tick += 1
        return True

    def run(self):
        while self.step():
            pass
        return pd.DataFrame(self.history)

    def summary(self):
        total_served = sum(n.vehicles_served for n in self.nodes)
        total_wait = sum(n.total_wait for n in self.nodes)
        emergency_served = sum(n.emergency_served for n in self.nodes)
        remaining = sum(len(q) for n in self.nodes for q in n.queues.values())
        avg_wait = total_wait / total_served if total_served else 0
        return {
            "Vehicles served": total_served,
            "Average wait (ticks)": round(avg_wait, 2),
            "Emergency vehicles served": emergency_served,
            "Vehicles still queued": remaining,
            "Total accumulated wait": total_wait,
        }

def astar_route(graph, start, goal):
    """A* pathfinding. Graph maps node -> iterable of (neighbor, travel_cost)."""
    heuristic = {node: float("inf") for node in graph}
    # A simple admissible heuristic: minimum number of hops to goal, computed by reverse BFS.
    reverse = {node: [] for node in graph}
    for node, edges in graph.items():
        for neighbor, cost in edges:
            reverse.setdefault(neighbor, []).append(node)
    q = deque([(goal, 0)])
    while q:
        node, distance = q.popleft()
        if heuristic.get(node, float("inf")) > distance:
            heuristic[node] = distance
            for prev in reverse.get(node, []):
                q.append((prev, distance + 1))
    frontier = [(heuristic.get(start, 0), 0, start, [start])]
    best_cost = {start: 0}
    while frontier:
        _, cost_so_far, current, path = heapq.heappop(frontier)
        if current == goal:
            return path, cost_so_far
        if cost_so_far > best_cost.get(current, float("inf")):
            continue
        for neighbor, edge_cost in graph.get(current, []):
            new_cost = cost_so_far + edge_cost
            if new_cost < best_cost.get(neighbor, float("inf")):
                best_cost[neighbor] = new_cost
                heapq.heappush(frontier, (new_cost + heuristic.get(neighbor, 0),
                                          new_cost, neighbor, path + [neighbor]))
    return [], float("inf")

def emergency_demo_route():
    """Small road network used to demonstrate A* emergency routing."""
    graph = {
        "Hospital": [("J1", 2), ("J2", 5)],
        "J1": [("Hospital", 2), ("J3", 2), ("J4", 5)],
        "J2": [("Hospital", 5), ("J3", 1), ("J5", 3)],
        "J3": [("J1", 2), ("J2", 1), ("J4", 2), ("J5", 2)],
        "J4": [("J1", 5), ("J3", 2), ("Destination", 2)],
        "J5": [("J2", 3), ("J3", 2), ("Destination", 1)],
        "Destination": [("J4", 2), ("J5", 1)],
    }
    return astar_route(graph, "Hospital", "Destination")
