import numpy as np
import random

class FastHVCalculator:
    @staticmethod
    def calculate_hv_contribution(new_point, existing_points, ref_point):
        if len(existing_points) == 0:
            return np.prod(np.maximum(0, ref_point - new_point))
        for point in existing_points:
            if all(point <= new_point) and any(point < new_point):
                return 0.0
        dominated_volume = np.prod(np.maximum(0, ref_point - new_point))
        overlap = 0.0
        for point in existing_points:
            overlap_dims = np.minimum(ref_point - new_point, ref_point - point)
            overlap += np.prod(np.maximum(0, overlap_dims)) * 0.5
        return max(0, dominated_volume - overlap)

class FastPreferenceSampler:
    def __init__(self):
        self.extreme_prefs = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0])
        ]
        self.boundary_prefs = [
            np.array([0.7, 0.3, 0.0]),
            np.array([0.3, 0.7, 0.0]),
            np.array([0.7, 0.0, 0.3]),
            np.array([0.3, 0.0, 0.7]),
            np.array([0.0, 0.7, 0.3]),
            np.array([0.0, 0.3, 0.7])
        ]

    def sample(self, pareto_front, episode, total_episodes=None):
        if total_episodes is None:
            total_episodes = max(episode + 1, 1000)

        progress = episode / max(total_episodes - 1, 1)

        if progress < 0.4:
            if random.random() < 0.8:
                return random.choice(self.extreme_prefs).copy()
            else:
                return random.choice(self.boundary_prefs).copy()
        elif progress < 0.8:
            r = random.random()
            if r < 0.35:
                return random.choice(self.extreme_prefs).copy()
            elif r < 0.70:
                return random.choice(self.boundary_prefs).copy()
            else:
                return np.random.dirichlet([1.5, 1.5, 1.5])
        else:
            if random.random() < 0.3:
                return random.choice(self.extreme_prefs).copy()
            else:
                return np.random.dirichlet([1.5, 1.5, 1.5])

    def get_phase_name(self, episode, total_episodes):
        progress = episode / max(total_episodes - 1, 1)
        if progress < 0.4:
            return "early"
        elif progress < 0.8:
            return "mid"
        else:
            return "late"

class LightweightHistoricalFront:
    def __init__(self, max_size=5000):
        self.objectives = []
        self.max_size = max_size
        self.min_vals = None
        self.max_vals = None

    def add_solution(self, objectives, schedule_info=None):
        if any(np.isnan(objectives)) or any(np.isinf(objectives)):
            return False

        obj = np.array(objectives)

        if len(self.objectives) > 0:
            all_obj = np.array(self.objectives)
            self.min_vals = np.minimum(all_obj.min(axis=0), obj)
            self.max_vals = np.maximum(all_obj.max(axis=0), obj)
        else:
            self.min_vals = obj.copy()
            self.max_vals = obj.copy()

        self.objectives.append(obj)

        if len(self.objectives) > self.max_size:
            remove_count = int(self.max_size * 0.2)
            indices_to_keep = random.sample(range(len(self.objectives)),
                                            len(self.objectives) - remove_count)
            self.objectives = [self.objectives[i] for i in sorted(indices_to_keep)]

        return True

    def get_objectives_array(self):
        if len(self.objectives) == 0:
            return np.array([])
        return np.array(self.objectives)

    def size(self):
        return len(self.objectives)

class FastParetoFront:
    def __init__(self, max_size=1000):
        self.objectives = []
        self.schedule_info = []
        self.max_size = max_size
        self.min_vals = None
        self.max_vals = None
        self.check_interval = 10
        self.add_count = 0

    def add_solution(self, objectives, schedule_info=None):
        if any(np.isnan(objectives)) or any(np.isinf(objectives)):
            return False

        obj = np.array(objectives)

        if len(self.objectives) > 0:
            all_obj = np.array(self.objectives)
            self.min_vals = np.minimum(all_obj.min(axis=0), obj)
            self.max_vals = np.maximum(all_obj.max(axis=0), obj)
        else:
            self.min_vals = obj.copy()
            self.max_vals = obj.copy()

        self.add_count += 1

        if self.add_count % self.check_interval != 0:
            if len(self.objectives) < self.max_size:
                self.objectives.append(obj)
                self.schedule_info.append(schedule_info)
                return True
            else:
                idx = random.randint(0, len(self.objectives) - 1)
                self.objectives[idx] = obj
                self.schedule_info[idx] = schedule_info
                return True

        for existing_obj in self.objectives:
            if self._dominates(existing_obj, obj):
                return False

        to_remove = []
        for i, existing_obj in enumerate(self.objectives):
            if self._dominates(obj, existing_obj):
                to_remove.append(i)
        for i in sorted(to_remove, reverse=True):
            self.objectives.pop(i)
            self.schedule_info.pop(i)

        self.objectives.append(obj)
        self.schedule_info.append(schedule_info)

        if len(self.objectives) > self.max_size:
            keep_indices = random.sample(range(len(self.objectives)), self.max_size)
            self.objectives = [self.objectives[i] for i in keep_indices]
            self.schedule_info = [self.schedule_info[i] for i in keep_indices]

        return True

    def _dominates(self, obj1, obj2):
        return all(o1 <= o2 for o1, o2 in zip(obj1, obj2)) and \
            any(o1 < o2 for o1, o2 in zip(obj1, obj2))

    def get_objectives_array(self):
        if len(self.objectives) == 0:
            return np.array([])
        return np.array(self.objectives)

    def get_normalized_objectives_array(self):
        objectives = self.get_objectives_array()
        if len(objectives) == 0:
            return np.array([])
        ranges = self.max_vals - self.min_vals
        ranges[ranges == 0] = 1.0
        return (objectives - self.min_vals) / ranges

    def size(self):
        return len(self.objectives)

    def get_top_solutions(self, n=10):
        if len(self.objectives) == 0:
            return []

        objs = self.get_objectives_array()
        schedules = self.schedule_info

        results = []
        for i, (obj, sched) in enumerate(zip(objs, schedules)):
            if obj[0] > 0 and obj[1] > 0 and obj[2] > 0:
                results.append({
                    'index': i,
                    'objectives': obj,
                    'schedule_info': sched,
                    'makespan': obj[0],
                    'energy': obj[1],
                    'delay': obj[2]
                })

        results.sort(key=lambda x: (x['makespan'], x['energy'], x['delay']))

        return results[:min(n, len(results))]
