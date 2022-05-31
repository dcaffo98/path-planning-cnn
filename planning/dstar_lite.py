
# -*- coding: UTF-8 -*-

'''
D* lite simulation code
Botao Hu, Guanya Shi, Yukai Liu
CS133 Robotics Final Project
All rights reserved
'''

import numpy as np
import heapq
from math import sin, cos
import cv2


def point_warping(r, c, start, goal, side, h, w, theta):
    dirs = np.where(start - goal > 0, 1, -1)
    dirs = np.where(start - goal == 0, 0, dirs)    
    # point = [r + goal[0] + (-2 if dirs[0] <= 0 else +2), c + goal[1] + (-2 if dirs[1] <= 0 else +2)]
    point = [r + goal[0], c + goal[1]]
    point[0] -= (side if dirs[0] < 0 else 0)
    point[1] -= (side if dirs[1] < 0 else 0)
    if dirs[0] >= 0:
        point[0] -= (side / 2.0)
    else:
        point[0] += (side / 2.0)
    # rotation about the goal
    # we invert theta since we are referring to an angle in the cartesian plane, while we're working in a plane with the y-axis being reversed
    point[0] -= goal[0]
    point[1] -= goal[1]
    tmp = [p for p in point]
    point[0] = tmp[1] * sin(-theta) + tmp[0] * cos(theta)
    point[1] = tmp[1] * cos(theta) - tmp[0] * sin(-theta)
    point[0] += goal[0]
    point[1] += goal[1]
    # adjust directions
    if dirs[0] < 0 and dirs[1] < 0:
        point[0] += (side * sin(-theta))
    if dirs[1] < 0:
        point[1] += (side * cos(theta))    
        if dirs[0] > 0:
            point[0] += (side * sin(-theta))
    point[0] = max(0, min(h - 1, int(round(point[0]))))
    point[1] = max(0, min(w - 1, int(round(point[1]))))
    return point

'''
This class used to store data in priority queue.
Comparing methods are overloaded.
'''
class Element:
    def __init__(self, key, value1, value2):
        self.key = key
        self.value1 = value1
        self.value2 = value2

    def __eq__(self, other):
        return np.sum(np.abs(self.key - other.key)) == 0
    
    def __ne__(self, other):
        return not self == other

    def __lt__(self, other):
        return ((self.value1, self.value2) < (other.value1, other.value2))

    def __le__(self, other):
        return ((self.value1, self.value2) <= (other.value1, other.value2))

    def __gt__(self, other):
        return ((self.value1, self.value2) > (other.value1, other.value2))

    def __ge__(self, other):
        return ((self.value1, self.value2) >= (other.value1, other.value2))

'''
Algorithm class
'''
class DStarLite:
    
    def __init__(self, map, x_goal, y_goal, x_start, y_start, max_it=100000, verbose=True, obst_margin=1, goal_margin=1, max_it_sp=5000):
        # initialize
        self.start = np.array([x_start, y_start])
        # original starting point, won't be touched throughout the algorithm
        self._original_start = self.start.copy()
        # starting point from the last map update
        self._last_update_start = self.start.copy()
        self.goal = np.array([x_goal, y_goal])
        self._goal = self.goal.copy()
        self.global_map = map
        self.obst_margin = obst_margin
        self.goal_margin = goal_margin
        self.manage_goal()
        self.max_it = max_it
        self.verbose = verbose
        self.max_it_sp = max_it_sp
        self.init()


    def unfeasible(self):
        """
        Map is deemed unfeasible if there aren't 2 conitguous free cells within the target neighborhood 
        """
        coords = np.mgrid[max(0, self.goal[0] - 1):self.goal[0] + 2, max(0, self.goal[1] - 1):self.goal[1] + 2]
        coords = np.stack((coords[0].ravel(), coords[1].ravel())).T
        for i in range(len(coords)):
            if self.global_map[coords[i, 0], coords[i, 1]] != np.inf and np.sum(np.abs(coords[i] - self.goal)) != 0:
                for j in range(len(coords)):
                    if i != j and self.global_map[coords[j, 0], coords[j, 1]] != np.inf and np.sum(np.abs(coords[j] - self.goal)) != 0:
                        norm_ = np.sum(np.abs(coords[i] - coords[j]))
                        if norm_ > 0 and norm_ <= 1:
                            return False
        return True
    
    def init(self, restart=False):
        self.queue = []
        self.k_m = 0
        self.rhs = np.full(self.global_map.shape, np.inf, dtype=np.float)
        self.g = self.rhs.copy()
        self.rhs[self.goal[0], self.goal[1]] = 0
        heapq.heappush(self.queue, Element(self.goal, *self.calculate_key(self.goal)))
        self.path = []
        self._step = 0
        self.last = None
        if restart:
            self._goal = self.goal
            self.last = self.start
      

    def move_goal_unfeasible(self, goal=None, margin=None):
        goal = goal if goal is not None else self.goal
        margin = margin if margin is not None else self.obst_margin
        self.goal = self.first_avlb(goal, margin=margin)
        if self.goal is None:
            raise ValueError("Goal is on an obstacle and cannot find a nearby point with current margin constraint")
        print("Goal moved from [{}, {}] to [{}, {}]".format(goal[0], goal[1], self.goal[0], self.goal[1]))

    def manage_goal(self):
        if self.unfeasible():
            print("Unfeasible path with current goal. Relocating...")
            self.move_goal_unfeasible(self._goal, self.obst_margin)
        need_move, goal_tmp = self.move_goal()
        if need_move:
            print("For safety, goal moved from [{}, {}] to [{}, {}]".format(self.goal[0], self.goal[1], goal_tmp[0], goal_tmp[1]))
            self.goal = goal_tmp

    def move_goal(self, k=2):
        h, w = self.global_map.shape
        diff = self.start - self.goal
        dx = dy = None
        # should be -diff[0] with negative sin
        theta = np.arctan2(diff[0], diff[1])
        max_r = int(round(np.linalg.norm(diff)))
        flag = False
        for r in range(1, max_r):
            dx = int(max(0, min(w - 1, round(r * cos(theta) + self.goal[1]))))
            dy = int(max(0, min(h - 1, round(r * sin(theta) + self.goal[0]))))
            obst = self.global_map[dy, dx] == np.inf
            flag |= obst
            if not obst and r >= k:
                break
            # flag = True -> move goal; flag = False -> move not required
        return flag, np.array((dy, dx))


    def first_avlb(self, u, v=None, margin=None):
        if v is None:
            v = self._last_update_start
        if margin is None:
            margin = self.obst_margin
        ret = None
        h, w = self.global_map.shape
        diff = self.start - self.goal
        theta = np.arctan2(-diff[0], diff[1])
        max_side = max(abs(diff[0]), abs(diff[1]))
        side = 3
        while side < max_side:
            square = []
            rows = [-1, side]
            cols = [-1, side]
            for i in range(2):
                for j in range(-1, side + 1):
                    point = point_warping(rows[i], j, self.start, self.goal, side, h, w, theta)
                    if self.global_map[point[0], point[1]] != np.inf:
                        square.append(point)
                for j in range(side):
                    point = point_warping(j, cols[i], self. start, self.goal, side, h, w, theta)
                    if self.global_map[point[0], point[1]] != np.inf:
                        square.append(point)
            square = np.array(square)
            min_dist = np.inf
            for i, point in enumerate(square):
                if point[0] != u[0] and point[1] != u[1] and not self.near_obstacles(point, margin):
                    dist = np.sum(np.abs(point - v))
                    if dist < min_dist:
                        min_dist = dist
                        ret = point
            if ret is not None:
                break
            if side == max_side - 1:
                max_side = min(h, w)
            side += 1
        return ret
    

    def calculate_key(self, s):
        key = [0, 0]
        key[0] = min(self.g[s[0],s[1]], self.rhs[s[0],s[1]]) + self.h_estimate(self.start, s) + self.k_m
        key[1] = min(self.g[s[0],s[1]], self.rhs[s[0],s[1]])
        return key
    

    def update_vertex(self, u):
        if np.sum(np.abs(u - self.goal)) != 0:
            s_list = self.succ(u)
            min_s = np.inf
            for s in s_list:
                if self.cost(u, s) + self.g[s[0],s[1]] < min_s:
                    min_s = self.cost(u, s) + self.g[s[0],s[1]]
            self.rhs[u[0],u[1]] = min_s
        if Element(u, 0, 0) in self.queue:
            self.queue.remove(Element(u, 0, 0))
            heapq.heapify(self.queue)
        if self.g[u[0],u[1]] != self.rhs[u[0],u[1]]:
            heapq.heappush(self.queue, Element(u, *self.calculate_key(u)))


    def shortest_path(self):
        i = 0
        while i < self.max_it_sp and len(self.queue) > 0 and heapq.nsmallest(1, self.queue)[0] < Element(self.start, *self.calculate_key(self.start)) \
                or self.rhs[self.start[0], self.start[1]] != self.g[self.start[0], self.start[1]]:
            k_old = heapq.nsmallest(1, self.queue)[0]
            u = heapq.heappop(self.queue).key
            temp = Element(u, *self.calculate_key(u))
            if k_old < temp:
                heapq.heappush(self.queue, temp)
            elif self.g[u[0],u[1]] > self.rhs[u[0],u[1]]:
                self.g[u[0],u[1]] = self.rhs[u[0],u[1]]
                s_list = self.succ(u)
                for s in s_list:
                    self.update_vertex(s)
            else:
                self.g[u[0],u[1]] = np.inf
                s_list = self.succ(u)
                s_list.append(u)
                for s in s_list:
                    self.update_vertex(s)
            i += 1
            if self.verbose:
                print("shortest_path step done ", i)
    

    # heuristic estimation
    def h_estimate(self, s1, s2):
        return np.linalg.norm(s1 - s2)


    def near_obstacles(self, u, margin=None):
        if margin is None:
            margin = self.obst_margin
        neighborhood = self.global_map[max(0, u[0] - margin):u[0] + margin + 1, max(0, u[1] - margin):u[1] + margin + 1]
        if np.any(neighborhood == np.inf):
            return True
        else:
            return False


    # fetch successors and predessors
    def succ(self, u):
        s_list = [ 
            np.array([u[0]-1,u[1]-1]), np.array([u[0]-1,u[1]]), np.array([u[0]-1,u[1]+1]), np.array([u[0],u[1]-1]), 
            np.array([u[0],u[1]+1]), np.array([u[0]+1,u[1]-1]), np.array([u[0]+1,u[1]]), np.array([u[0]+1,u[1]+1])
        ]
        row, col = self.global_map.shape
        real_list = []
        for s in s_list:
            if s[0] >= 0 and s[0] < row and s[1] >= 0 and s[1] < col:
                real_list.append(s)
        return real_list


    def invalid_diag_move(self, u1, u2):
        # deprecated for the time being
        if u1[0] > u2[0]:
            if u1[1] > u2[1]:
                v1 = np.array([u1[0], u1[1] - 1])
                v2 = np.array([u1[0] - 1, u1[1]])
            else:
                v1 = np.array([u1[0], u1[1] + 1])
                v2 = np.array([u1[0] - 1, u1[1]])
        else:
            if u1[1] > u2[1]:
                v1 = np.array([u1[0], u1[1] - 1])
                v2 = np.array([u1[0] + 1, u1[1]])
            else:
                v1 = np.array([u1[0], u1[1] + 1])
                v2 = np.array([u1[0] + 1, u1[1]])
        if self.global_map[v1[0], v1[1]] == np.inf or self.global_map[v2[0], v2[1]] == np.inf:
            return True
        return False


    def diagonally_adjacent(self, u1, u2):
        return np.all(np.abs(u1 - u2) == 1)


    # calculate cost between nodes
    def cost(self, u1, u2):
        if self.global_map[u1[0], u1[1]] == np.inf or \
           (self.global_map[u2[0], u2[1]] == np.inf and not self.within_margin(self.goal, u2, 0)) or \
           (not self.within_margin(self.goal, u2) and not self.within_margin(self._last_update_start, u2) and self.near_obstacles(u2)):
            c = np.inf
        else:
            c = self.h_estimate(u1, u2)
        return c                
                

    # update map information and replan
    def scan(self, last, new_map):
        changed_nodes = np.argwhere(self.global_map != new_map)
        if np.any(changed_nodes):
            self.global_map = new_map
            if self.unfeasible():
                self.move_goal_unfeasible(self.goal, self.obst_margin)
                need_move, new_goal = self.move_goal()
                self.goal = new_goal if need_move else self.goal 
                self.init(restart=True)
                last = self.start
            else:
                self.k_m += self.h_estimate(last, self.start)
                last = self.start.copy()
                for s in changed_nodes:
                    self.update_vertex(s)
            self._last_update_start = self.start
            self.shortest_path()
        return last

    
    def within_margin(self, u, v, margin=None):
        if margin is None:
            margin = self.obst_margin
        if np.linalg.norm(u - v) <= margin:
            return True
        else:
            return False


    def goal_reached(self, u=None, margin=None):
        if u is None:
            u = self.start
        if margin is None:
            margin = self.goal_margin
        return self.within_margin(self.goal, u, margin)


    def step(self, new_map=None):
        if new_map is None:
            new_map = self.global_map
        if self._step == 0:
            self.last = self.start
            self.last = self.scan(self.last, new_map)
            self.path.append(self.start)
            self.shortest_path()
            ret = self.path[-1]
        elif self._step < self.max_it:
            ret = None
            temp = None
            self.last = self.scan(self.last, new_map)
            if not self.goal_reached():
                if self.verbose:
                    print("curr_location:", self.start)
                s_list = self.succ(self.start)
                min_s = np.inf
                for s in s_list:
                    if self.cost(self.start, s) + self.g[s[0],s[1]] < min_s:
                        min_s = self.cost(self.start, s) + self.g[s[0],s[1]]
                        temp = s
                if temp is None or self.global_map[temp[0], temp[1]] == np.inf:
                    raise ValueError("Cannot find a feasible path")
                self.start = temp.copy()
                self.path.append(self.start)
                ret = self.start
        else:
            raise ValueError("Cannot find a feasible path within given amount of steps")
        self._step += 1
        return ret

    @staticmethod
    def get_ds_map(map, k=None):
        if k is not None:
            map = cv2.dilate(map.copy(), np.ones((k, k)))
        return np.where(map > 0, np.inf, 0)
