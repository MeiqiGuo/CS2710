# -*- coding: UTF-8 -*-

"""
This script solves puzzles with search algorithm.

Author: Meiqi Guo (meg168@pitt.edu)
Date: 09/12/2018
"""

from collections import deque, defaultdict
import argparse
import os
import ast
import heapq
from dataclasses import dataclass, field
from typing import Any
import numpy as np
import math
import time


@dataclass(order=True)
class PrioritizedItem:
    priority: float
    item: Any=field(compare=False)


class Node(object):
    """
    This class builds the data structure of nodes.
    """
    def __init__(self, state, parent, path_cost):
        self.state = state
        self.parent = parent
        self.path_cost = path_cost


def solution(node, output_file):
    """
    This function writes solution paths in the output file.
    :param node:
    :param output_file:
    :return:
    """
    output_file = open(output_file, 'w')
    solution_path = [node.state]
    output_file.write("Path cost: {}\n".format(node.path_cost))
    while node.parent:
        solution_path.append(node.parent.state)
        node = node.parent
    while len(solution_path):
        state = solution_path.pop()
        output_file.write("{}\n".format(state))
    output_file.close()
    print("Solution path is written in the output file.")
    return


class CheckState(object):
    def __init__(self, state):
        self.state = state

    def if_in_frontier(self, frontier):
        """
        This function checks if the state is already in the frontier list.
        :param frontier:
        :return: True -> the state is already in the frontier list
                 False -> the state is not in the frontier list
        """
        for f in frontier:
            if f.state == self.state:
                return True
        return False

    def if_same_as_parents_states(self, parent):
        """
        Check the state of node against those on the path from the root to its parent node.
        :param parent: the parent node of the checking node.
        :return:True -> the state is same as some parent's state on the path
                False -> the state is different from any parent's state on the path
        """
        while parent:
            if parent.state == self.state:
                return True
            parent = parent.parent
        return False


class BreadthFirstSearch(object):
    """
    This class uses graph-search BFS as search strategy.
    """

    def __init__(self, problem, output_file):
        self.problem = problem
        self.output_file = output_file
        self.time = 0
        self.num_nodes = 0
        self.num_nodes_max = 0
        self.num_explored = 0

    def search(self):
        node = Node(self.problem.init_state, None, 0)
        frontier = deque([node])
        explored = set()
        self.num_nodes += 1
        self.num_nodes_max += 1
        while True:
            if len(frontier) == 0:
                output_file = open(self.output_file, 'w')
                output_file.write("No solution.\n")
                output_file.close()
                return
            node = frontier.popleft()
            self.num_nodes -= 1
            if self.problem.goal_test(node.state):
                solution(node, self.output_file)
                return
            explored.add(node.state)
            self.num_explored += 1
            for (child_state, step_cost) in self.problem.get_children_states(node.state):
                if (child_state not in explored) & (not CheckState(child_state).if_in_frontier(frontier)):
                    path_cost = node.path_cost + step_cost
                    child = Node(child_state, node, path_cost)
                    frontier.append(child)
                    self.time += 1
                    self.num_nodes += 1
            if self.num_nodes_max < self.num_nodes:
                self.num_nodes_max = self.num_nodes

    def complexity(self):
        output_file = open(self.output_file, 'a')
        output_file.write("The total number of nodes created: {}.\n".format(self.time))
        output_file.write("The biggest size of the frontier list grew to: {}.\n".format(self.num_nodes_max))
        output_file.write("The biggest size of the explored list grew to: {}.\n".format(self.num_explored))
        output_file.close()


class DepthFirstSearch(object):
    """
    This class uses tree-search DFS as search strategy.
    """

    def __init__(self, problem, output_file):
        self.problem = problem
        self.output_file = output_file
        self.time = 0
        self.num_nodes = 0
        self.num_nodes_max = 0

    def search(self, cutoff=1e8):
        """

        :param cutoff: if the time complexity is more than cutoff, then the search process will stop.
                        It forces stopping infinite loop because of duplicate states.
        :return:
        """
        node = Node(self.problem.init_state, None, 0)
        frontier = [node]
        self.num_nodes += 1
        self.num_nodes_max += 1
        while self.time < cutoff:
            if len(frontier) == 0:
                output_file = open(self.output_file, 'w')
                output_file.write("No solution.\n")
                output_file.close()
                return
            node = frontier.pop()
            self.num_nodes -= 1
            if self.problem.goal_test(node.state):
                solution(node, self.output_file)
                return
            for (child_state, step_cost) in self.problem.get_children_states(node.state):
                if not CheckState(child_state).if_same_as_parents_states(node):
                    path_cost = node.path_cost + step_cost
                    child = Node(child_state, node, path_cost)
                    frontier.append(child)
                    self.time += 1
                    self.num_nodes += 1
            if self.num_nodes_max < self.num_nodes:
                self.num_nodes_max = self.num_nodes

    def complexity(self):
        output_file = open(self.output_file, 'a')
        output_file.write("The total number of nodes created: {}.\n".format(self.time))
        output_file.write("The biggest size of the frontier list grew to: {}.\n".format(self.num_nodes_max))
        output_file.close()


class DepthLimitedSearch(object):
    """
    This class uses tree-search DLS as search strategy.
    """

    def __init__(self, problem, max_depth):
        self.problem = problem
        self.max_depth = max_depth
        self.time = 0
        self.num_nodes = 0
        self.num_nodes_max = 0

    def search(self):
        node = Node(self.problem.init_state, None, 0)
        frontier = [node]
        self.num_nodes += 1
        self.num_nodes_max += 1
        while True:
            if len(frontier) == 0:
                return False
            node = frontier.pop()
            self.num_nodes -= 1
            if self.problem.goal_test(node.state):
                return node
            if node.path_cost < self.max_depth:
                for (child_state, step_cost) in self.problem.get_children_states(node.state):
                    if not CheckState(child_state).if_same_as_parents_states(node):
                        path_cost = node.path_cost + step_cost
                        child = Node(child_state, node, path_cost)
                        frontier.append(child)
                        self.time += 1
                        self.num_nodes += 1
                if self.num_nodes_max < self.num_nodes:
                    self.num_nodes_max = self.num_nodes


class IterDeepSearch(object):
    """
    This class uses tree-search Iterative Deepening Depth-first Search as search strategy.
    """
    def __init__(self, problem, output_file):
        self.problem = problem
        self.output_file = output_file
        self.time = 0
        self.num_nodes = 0
        self.num_nodes_max = 0

    def search(self):
        depth = 0
        while True:
            dls = DepthLimitedSearch(self.problem, depth)
            result = dls.search()
            self.time += dls.time
            if result:
                solution(result, self.output_file)
                assert(result.path_cost == depth), "This implemented IterDeepSearch algorithm can only be used " \
                                                   "when all step costs are equal."
                self.num_nodes_max = dls.num_nodes_max
                return
            else:
                depth += 1

    def complexity(self):
        output_file = open(self.output_file, 'a')
        output_file.write("The total number of nodes created: {}.\n".format(self.time))
        output_file.write("The biggest size of the frontier list grew to: {}.\n".format(self.num_nodes_max))
        output_file.close()


class UniCostSearch(object):
    """
    This class uses graph-search Uniform-Cost search.
    """
    def __init__(self, problem, output_file):
        self.problem = problem
        self.output_file = output_file
        self.time = 0
        self.num_nodes = 0
        self.num_nodes_max = 0
        self.num_explored = 0

    def search(self):
        node = Node(self.problem.init_state, None, 0)
        frontier = []
        heapq.heappush(frontier, PrioritizedItem(node.path_cost, node))
        explored = set()
        self.num_nodes += 1
        self.num_nodes_max += 1
        while True:
            if len(frontier) == 0:
                output_file = open(self.output_file, 'w')
                output_file.write("No solution.\n")
                output_file.close()
                return
            node = heapq.heappop(frontier).item
            self.num_nodes -= 1
            if self.problem.goal_test(node.state):
                solution(node, self.output_file)
                return
            explored.add(node.state)
            self.num_explored += 1
            for (child_state, step_cost) in self.problem.get_children_states(node.state):
                if child_state not in explored:
                    path_cost = node.path_cost + step_cost
                    child = Node(child_state, node, path_cost)
                    heapq.heappush(frontier, PrioritizedItem(child.path_cost, child))
                    self.time += 1
                    self.num_nodes += 1
            if self.num_nodes_max < self.num_nodes:
                self.num_nodes_max = self.num_nodes

    def complexity(self):
        output_file = open(self.output_file, 'a')
        output_file.write("The total number of nodes created: {}.\n".format(self.time))
        output_file.write("The biggest size of the frontier list grew to: {}.\n".format(self.num_nodes_max))
        output_file.write("The biggest size of the explored list grew to: {}.\n".format(self.num_explored))
        output_file.close()


class GreedySearch(object):
    """
    Graph-search Greedy Search.
    """
    def __init__(self, problem, output_file):
        self.problem = problem
        self.output_file = output_file
        self.time = 0
        self.num_nodes = 0
        self.num_nodes_max = 0
        self.num_explored = 0

    def search(self):
        node = Node(self.problem.init_state, None, 0)
        frontier = []
        heapq.heappush(frontier, PrioritizedItem(self.problem.heuristic_function(node.state), node))
        explored = set()
        self.num_nodes += 1
        self.num_nodes_max += 1
        while True:
            if len(frontier) == 0:
                output_file = open(self.output_file, 'w')
                output_file.write("No solution.\n")
                output_file.close()
                return
            node = heapq.heappop(frontier).item
            self.num_nodes -= 1
            if self.problem.goal_test(node.state):
                solution(node, self.output_file)
                return
            explored.add(node.state)
            self.num_explored += 1
            for (child_state, step_cost) in self.problem.get_children_states(node.state):
                if child_state not in explored:
                    path_cost = node.path_cost + step_cost
                    child = Node(child_state, node, path_cost)
                    child_h = self.problem.heuristic_function(child.state)
                    heapq.heappush(frontier, PrioritizedItem(child_h, child))
                    self.time += 1
                    self.num_nodes += 1
            if self.num_nodes_max < self.num_nodes:
                self.num_nodes_max = self.num_nodes

    def complexity(self):
        output_file = open(self.output_file, 'a')
        output_file.write("The total number of nodes created: {}.\n".format(self.time))
        output_file.write("The biggest size of the frontier list grew to: {}.\n".format(self.num_nodes_max))
        output_file.write("The biggest size of the explored list grew to: {}.\n".format(self.num_explored))
        output_file.close()


class AStarSearch(object):
    """
    Graph-search A* Search.
    """
    def __init__(self, problem, output_file):
        self.problem = problem
        self.output_file = output_file
        self.time = 0
        self.num_nodes = 0
        self.num_nodes_max = 0
        self.num_explored = 0

    def search(self):
        node = Node(self.problem.init_state, None, 0)
        frontier = []
        heapq.heappush(frontier, PrioritizedItem(self.problem.heuristic_function(node.state), node))
        explored = set()
        self.num_nodes += 1
        self.num_nodes_max += 1
        while True:
            if len(frontier) == 0:
                output_file = open(self.output_file, 'w')
                output_file.write("No solution.\n")
                output_file.close()
                return
            node = heapq.heappop(frontier).item
            self.num_nodes -= 1
            if self.problem.goal_test(node.state):
                solution(node, self.output_file)
                return
            explored.add(node.state)
            self.num_explored += 1
            for (child_state, step_cost) in self.problem.get_children_states(node.state):
                if child_state not in explored:
                    path_cost = node.path_cost + step_cost
                    child = Node(child_state, node, path_cost)
                    child_f = self.problem.heuristic_function(child.state) + path_cost
                    heapq.heappush(frontier, PrioritizedItem(child_f, child))
                    self.time += 1
                    self.num_nodes += 1
            if self.num_nodes_max < self.num_nodes:
                self.num_nodes_max = self.num_nodes

    def complexity(self):
        output_file = open(self.output_file, 'a')
        output_file.write("The total number of nodes created: {}.\n".format(self.time))
        output_file.write("The biggest size of the frontier list grew to: {}.\n".format(self.num_nodes_max))
        output_file.write("The biggest size of the explored list grew to: {}.\n".format(self.num_explored))
        output_file.close()


class CostLimitedSearch(object):
    """
    This class uses tree-search Cost Limited Deepening First Search as search strategy.
    """

    def __init__(self, problem, max_cost):
        self.problem = problem
        self.max_cost = max_cost
        self.time = 0
        self.num_nodes = 0
        self.num_nodes_max = 0
        self.next_cutoff = math.inf

    def search(self):
        node = Node(self.problem.init_state, None, 0)
        frontier = [node]
        self.num_nodes += 1
        self.num_nodes_max += 1
        while True:
            if len(frontier) == 0:
                return False, self.next_cutoff
            node = frontier.pop()
            self.num_nodes -= 1
            f_node = self.problem.heuristic_function(node.state) + node.path_cost
            if f_node <= self.max_cost:
                if self.problem.goal_test(node.state):
                    return node, self.next_cutoff
                for (child_state, step_cost) in self.problem.get_children_states(node.state):
                    if not CheckState(child_state).if_same_as_parents_states(node):
                        path_cost = node.path_cost + step_cost
                        child = Node(child_state, node, path_cost)
                        frontier.append(child)
                        self.time += 1
                        self.num_nodes += 1
                if self.num_nodes_max < self.num_nodes:
                    self.num_nodes_max = self.num_nodes
            else:
                if f_node < self.next_cutoff:
                    self.next_cutoff = f_node


class IDAStarSearch(object):
    """
    Tree-search Iterative-Deepening A* Search
    """
    def __init__(self, problem, output_file):
        self.problem = problem
        self.output_file = output_file
        self.time = 0
        self.num_nodes = 0
        self.num_nodes_max = 0

    def search(self):
        max_cost = self.problem.heuristic_function(self.problem.init_state)
        while True:
            cls = CostLimitedSearch(self.problem, max_cost)
            result, max_cost = cls.search()
            self.time += cls.time
            if result:
                solution(result, self.output_file)
                self.num_nodes_max = cls.num_nodes_max
                return
            if max_cost == math.inf:
                output_file = open(self.output_file, 'w')
                output_file.write("No solution.\n")
                output_file.close()
                return

    def complexity(self):
        output_file = open(self.output_file, 'a')
        output_file.write("The total number of nodes created: {}.\n".format(self.time))
        output_file.write("The biggest size of the frontier list grew to: {}.\n".format(self.num_nodes_max))
        output_file.close()


class WaterJugsPb(object):
    """
    Water Jugs Problem
    """
    def __init__(self, conf, heuri_key=None):
        self.conf = open(conf, 'r')
        assert(self.conf.readline().strip() == 'jugs'), 'Verify the configuration file is correct.'
        self.capacities = ast.literal_eval(self.conf.readline())
        self.init_state = ast.literal_eval(self.conf.readline())
        self.goal_state = ast.literal_eval(self.conf.readline())
        self.num_jugs = len(self.capacities)
        if heuri_key:
            heuri_map = {'sum': self.heuristic_sum_water_in_jugs}
            try:
                self.heuristic_function = heuri_map[heuri_key]
            except KeyError:
                print("There is no heuristic function {} in this problem.".format(heuri_key))

    def goal_test(self, state):
        """
        The goal test function: state => True/False
        :param state:
        :return:
        """
        if state == self.goal_state:
            return True
        else:
            return False

    def get_children_states(self, state):
        """
        This function returns all possible successor states as well as the step cost.
        (state) => a list of (child_state, step_cost)
        :param state:
        :return:
        """
        list_children_states = []
        for i in range(self.num_jugs):

            for v in [0, self.capacities[i]]:
                # we can dump each jug on the ground or fill it from tap
                new_state = list(state)
                new_state[i] = v
                new_state = tuple(new_state)
                if new_state != state:  # if the ith jug is already empty/full, there is no change of state
                    list_children_states.append((new_state, 1))

            for j in range(self.num_jugs):
                # we can pour it to another jug
                if i == j:
                    continue
                else:
                    v = min(state[i], self.capacities[j] - state[j])
                    if v:
                        # if the ith jug is already empty or the jth jug is already full, there is no change of state
                        new_state = list(state)
                        new_state[i] -= v
                        new_state[j] += v
                        new_state = tuple(new_state)
                        list_children_states.append((new_state, 1))

        return list_children_states

    def heuristic_sum_water_in_jugs(self, state):
        """
        This heuristic function computes the total gallons of water in jugs in the current state. H(x) is equal to the
        difference between current state and goal state.
        :param state:
        :return:
        """
        return abs(sum(state) - sum(self.goal_state))


class PathPlanPb(object):
    """
    Path Planning Problem
    """
    def __init__(self, conf, heuri_key=None):
        self.conf = open(conf, 'r')
        lines = self.conf.readlines()
        assert(lines[0].strip() == 'cities'), 'Verify the configuration file is correct.'
        self.cities = {}
        for city in ast.literal_eval(lines[1]):
            self.cities[city[0]] = (city[1], city[2])
        self.init_state = ast.literal_eval(lines[2])
        self.goal_state = ast.literal_eval(lines[3])
        self.paths = defaultdict(list)
        for line in lines[4:]:
            path = ast.literal_eval(line)
            self.paths[path[0]].append((path[1], path[2]))
            self.paths[path[1]].append((path[0], path[2]))
        if heuri_key:
            heuri_map = {'sld': self.HeuristicSLD}
            try:
                self.heuristic_function = heuri_map[heuri_key]
            except KeyError:
                print("There is no heuristic function {} in this problem.".format(heuri_key))

    def goal_test(self, state):
        """
        The goal test function: state => True/False
        :param state:
        :return:
        """
        if state == self.goal_state:
            return True
        else:
            return False

    def get_children_states(self, state):
        """
        This function returns all possible successor states as well as the step cost.
        (state) => a list of (child_state, step_cost)
        :param state:
        :return:
        """
        list_children_states = self.paths[state]
        return list_children_states

    def HeuristicSLD(self, state):
        """
        This is the straight-line distance heuristic function.
        :param state:
        :return:
        """
        state_position = self.cities[state]
        goal_position = self.cities[self.goal_state]
        h = np.sqrt((state_position[0] - goal_position[0]) ** 2 + (state_position[1] - goal_position[1]) ** 2)
        return h


class PancakePb(object):
    """
    Burnt Pancakes Problem
    """
    def __init__(self, conf, heuri_key=None):
        self.conf = open(conf, 'r')
        lines = self.conf.readlines()
        assert(lines[0].strip() == 'pancakes'), 'Verify the configuration file is correct.'
        self.init_state = tuple(ast.literal_eval(lines[1]))
        self.goal_state = tuple(ast.literal_eval(lines[2]))
        self.pancake_num = len(self.init_state)
        #TODO
        if heuri_key:
            heuri_map = {'bp': self.HeuristicBP}
            try:
                self.heuristic_function = heuri_map[heuri_key]
            except KeyError:
                print("There is no heuristic function {} in this problem.".format(heuri_key))

    def goal_test(self, state):
        """
        The goal test function: state => True/False
        :param state:
        :return:
        """
        for i in range(self.pancake_num):
            if state[i] == i + 1:
                continue
            else:
                return False
        return True

    def get_children_states(self, state):
        """
        This function returns all possible successor states as well as the step cost.
        (state) => a list of (child_state, step_cost)
        :param state:
        :return:
        """
        list_children_states = []
        state = list(state)
        for i in range(self.pancake_num, 0, -1):
            flip_part = state[:i]
            flip_part.reverse()
            flip_part = [-x for x in flip_part]
            children_state = flip_part + state[i:]
            list_children_states.append((tuple(children_state), 1))
        return list_children_states

    def HeuristicBP(self, state):
        """
        This is the breakpoints heuristic function.
        :param state:
        :return:
        """
        bp = 0
        for i in range(self.pancake_num - 1):
            if abs(state[i] - state[i+1]) != 1:
                bp += 1
        if state[self.pancake_num - 1] != self.pancake_num:
            bp += 1
        return bp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("puzzle", help="configuration file for the puzzle your program has to solve")
    parser.add_argument("algorithm", help="keyword that specifies which search algorithm to use to solve the puzzle",
                        choices=['bfs', 'dfs', 'iddfs', 'unicost', 'greedy', 'astar', 'idastar'])
    parser.add_argument("--heuristic_function", help="specify different heuristic functions for ​informed search",
                        choices=['sld', 'sum', 'bp'])
    args = parser.parse_args()
    pb_map = {'jugs': WaterJugsPb, 'cities': PathPlanPb, 'pancakes': PancakePb}
    algo_map = {'bfs': BreadthFirstSearch, 'dfs': DepthFirstSearch, 'iddfs': IterDeepSearch, 'unicost': UniCostSearch,
                'greedy': GreedySearch, 'astar': AStarSearch, 'idastar': IDAStarSearch}
    puzzle_name = open(args.puzzle, 'r').readline().strip()
    time_start = time.time()
    if args.heuristic_function:
        print("Begin solving {0} puzzle by {1} with {2}...".format(puzzle_name, args.algorithm, args.heuristic_function))
        problem = pb_map[puzzle_name](args.puzzle, args.heuristic_function)
        search_algo = algo_map[args.algorithm](problem, os.path.join('output',
                                                                     args.puzzle.split('.')[0].split('_')[1]
                                                                     + '_by_' + args.algorithm + '_with_' +
                                                                     args.heuristic_function + '.output'))
    else:
        print("Begin solving {0} puzzle by {1}...".format(puzzle_name, args.algorithm))
        problem = pb_map[puzzle_name](args.puzzle)
        search_algo = algo_map[args.algorithm](problem, os.path.join('output',
                                                                     args.puzzle.split('.')[0].split('_')[1] + '_by_'
                                                                     + args.algorithm + '.output'))

    #print(problem.init_state)
    #print(problem.get_children_states(problem.init_state))
    #print(problem.goal_test(problem.goal_state))
    #print(problem.HeuristicBP(problem.init_state))
    search_algo.search()
    search_algo.complexity()
    time_end = time.time()
    print("Running time: {}s".format(time_end - time_start))

