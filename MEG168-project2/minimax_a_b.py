# -*- coding: UTF-8 -*-

"""
This script searches tree by minimax algorithm.

Author: Meiqi Guo (meg168@pitt.edu)
Date: 10/04/2018
"""

import argparse
import ast
import math


class AlphaBetaSearch(object):
    def __init__(self, game_tree):
        self.game_tree = game_tree

    def main(self, state):
        alpha = -math.inf
        beta = math.inf
        v, action_state = self.max_value(state, alpha, beta)
        return v, action_state

    def max_value(self, state, alpha, beta):
        print("Visited state: {0}.".format(state.state))
        if self.game_tree.terminal_test(state):
            return state.utility
        v = -math.inf
        return_state = None
        for next_state in self.game_tree.get_next_states(state):
            next_v = self.min_value(next_state, alpha, beta)
            if v < next_v:
                v = next_v
                return_state = next_state.state

            if v >= beta:
                return v, return_state
            alpha = max(alpha, v)
        return v, return_state

    def min_value(self, state, alpha, beta):
        print("Visited state: {0}.".format(state.state))
        if self.game_tree.terminal_test(state):
            return state.utility
        v = math.inf
        for next_state in self.game_tree.get_next_states(state):
            next_v, _ = self.max_value(next_state, alpha, beta)
            v = min(v, next_v)
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v


class MiniMaxSearch(object):
    def __init__(self, game_tree):
        self.game_tree = game_tree

    def main(self, state):
        v, action_state = self.max_value(state)
        return v, action_state

    def max_value(self, state):
        print("Visited state: {0}.".format(state.state))
        if self.game_tree.terminal_test(state):
            return state.utility
        v = -math.inf
        return_state = None
        for next_state in self.game_tree.get_next_states(state):
            next_v = self.min_value(next_state)
            if v < next_v:
                v = next_v
                return_state = next_state.state
        return v, return_state

    def min_value(self, state):
        print("Visited state: {0}.".format(state.state))
        if self.game_tree.terminal_test(state):
            return state.utility
        v = math.inf
        for next_state in self.game_tree.get_next_states(state):
            next_v, _ = self.max_value(next_state)
            v = min(v, next_v)
        return v


class GameTree(object):
    def __init__(self, nested_list):
        self.root = self.build_tree_from_list(nested_list)

    def build_tree_from_list(self, nested_list):
        if type(nested_list[1]) == list:
            parent = State(nested_list[0])
            for element in nested_list[1:]:
                child = self.build_tree_from_list(element)
                parent.add_child(child)
            return parent
        else:
            parent = State(nested_list[0])
            for element in nested_list[1:]:
                child = State(element[0], element[1])
                parent.add_child(child)
            return parent

    def get_next_states(self, state):
        return state.children

    def terminal_test(self, state):
        if state.utility is not None:
            return True
        else:
            return False


class State(object):
    """
    This class builds the data structure of State.
    """
    def __init__(self, state, utility=None):
        self.state = state
        self.children = []
        self.utility = utility

    def add_child(self, child):
        self.children.append(child)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game_tree", help="configuration file for the game tree your program has to search")
    args = parser.parse_args()
    game_tree_list = ast.literal_eval(open(args.game_tree, 'r').readline())
    game_tree = GameTree(game_tree_list)
    print("Begin searching game tree by vanilla minimax algorithm...")
    final_utility, action_state = MiniMaxSearch(game_tree).main(game_tree.root)
    print("Final utility is {0}.\n Action is to go to state {1}.\n".format(final_utility, action_state))
    print("Begin searching game tree by alpha-beta search algorithm...")
    final_utility, action_state = AlphaBetaSearch(game_tree).main(game_tree.root)
    print("Final utility is {0}.\n Action is to go to state {1}.\n".format(final_utility, action_state))



