# -*- coding: utf-8 -*-
from ple import PLE
import frogger_new
import numpy as np
from pygame.constants import K_w, K_a, K_d, K_s
from collections import defaultdict
import pygame
import time
import pickle
from constants import kPlayCellSize, kPlayWidth, kPlayHeight, kPlayYHomeLimit
import argparse
import logging


class NaiveAgent():
    def __init__(self, actions):
        self.actions = actions

    def pickAction(self, obs, train=False):
        if train:
            N_per_action = {}
            N_sum = 0
            # N is related to the position of the frog and the taken action
            frog_x = int(obs["frog_x"])
            frog_y = int(obs["frog_y"])
            for action in self.actions:
                if (frog_x, frog_y, action) in self.N:
                    N_per_action[action] = self.N[(frog_x, frog_y, action)]
                    N_sum += N_per_action[action]
                else:
                    N_per_action[action] = 0

            max_f = -float("inf")
            for action in self.actions:
                X = self.getFeatureVector(obs, action)
                Q = self.computeQ(X)
                f = self.UCB(Q, N_sum, N_per_action[action])

                logging.debug("N_sum: {}\n".format(N_sum))
                logging.debug("N_action: {}\n".format(N_per_action[action]))
                logging.debug("X: {}\n".format(X))
                logging.debug("Q: {}\n".format(Q))
                logging.debug("f: {}\n".format(f))

                if f > max_f:
                    max_f = f
                    self.last_X = X  # Can save last_Q instead of last_X
                    max_action = action
            # update N
            self.N[(frog_x, frog_y, max_action)] += 1
            self.step += 1
            return max_action
        else:
            max_Q = -float("inf")
            for action in self.actions:
                X = self.getFeatureVector(obs, action)
                Q = self.computeQ(X)
                logging.debug("X: {}\n".format(X))
                logging.debug("Q: {}\n".format(Q))
                if Q > max_Q:
                    max_Q = Q
                    self.last_X = X
                    max_action = action
            return max_action

    def getFeatureVector(self, obs, action):

        def getDist(frog, object):
            if frog.colliderect(object):
                return 0
            else:
                if frog.centerx < object.centerx:
                    dist_x = object.left - frog.right
                else:
                    dist_x = frog.left - object.right
                return dist_x

        frog_x = int(obs['frog_x'])
        frog_y = int(obs['frog_y'])
        if action == K_s:
            frog_y += kPlayCellSize[1]
        elif action == K_d:
            frog_x += kPlayCellSize[0]
        elif action == K_w:
            frog_y -= kPlayCellSize[1]
        elif action == K_a:
            frog_x -= kPlayCellSize[0]
        else:
            pass
        frog = pygame.Rect(frog_x, frog_y, obs['rect_w'], obs['rect_h'])
        logging.debug("frog: {}\n".format(frog))
        if frog.right > kPlayWidth - frog.width / 2.0:  # Avoid boundary
            min_dist = 0.0
        elif frog.left < 0 + frog.width / 2.0:
            min_dist = 0.0
        else:
            dist_car = []
            dist_river = []
            for car in obs['cars']:
                if abs(car.centery - frog.centery) < kPlayCellSize[1] / 2:  # if there are cars in the same row
                    dist = getDist(frog, car)
                    if dist == 0:
                        dist_scaled = -(1 - abs(car.centerx - frog.centerx) / (car.width / 2.0 + frog.width / 2.0))
                    else:
                        dist_scaled = dist * 1.0 / (kPlayWidth - car.width - frog.width)
                    dist_car.append(dist_scaled)
            if len(dist_car) != 0:
                min_dist = min(dist_car)
            else:
                for river in obs['rivers'] + obs['homeR']:  # if there are rivers or homes in the same row
                    if abs(river.centery - frog.centery) < kPlayCellSize[1] / 2:
                        dist = getDist(frog, river)
                        if dist == 0:
                            dist_scaled = 1 - abs(river.centerx - frog.centerx) / (river.width / 2.0 + frog.width / 2.0)
                        else:
                            dist_scaled = - (dist * 1.0 / (kPlayWidth / 2.0 - frog.width))
                        dist_river.append(dist_scaled)
                logging.debug("dist_river: {}\n".format(dist_river))
                if len(dist_river) != 0:
                    min_dist = max(dist_river)
                else:
                    if frog.top > kPlayYHomeLimit:  # Nothing in the same row as the frog
                        min_dist = 1.0
                    else:  # No available home because of crocodile
                        min_dist = -1.0
        if min_dist > 0.2:  # since from 0.2 to 1 are not so different. They are all safe enough for frog.
            min_dist = 1.0
        elif min_dist < - 0.2:
            min_dist = -1.0
        else:
            if min_dist > 0:
                min_dist = min_dist ** 0.5
            else:
                min_dist = -(-min_dist) ** 0.5
        X = [0.0] * (len(self.actions) + 2)
        X[0] = 1.0  # X[0] is always 1
        for i, K in enumerate(self.actions):
            if action == K:
                X[i + 1] = min_dist  # X[i] is the minimum distance if taking the K action

        dist_home_x = []
        min_dist_home = None
        for home in obs['homeR']:
            if frog.colliderect(home):
                min_dist_home = 0
                break
        if min_dist_home is None:
            for home in obs['homeR']:
                dist_x = getDist(frog, home)
                dist_home_x.append(dist_x)
            if len(dist_home_x) != 0:
                min_dist_x = min(dist_home_x)
            else:
                min_dist_x = 0.0
            if frog.top > kPlayYHomeLimit:
                min_dist_y = frog.top - kPlayYHomeLimit
            else:
                min_dist_y = 0.0
            min_dist_home = np.sqrt(min_dist_x ** 2 + min_dist_y ** 2)
        dist_home_scaled = min_dist_home / kPlayHeight
        dist_home_scaled = - 2.0 * dist_home_scaled + 1.0
        X[-1] = dist_home_scaled  # X[-1] is the scaled distance from the nearest available home
        return X

    def update(self, reward, obs):

        if obs is None:
            Q_sample = reward
        else:
            Q_max = -float("inf")
            for action in self.actions:
                X = self.getFeatureVector(obs, action)
                Q = self.computeQ(X)
                if Q > Q_max:
                    Q_max = Q
            Q_sample = reward + self.gamma * Q_max
        self.alpha = 1.0 / (self.step + 1) ** 0.25
        Q_old = self.computeQ(self.last_X)
        logging.debug("Reward: {}\n".format(reward))
        logging.debug("Q_sample: {}\n".format(Q_sample))
        logging.debug("Q_old: {}\n".format(Q_old))
        logging.debug("alpha: {}\n".format(self.alpha))
        logging.debug("last_X: {}\n".format(self.last_X))

        diff = self.alpha * (Q_sample - Q_old)
        self.loss.append((Q_sample - Q_old) ** 2 * 0.5)
        logging.debug("W: {}\n".format(self.W))
        W_new = self.W
        for i in range(len(self.last_X)):
            W_new[i] = self.W[i] + diff * self.last_X[i]
        self.W = W_new
        logging.debug("Updated W: {}\n".format(self.W))

    def qLearningInit(self, fileName=None):
        if fileName is None:
            self.W = [0.0] * (len(self.actions) + 2)
            self.N = defaultdict(int)
            self.loss = []
            self.step = 0
        else:
            self.W = pickle.load(open("save/W_" + fileName + '.pkl', 'rb'))
            self.N = pickle.load(open("save/N_" + fileName + '.pkl', 'rb'))
            self.loss = []
            self.step = pickle.load(open("save/iterNum_" + fileName + '.pkl', 'rb'))
        self.gamma = 0.9

    def computeQ(self, X):
        Q = np.dot(self.W, X)
        return Q

    def UCB(self, Q, N_sum, N):
        return Q + np.sqrt(2*np.log(N_sum+1)/(N+1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained_data", default=None, help="the timestamp of trained data file")
    parser.add_argument('--isTrain', default=False, action='store_true', help="indicator to train the agent")
    parser.add_argument('--converge_threshold', type=float, help="the threshold of episode loss for convergence, by default 0.2")
    parser.add_argument('--episode_number', type=int, help="define the maximum episode number for running this agent")
    args = parser.parse_args()
    t = time.localtime()
    timestamp = time.strftime('%b-%d_%H%M', t)
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        filename='output_' + timestamp + '.log')

    game = frogger_new.Frogger()
    fps = 30
    p = PLE(game, fps=fps, force_fps=False)
    agent = NaiveAgent(p.getActionSet())
    agent.qLearningInit(fileName=args.trained_data)

    reward = None
    loss_episode = float("inf")
    loss_episodes = []
    W_episodes = []
    if args.converge_threshold:
        converge = args.converge_threshold
    else:
        converge = 0.2
    if args.trained_data:
        iterNum = pickle.load(open("save/iterNum_" + args.trained_data + '.pkl', 'rb'))
    else:
        iterNum = 0
    episode = 0
    if args.episode_number:
        episode_number = args.episode_number
    else:
        episode_number = float("inf")
    rewardPosCounter = 0
    rewardNegCounter = 0
    time_start = time.time()
    while loss_episode > converge:
        logging.info("episode {}: \n".format(episode))
        logging.info("Iteration {}: \n".format(iterNum))
        if p.game_over():
            episode += 1
            obs = game.getGameState()
            logging.info("GAME OVER: {}\n".format(obs))
            if args.isTrain:
                if reward is not None:
                    agent.update(reward, obs=None)
                    loss_episode = sum(agent.loss)
                    logging.info("Episode loss: {}\n".format(loss_episode))
                    loss_episodes.append(loss_episode)
                    W_episodes.append(agent.W[:])
                    agent.loss = []
                if not episode % 50:
                    t = time.localtime()
                    timestamp = time.strftime('%b-%d_%H%M', t)
                    pickle.dump(agent.W, open("save/W_" + timestamp + ".pkl", 'wb'))
                    pickle.dump(agent.N, open("save/N_" + timestamp + ".pkl", 'wb'))
                    pickle.dump(loss_episodes, open("save/Loss_" + timestamp + ".pkl", 'wb'))
                    pickle.dump(W_episodes, open("save/Wepisodes_" + timestamp + ".pkl", 'wb'))
                    pickle.dump(iterNum, open("save/iterNum_" + timestamp + ".pkl", "wb"))
            p.reset_game()
            obs = game.getGameState()
            logging.info("obs: {}\n".format(obs))
        else:
            obs = game.getGameState()
            logging.info("obs: {}\n".format(obs))
            if args.isTrain:
                if reward is not None:
                    agent.update(reward, obs)

        if episode >= episode_number:
            break

        action = agent.pickAction(obs, train=args.isTrain)
        logging.info("action taken: {}\n".format(action))
        reward = p.act(action)
        logging.info("Reward: {}\n".format(reward))
        logging.info ("Game score: {}\n".format(game.score))

        if reward > 0.9:
            rewardPosCounter += 1
        elif reward < 0:
            rewardNegCounter += 1
        else:
            pass
        iterNum += 1

    time_end = time.time()
    logging.info("Running time: {} s.\n".format(time_end - time_start))
    logging.info("{} frogs have entered in home.\n".format(rewardPosCounter))
    logging.info("{} frogs have died.\n".format(rewardNegCounter))
    t = time.localtime()
    timestamp = time.strftime('%b-%d_%H%M', t)
    pickle.dump(agent.W, open("save/W_" + timestamp + ".pkl", 'wb'))
    pickle.dump(agent.N, open("save/N_" + timestamp + ".pkl", 'wb'))
    pickle.dump(loss_episodes, open("save/Loss_" + timestamp + ".pkl", 'wb'))
    pickle.dump(W_episodes, open("save/Wepisodes_" + timestamp + ".pkl", 'wb'))
    pickle.dump(iterNum, open("save/iterNum_" + timestamp + ".pkl", "wb"))
