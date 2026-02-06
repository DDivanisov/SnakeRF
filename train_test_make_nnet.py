import torch.nn as nn
from snake_game import SnakeGame
import numpy as np
from collections import deque

class ScoreMaker:
    def __init__(self, epsilon=1.0, epsilon_min=0.03, epsilon_decay=0.9985):
        # For training
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.epsilons = []

        self.recent_scores = deque(maxlen=100)
        self.avg_score = 0
        self.avg_scores = []
        
        self.recent_moves = deque(maxlen=100)
        self.avg_move = 0
        self.avg_moves = []
        #For testing

    def epsilone_decay_step(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_epsilon(self):
        return self.epsilon

    def add_score_moves_training(self, score, moves):
        self.recent_scores.append(score)
        self.recent_moves.append(moves)
    
    def get_average_score_moves_training(self):
        self.avg_score = np.mean(self.recent_scores) if self.recent_scores else 0
        self.avg_scores.append(self.avg_score)

        self.avg_move = np.mean(self.recent_moves) if self.recent_moves else 0
        self.avg_moves.append(self.avg_move)

        self.epsilons.append(self.epsilon)

        return self.avg_score, self.avg_move
    
    def get_training_statistic(self):
        return self.avg_scores, self.avg_moves, self.epsilons

def train_model(agent, env:SnakeGame, view_type, num_episodes=2000, round2=False):
    if not round2:
        score_maker = ScoreMaker(epsilon=1.0, epsilon_min=0.03, epsilon_decay=0.9985)
    else:
        score_maker = ScoreMaker(epsilon=0.03, epsilon_min=0.00000001, epsilon_decay=0.99)


    for episode in range(num_episodes+1):
        terminated = False
        _ = env.reset()
        state = env.get_state(view_type)

        if len(state.shape) > 1: 
            state = state.flatten()


        if episode % 50 == 0:
            avg_score, avg_moves = score_maker.get_average_score_moves_training()
            print(f'Average Score {episode} episode over last 100 episodes, score: {avg_score} steps: {avg_moves} epsilon: {score_maker.get_epsilon()}')
        
        move = 0
        while True:

            if terminated:
                break

            action_i = agent.get_action(state, score_maker.get_epsilon())
            action = np.zeros(3)
            action[action_i] = 1

            terminated, score, reward, next_state = env.play_step(action, view_type, episode)
            if len(next_state.shape) > 1: 
                next_state = next_state.flatten()


            agent.update_memory(state, action_i, reward, next_state, terminated)
            state = next_state

            move += 1
            
            if move % 5 == 0:
                agent.train()


        score_maker.add_score_moves_training(score, move)
        
        agent.train()
        if episode % 10 == 0:
            agent.update_target_model()

        score_maker.epsilone_decay_step()

    return score_maker.get_training_statistic()


def test_model(agent, iterations, env:SnakeGame, view_type, name_agent,episodes):
    total_moves = 0
    total_score = 0

    for iter in range(iterations):
        print(iter)
        moves = 0
        _ = env.reset()
        state = env.get_state(view_type)
        if len(state.shape) > 1: 
            state = state.flatten()

        terminated = False
        while terminated is False:
            action_i = agent.get_action(state, epsilon=0)
            action = np.zeros(3)
            action[action_i] = 1

            terminated, score, reward, next_state = env.play_step(action, view_type, iter)
            if len(next_state.shape) > 1: 
                next_state = next_state.flatten()
            agent.update_memory(state, action_i, reward, next_state, terminated)
            state = next_state
            moves += 1
            

        total_moves += moves
        total_score += score

    avg_moves = total_moves / iterations
    avg_score = total_score / iterations
    

        
    print(f'Test Results over {iterations} iterations: Average Moves: {avg_moves}, Average Score: {avg_score}')

    print("Save agent? (y/n)")
    choice = input().lower()
    if choice == 'y':
        agent.save(f'{name_agent}_{view_type}_type_{avg_score}_episodes_{episodes}',f'[bat=,mem=]')
    


def build_model(input_dim, output_dim):
    
        
    
    model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        
    
    return model