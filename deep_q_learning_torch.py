import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


class DQN:
    def __init__(self, state_space_shape, num_actions, model, target_model, learning_rate=0.1,
                 discount_factor=0.95, batch_size=16, memory_size=100):
        self.state_space_shape = state_space_shape
        self.num_actions = num_actions
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.model = model
        self.target_model = target_model
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.update_target_model()

    def update_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.num_actions)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return torch.argmax(self.model(state)).item()

    def load(self, model_name, episode):
        self.model.load_state_dict(torch.load(f'dqn_{model_name}_{episode}.pt'))

    def save(self, model_name, episode):
        torch.save(self.model.state_dict(), f'dqn_{model_name}_{episode}.pt')

    def train(self):
        if len(self.memory) < self.batch_size:
            return  # Don't train until we have enough samples

        batch_size = min(self.batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_size)

        states, targets = [], []

        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

            target = self.model(state_tensor).detach().clone().squeeze()
            if done:
                target[action] = reward
            else:
                with torch.no_grad():
                    max_future_q = torch.max(self.target_model(next_state_tensor)).item()
                target[action] = reward + self.discount_factor * max_future_q

            states.append(state)
            targets.append(target)

        states_tensor = torch.tensor(np.array(states), dtype=torch.float32)
        targets_tensor = torch.stack(targets)

        self.optimizer.zero_grad()
        outputs = self.model(states_tensor)
        loss = self.criterion(outputs, targets_tensor)
        loss.backward()
        self.optimizer.step()


class DDQN(DQN):
    def load(self, model_name, episode):
        self.model.load_state_dict(torch.load(f'ddqn_{model_name}_{episode}.pt'))

    def save(self, model_name, episode):
        torch.save(self.model.state_dict(), f'ddqn_{model_name}_{episode}.pt')

    def train(self):
        batch_size = min(self.batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_size)

        states, targets = [], []

        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

            target = self.model(state_tensor).detach().clone().squeeze()
            if done:
                target[action] = reward
            else:
                with torch.no_grad():
                    next_q_values = self.model(next_state_tensor)
                    best_action = torch.argmax(next_q_values)
                    max_q = self.target_model(next_state_tensor)[0][best_action].item()
                target[action] = reward + self.discount_factor * max_q

            states.append(state)
            targets.append(target)

        states_tensor = torch.tensor(np.array(states), dtype=torch.float32)
        targets_tensor = torch.stack(targets)

        self.optimizer.zero_grad()
        outputs = self.model(states_tensor)
        loss = self.criterion(outputs, targets_tensor)
        loss.backward()
        self.optimizer.step()
