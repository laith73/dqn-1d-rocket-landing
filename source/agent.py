import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class DQNAgent:
    """
    Deep Q-Network Agent
    """
    def __init__(self, state_dim=2, action_dim=21, hidden_dim=128,  learning_rate=0.001,
                 discount_factor=0.99, epsilon_start=1.0, epsilon_end=0.01,
                 epsilon_decay=0.995, batch_size=128, buffer_size=10000,target_update_freq=200):

        self.gamma = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.action_dim = action_dim

        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),  
            nn.Linear(hidden_dim, action_dim)
        )

        self.target_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim) 
        )

        self.device = torch.device("cpu")
        print(self.device)

        self.target_net.to(self.device)
        self.policy_net.to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)

        # Step counter
        self.step_count = 0

        # Set during training
        self.a_low = None
        self.a_high = None

    def _get_action_value(self, action_idx):
        """Convert action index to continuous action."""
        normalized = action_idx / (self.action_dim - 1)
        return self.a_low + normalized * (self.a_high - self.a_low)
    
    def get_action(self, state, training=True):
        if training and np.random.random() < self.epsilon:
            action_indx = np.random.randint(self.action_dim)
            return action_indx, self._get_action_value(action_indx)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.policy_net(state_t)
            action_indx = torch.argmax(q_values).item()

        return action_indx, self._get_action_value(action_indx)

    def update(self):
        
        if len(self.replay_buffer) < self.batch_size * 3:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Current Q-values
        q_values = self.policy_net(states).gather(1, actions)

        # Target Q-values
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target = rewards + self.gamma * next_q * (1 - dones)

        loss = F.mse_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update target network
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def train(self, env, num_episodes=500):
        self.a_low = env.A_low_limits
        self.a_high = env.A_upper_limits
    
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            total_reward = 0

            while not done:
                action_indx, action = self.get_action(state, training=True)
                next_state, reward, done, _ = env.step(action)
                total_reward += reward

                self.replay_buffer.append((state, action_indx, reward, next_state, done))
                self.update()

                state = next_state

            self.decay_epsilon()


class Agent:
    """
    RL Agent Template.

    Students should implement:
    - learn(): training logic that interacts with environment
    - get_action(): policy that returns action given state
    """

    def __init__(self):
        self.q_agent = DQNAgent()
        self.num_episodes = 250


    def learn(self, env):
        """
        Train the agent by interacting with the environment.

        Args:
            env: Environment instance with reset() and step() methods

        *** STUDENTS IMPLEMENT THIS ***
        """
        self.q_agent.train(env, self.num_episodes)

    def get_action(self, state):
        """
        Return action for given state using learned policy.

        Args:
            state: Current state (numpy array)

        Returns:
            action: Action in [A_low_limits, A_upper_limits]

        *** STUDENTS IMPLEMENT THIS ***
        """
        _,action = self.q_agent.get_action(state, training=False)
        return action
