import signal
from agent import Agent
from env import Environment

# Training timeout in seconds (3 minutes)
LEARN_TIMEOUT = 3 * 60

class TimeoutError(Exception):
    """Raised when training exceeds time limit."""
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Training exceeded time limit!")


def evaluate(agent, env, num_episodes=100):
    """
    Evaluate agent performance on environment.

    Args:
        agent: Trained Agent instance
        env: Environment instance
        num_episodes: Number of episodes to evaluate

    Returns:
        avg_reward: Average total reward per episode
    """
    total_rewards = []

    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.get_action(state)
            state, reward, done, _ = env.step(action)
            episode_reward += reward

        total_rewards.append(episode_reward)

    return sum(total_rewards) / len(total_rewards)


def train_with_timeout(agent, env, timeout=LEARN_TIMEOUT):
    """
    Train agent with optional timeout.

    Args:
        agent: Agent instance
        env: Environment instance
        timeout: Timeout in seconds (None = no timeout)
    """
    if timeout is not None:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)

    try:
        agent.learn(env)
    except TimeoutError as e:
        print(f"Warning: {e}")
    finally:
        if timeout is not None:
            signal.alarm(0)


def main():
    # Create environment and agent
    env = Environment()
    agent = Agent()

    # Train agent (set timeout=LEARN_TIMEOUT to enable 3 min limit)
    train_with_timeout(agent, env, timeout=None)

    # Evaluate agent
    avg_reward = evaluate(agent, env)
    print(f"Average reward over 100 episodes: {avg_reward:.4f}")


if __name__ == "__main__":
    main()
