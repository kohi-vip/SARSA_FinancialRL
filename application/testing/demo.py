"""
Demo script showing how to use the organized agents and environments
"""
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import agents
from agents.sarsa.deep_sarsa_agent import DeepSARSAAgent
from agents.dqn_agent import DQNAgent

# Import environment
from environments.env_stocktrading import StockTradingEnv

# Example usage
def demo():
    print("=== SARSA Financial RL Demo ===")
    print("\nAvailable Agents:")
    print("1. DeepSARSAAgent (FinRL integrated)")
    print("2. DQNAgent (Yang et al. 2020)")
    print("3. DeepSARSAAgent Paper (Yang et al. 2020)")

    print("\nAvailable Environment:")
    print("- StockTradingEnv (FinRL)")

    print("\nTraining Notebooks:")
    print("- training/new_train.ipynb")
    print("- training/agent_training_fpt.ipynb")

    print("\nTo use an agent:")
    print("""
from agents.sarsa.deep_sarsa_agent import DeepSARSAAgent
from environments.env_stocktrading import StockTradingEnv

# Initialize environment
env = StockTradingEnv(...)

# Initialize agent
agent = DeepSARSAAgent(env, state_dim=10, action_dim=5)

# Train
agent.train(num_episodes=100)
""")

if __name__ == "__main__":
    demo()