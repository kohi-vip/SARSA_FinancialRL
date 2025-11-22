"""
Run experiments for multiple RL agents in stock trading
Supports Deep SARSA, DQN, and Policy Gradient agents
"""

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from agents.d_sarsa.d_sarsa import train_deep_sarsa
from agents.dqn.dqn_agent import DQNAgent
from agents.policy_gradient.policy_gradient_agent import PolicyGradientAgent, train_policy_gradient_agent, evaluate_policy_gradient_agent


def run_deep_sarsa_experiment(mdp, train_series, test_series, episodes, gamma, alpha, epsilon_start, epsilon_min, epsilon_decay, nn_epochs, nn_lr, num_runs=20):
    """
    Run Deep SARSA experiment multiple times
    
    Args:
        mdp: StockTradingMDP instance
        train_series: Training DataFrame
        test_series: Test DataFrame
        config: Dict with hyperparameters
        num_runs: Number of runs
        
    Returns:
        Dict with results
    """
    all_final_profits = []
    all_learning_curves = []
    
    for run in tqdm(range(num_runs), desc=f"Running Deep SARSA {num_runs} times"):
        # Train the agent
        pi_deep, qsa, learning_curve = train_deep_sarsa(mdp, train_series, test_series, verbose=False, episodes=episodes, gamma=gamma, alpha=alpha, epsilon_start=epsilon_start, epsilon_min=epsilon_min, epsilon_decay=epsilon_decay, nn_epochs=nn_epochs, nn_lr=nn_lr)
        
        # Evaluate on test set
        final_profit = mdp.interact_test(pi_deep, train_series=train_series, test_series=test_series, series_name='test', verbose=False)
        
        all_final_profits.append(final_profit)
        all_learning_curves.append(learning_curve)
    
    # Calculate statistics
    final_profit = np.mean(all_final_profits)
    learning_curve = np.mean(all_learning_curves, axis=0)
    std_final_profit = np.std(all_final_profits)
    std_learning_curve = np.std(all_learning_curves, axis=0)
    
    return {
        'agent': 'Deep SARSA',
        'final_profit': final_profit,
        'std_final_profit': std_final_profit,
        'learning_curve': learning_curve,
        'std_learning_curve': std_learning_curve,
        'all_final_profits': all_final_profits,
        'all_learning_curves': all_learning_curves
    }


def run_dqn_experiment(mdp, train_series, test_series, episodes, gamma, epsilon_start, epsilon_min, epsilon_decay, nn_lr, num_runs=20):
    """
    Run DQN experiment multiple times
    
    Args:
        mdp: StockTradingMDP instance
        train_series: Training DataFrame
        test_series: Test DataFrame
        config: Dict with hyperparameters
        num_runs: Number of runs
        
    Returns:
        Dict with results
    """
    all_final_profits = []
    all_learning_curves = []
    
    state_dim = 7  # [price, balance, shares, MACD, RSI, CCI, ADX]
    n_actions = len(mdp.A)
    
    for run in tqdm(range(num_runs), desc=f"Running DQN {num_runs} times"):
        # Initialize agent
        agent = DQNAgent(
            state_dim=state_dim,
            n_actions=n_actions,
            lr=nn_lr,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_min,
            epsilon_decay=epsilon_decay,
            buffer_capacity=10000,
            batch_size=64,
            target_update=10
        )
        
        # Training loop
        learning_curve = []
        episodes_num = episodes
        
        for episode in range(episodes):
            # Reset environment (simulate from training data)
            state = [
                float(train_series.iloc[0]['close']),
                mdp.balance_init,
                0,
                float(train_series.iloc[0]['MACD']),
                float(train_series.iloc[0]['RSI']),
                float(train_series.iloc[0]['CCI']),
                float(train_series.iloc[0]['ADX'])
            ]
            
            episode_reward = 0
            done = False
            step = 0
            max_steps = len(train_series) - 1
            
            while not done and step < max_steps:
                # Select action
                action_idx = agent.select_action(state)
                action = mdp.A[action_idx]  # Convert to actual action (-k to k)
                
                # Get next state
                next_row = train_series.iloc[step + 1]
                next_state = mdp.update_state(state, action, next_row)
                
                # Calculate reward
                reward = mdp.reward(state, next_state)
                
                # Check if done (end of data)
                done = (step + 1 >= max_steps)
                
                # Store transition
                agent.store_transition(state, action_idx, reward, next_state, done)
                
                # Train
                agent.train_step()
                
                state = next_state
                episode_reward += reward
                step += 1
            
            # Evaluate on test set after each episode
            test_profit = evaluate_dqn_on_test(agent, mdp, test_series)
            learning_curve.append(test_profit)
        
        # Final evaluation
        final_profit = evaluate_dqn_on_test(agent, mdp, test_series)
        all_final_profits.append(final_profit)
        all_learning_curves.append(learning_curve)
    
    # Calculate statistics
    final_profit = np.mean(all_final_profits)
    learning_curve = np.mean(all_learning_curves, axis=0)
    std_final_profit = np.std(all_final_profits)
    std_learning_curve = np.std(all_learning_curves, axis=0)
    
    return {
        'agent': 'DQN',
        'final_profit': final_profit,
        'std_final_profit': std_final_profit,
        'learning_curve': learning_curve,
        'std_learning_curve': std_learning_curve,
        'all_final_profits': all_final_profits,
        'all_learning_curves': all_learning_curves
    }


def evaluate_dqn_on_test(agent, mdp, test_series):
    """Evaluate DQN agent on test series"""
    state = [
        float(test_series.iloc[0]['close']),
        mdp.balance_init,
        0,
        float(test_series.iloc[0]['MACD']),
        float(test_series.iloc[0]['RSI']),
        float(test_series.iloc[0]['CCI']),
        float(test_series.iloc[0]['ADX'])
    ]
    
    total_reward = 0
    for i in range(1, len(test_series)):
        action_idx = agent.select_action(state, training=False)
        action = mdp.A[action_idx]
        
        next_row = test_series.iloc[i]
        next_state = mdp.update_state(state, action, next_row)
        reward = mdp.reward(state, next_state)
        
        total_reward += reward
        state = next_state
    
    return total_reward


def run_policy_gradient_experiment(mdp, train_series, test_series, episodes, gamma, nn_lr, num_runs=20):
    """
    Run Policy Gradient experiment multiple times
    
    Args:
        mdp: StockTradingMDP instance
        train_series: Training DataFrame
        test_series: Test DataFrame
        config: Dict with hyperparameters
        num_runs: Number of runs
        
    Returns:
        Dict with results
    """
    all_final_profits = []
    all_learning_curves = []
    
    state_dim = 7
    action_dim = len(mdp.A)
    
    for run in tqdm(range(num_runs), desc=f"Running Policy Gradient {num_runs} times"):
        # Initialize agent
        agent = PolicyGradientAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=128,
            lr=nn_lr,
            gamma=gamma
        )
        
        # Training loop
        learning_curve = []
        num_episodes = episodes
        max_steps = 1000
        update_freq = 10
        
        # Custom training loop to collect learning curve
        episode_rewards = []
        
        for episode in range(num_episodes):
            # Simulate one episode
            state = [
                float(train_series.iloc[0]['close']),
                mdp.balance_init,
                0,
                float(train_series.iloc[0]['MACD']),
                float(train_series.iloc[0]['RSI']),
                float(train_series.iloc[0]['CCI']),
                float(train_series.iloc[0]['ADX'])
            ]
            
            episode_reward = 0
            done = False
            step = 0
            
            while not done and step < max_steps and step < len(train_series) - 1:
                action, log_prob = agent.select_action(state)
                actual_action = mdp.A[action]  # Convert to actual action
                
                next_row = train_series.iloc[step + 1]
                next_state = mdp.update_state(state, actual_action, next_row)
                reward = mdp.reward(state, next_state)
                
                agent.store_transition(state, action, reward, log_prob)
                
                state = next_state
                episode_reward += reward
                step += 1
                done = (step >= len(train_series) - 1)
            
            # Update policy periodically
            if (episode + 1) % update_freq == 0:
                agent.update_policy()
            
            episode_rewards.append(episode_reward)
            
            # Evaluate on test set every 100 episodes
            if (episode + 1) % 100 == 0:
                test_profit = evaluate_pg_on_test(agent, mdp, test_series)
                learning_curve.append(test_profit)
        
        # Final evaluation
        final_profit = evaluate_pg_on_test(agent, mdp, test_series)
        all_final_profits.append(final_profit)
        all_learning_curves.append(learning_curve)
    
    # Calculate statistics
    final_profit = np.mean(all_final_profits)
    learning_curve = np.mean(all_learning_curves, axis=0) if all_learning_curves else []
    std_final_profit = np.std(all_final_profits)
    std_learning_curve = np.std(all_learning_curves, axis=0) if all_learning_curves else []
    
    return {
        'agent': 'Policy Gradient',
        'final_profit': final_profit,
        'std_final_profit': std_final_profit,
        'learning_curve': learning_curve,
        'std_learning_curve': std_learning_curve,
        'all_final_profits': all_final_profits,
        'all_learning_curves': all_learning_curves
    }


def evaluate_pg_on_test(agent, mdp, test_series):
    """Evaluate Policy Gradient agent on test series"""
    state = [
        float(test_series.iloc[0]['close']),
        mdp.balance_init,
        0,
        float(test_series.iloc[0]['MACD']),
        float(test_series.iloc[0]['RSI']),
        float(test_series.iloc[0]['CCI']),
        float(test_series.iloc[0]['ADX'])
    ]
    
    total_reward = 0
    for i in range(1, len(test_series)):
        action, _ = agent.select_action(state)
        actual_action = mdp.A[action]
        
        next_row = test_series.iloc[i]
        next_state = mdp.update_state(state, actual_action, next_row)
        reward = mdp.reward(state, next_state)
        
        total_reward += reward
        state = next_state
    
    return total_reward


def run_experiments(mdp, train_series, test_series, shared_config, num_runs=20):
    """
    Run experiments for multiple agents with shared hyperparameters
    
    Args:
        mdp: StockTradingMDP instance
        train_series: Training DataFrame
        test_series: Test DataFrame
        shared_config: Dict with shared hyperparameters:
            {
                'episodes': int,
                'gamma': float,
                'alpha': float,  # for SARSA
                'epsilon_start': float,
                'epsilon_min': float,
                'epsilon_decay': float,
                'nn_epochs': int,  # for SARSA
                'nn_lr': float,  # for NN learning rate
                'enabled_agents': list of int, e.g., [1,0,0] for [SARSA, DQN, PG]
            }
        num_runs: Number of runs per agent
        
    Returns:
        Dict with results for each enabled agent
    """
    # Extract shared hyperparameters
    episodes = shared_config['episodes']
    gamma = shared_config['gamma']
    alpha = shared_config['alpha']
    epsilon_start = shared_config['epsilon_start']
    epsilon_min = shared_config['epsilon_min']
    epsilon_decay = shared_config['epsilon_decay']
    nn_epochs = shared_config['nn_epochs']
    nn_lr = shared_config['nn_lr']
    enabled_agents = shared_config['enabled_agents']  # [SARSA, DQN, PG]
    
    results = {}
    
    # Run Deep SARSA if enabled
    if enabled_agents[0]:
        print(f"\n{'='*80}")
        print("RUNNING DEEP SARSA EXPERIMENTS")
        print(f"{'='*80}")
        results['sarsa'] = run_deep_sarsa_experiment(mdp, train_series, test_series, episodes, gamma, alpha, epsilon_start, epsilon_min, epsilon_decay, nn_epochs, nn_lr, num_runs)
    
    # Run DQN if enabled
    if enabled_agents[1]:
        print(f"\n{'='*80}")
        print("RUNNING DQN EXPERIMENTS")
        print(f"{'='*80}")
        results['dqn'] = run_dqn_experiment(mdp, train_series, test_series, episodes, gamma, epsilon_start, epsilon_min, epsilon_decay, nn_lr, num_runs)
    
    # Run Policy Gradient if enabled
    if enabled_agents[2]:
        print(f"\n{'='*80}")
        print("RUNNING POLICY GRADIENT EXPERIMENTS")
        print(f"{'='*80}")
        results['pg'] = run_policy_gradient_experiment(mdp, train_series, test_series, episodes, gamma, nn_lr, num_runs)
    
    # Print summary
    print(f"\n{'='*100}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*100}")
    
    for agent_key, res in results.items():
        print(f"\n{res['agent']}:")
        print(f"  • Average Final Profit: ${res['final_profit']:.2f} ± ${res['std_final_profit']:.2f}")
        if len(res['learning_curve']) > 0:
            print(f"  • Best Training Profit: ${np.max(res['learning_curve']):.2f}")
            print(f"  • Training Stability (std): ${np.mean(res['std_learning_curve']):.2f}")
    
    return results



