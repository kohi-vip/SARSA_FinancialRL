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


def calculate_volatility(portfolio_history):
    """
    Calculate volatility (standard deviation of returns)
    
    Args:
        portfolio_history: List of portfolio values over time
        
    Returns:
        Volatility as a percentage
    """
    if len(portfolio_history) < 2:
        return 0.0
    
    # Calculate daily returns
    returns = np.diff(portfolio_history) / portfolio_history[:-1]
    
    # Calculate volatility (annualized)
    volatility = np.std(returns) * np.sqrt(252)  # 252 trading days per year
    
    return volatility * 100  # Return as percentage


def calculate_sharpe_ratio(portfolio_history, risk_free_rate=0.02):
    """
    Calculate Sharpe Ratio using excess returns approach with outlier handling
    
    Args:
        portfolio_history: List of portfolio values over time
        risk_free_rate: Annual risk-free rate (default 2%)
        
    Returns:
        Sharpe Ratio
        
    Note: Uses robust outlier filtering to handle extreme returns that can
    occur when portfolio value approaches zero or has large swings.
    """
    if len(portfolio_history) < 2:
        return 0.0
    
    portfolio_array = np.array(portfolio_history)
    
    # Filter out portfolios that went negative or zero (invalid states)
    if np.any(portfolio_array <= 0):
        # If portfolio hit zero/negative, return very negative Sharpe
        return -10.0
    
    # Calculate daily returns (percentage change)
    returns = np.diff(portfolio_array) / portfolio_array[:-1]
    
    # Remove any inf or nan values
    returns = returns[np.isfinite(returns)]
    
    if len(returns) < 10:  # Need minimum data points
        return 0.0
    
    # Robust outlier filtering using IQR method
    # This prevents extreme outliers from dominating the calculation
    q1 = np.percentile(returns, 25)
    q3 = np.percentile(returns, 75)
    iqr = q3 - q1
    
    # More lenient bounds: 3 * IQR (captures ~99.7% of normal data)
    lower_bound = q1 - 3 * iqr
    upper_bound = q3 + 3 * iqr
    
    # Filter returns, but keep at least 80% of data
    filtered_returns = returns[(returns >= lower_bound) & (returns <= upper_bound)]
    
    # If we filtered out too much, use winsorization instead
    if len(filtered_returns) < 0.8 * len(returns):
        # Winsorize: cap extreme values at 5th and 95th percentiles
        p5 = np.percentile(returns, 5)
        p95 = np.percentile(returns, 95)
        filtered_returns = np.clip(returns, p5, p95)
    
    if len(filtered_returns) == 0:
        return 0.0
    
    # Calculate daily risk-free rate (compound)
    daily_risk_free = (1 + risk_free_rate) ** (1/252) - 1
    
    # Calculate excess returns
    excess_returns = filtered_returns - daily_risk_free
    
    # Calculate mean and std of excess returns
    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns, ddof=1)  # Sample std deviation
    
    if std_excess == 0 or np.isnan(std_excess):
        return 0.0
    
    # Annualize and calculate Sharpe Ratio
    # Formula: mean * sqrt(T) / std
    sharpe_ratio = mean_excess * np.sqrt(252) / std_excess
    
    return sharpe_ratio


def calculate_max_drawdown(portfolio_history):
    """
    Calculate Maximum Drawdown
    
    Args:
        portfolio_history: List of portfolio values over time
        
    Returns:
        Maximum drawdown as a percentage
    """
    if len(portfolio_history) < 2:
        return 0.0
    
    portfolio_array = np.array(portfolio_history)
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(portfolio_array)
    
    # Calculate drawdown at each point
    drawdown = (portfolio_array - running_max) / running_max
    
    # Maximum drawdown
    max_drawdown = np.min(drawdown)
    
    return max_drawdown * 100  # Return as percentage


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
    all_portfolio_histories = []
    
    for run in tqdm(range(num_runs), desc=f"Running Deep SARSA {num_runs} times"):
        # Train the agent
        pi_deep, qsa, learning_curve = train_deep_sarsa(mdp, train_series, test_series, verbose=False, episodes=episodes, gamma=gamma, alpha=alpha, epsilon_start=epsilon_start, epsilon_min=epsilon_min, epsilon_decay=epsilon_decay, nn_epochs=nn_epochs, nn_lr=nn_lr)
        
        # Evaluate on test set
        final_profit = mdp.interact_test(pi_deep, train_series=train_series, test_series=test_series, series_name='test', verbose=False)
        
        # Get portfolio history
        prev_row = train_series.iloc[-1]
        state_init = [
            float(prev_row['close']), 
            mdp.balance_init, 
            0,
            float(prev_row['MACD']),
            float(prev_row['RSI']),
            float(prev_row['CCI']),
            float(prev_row['ADX']),
            float(prev_row['VWAP']),
            float(prev_row['CORR_FX'])
        ]
        states, _, _ = mdp.simulate(test_series, state_init, pi_deep, True)
        portfolio_history = [s[1] + s[0] * s[2] for s in states]
        
        all_final_profits.append(final_profit)
        all_learning_curves.append(learning_curve)
        all_portfolio_histories.append(portfolio_history)
    
    # Calculate statistics
    final_profit = np.mean(all_final_profits)
    learning_curve = np.mean(all_learning_curves, axis=0)
    std_final_profit = np.std(all_final_profits)
    std_learning_curve = np.std(all_learning_curves, axis=0)
    portfolio_history = np.mean(all_portfolio_histories, axis=0)
    
    # Calculate ROI
    initial_investment = mdp.balance_init
    final_portfolio = initial_investment + final_profit
    roi = ((final_portfolio - initial_investment) / initial_investment) * 100
    
    # Calculate financial metrics across all runs
    all_volatilities = [calculate_volatility(ph) for ph in all_portfolio_histories]
    all_sharpe_ratios = [calculate_sharpe_ratio(ph) for ph in all_portfolio_histories]
    all_max_drawdowns = [calculate_max_drawdown(ph) for ph in all_portfolio_histories]
    
    volatility = np.mean(all_volatilities)
    sharpe_ratio = np.mean(all_sharpe_ratios)
    max_drawdown = np.mean(all_max_drawdowns)
    
    return {
        'agent': 'Deep SARSA',
        'final_profit': final_profit,
        'std_final_profit': std_final_profit,
        'learning_curve': learning_curve,
        'std_learning_curve': std_learning_curve,
        'all_final_profits': all_final_profits,
        'all_learning_curves': all_learning_curves,
        'portfolio_history': portfolio_history,
        'all_portfolio_histories': all_portfolio_histories,
        'roi': roi,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'trained_agent': qsa  # Add trained Q-network
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
    all_portfolio_histories = []
    
    state_dim = 9  # [price, balance, shares, MACD, RSI, CCI, ADX, VWAP, CORR_FX]
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
            target_update=10,
            lr_update=0.7
        )
        
        # Training loop
        learning_curve = []
        eps = epsilon_start
        
        for episode in range(episodes):
            # Decay epsilon
            eps *= epsilon_decay
            current_eps = max(epsilon_min, eps)
            
            # Reset environment (simulate from training data)
            state = [
                float(train_series.iloc[0]['close']),
                mdp.balance_init,
                0,
                float(train_series.iloc[0]['MACD']),
                float(train_series.iloc[0]['RSI']),
                float(train_series.iloc[0]['CCI']),
                float(train_series.iloc[0]['ADX']),
                float(train_series.iloc[0]['VWAP']),
                float(train_series.iloc[0]['CORR_FX'])
            ]
            
            # Collect trajectory
            states = [state]
            actions_taken = []
            rewards = []
            
            step = 0
            max_steps = len(train_series) - 1
            
            while step < max_steps:
                # Select action with current epsilon
                action_idx = agent.select_action(state, training=True)
                agent.epsilon = current_eps  # Temporarily set epsilon for selection
                action = mdp.A[action_idx]
                
                # Get next state
                next_row = train_series.iloc[step + 1]
                next_state = mdp.update_state(state, action, next_row)
                
                # Calculate reward
                reward = mdp.reward(state, next_state)
                
                # Store
                states.append(next_state)
                actions_taken.append(action_idx)
                rewards.append(reward)
                
                state = next_state
                step += 1
            
            # Add experiences to replay buffer
            for i in range(len(states) - 1):
                done = 1.0 if i == len(states) - 2 else 0.0
                agent.store_transition(states[i], actions_taken[i], rewards[i], states[i+1], done)
            
            # Training step - multiple updates per episode
            if len(agent.replay_buffer) >= 64:
                for _ in range(10):  # Multiple updates per episode
                    agent.train_step()
            
            # Evaluate on test set
            test_profit = evaluate_dqn_on_test(agent, mdp, test_series)
            learning_curve.append(test_profit)
        
        # Final evaluation
        final_profit = evaluate_dqn_on_test(agent, mdp, test_series)
        
        # Get portfolio history
        def pi_dqn(s, greedy=True, eps=0.0):
            action_idx = agent.select_action(s, training=False)
            return mdp.A[action_idx]
        
        prev_row = train_series.iloc[-1]
        state_init = [
            float(prev_row['close']), 
            mdp.balance_init, 
            0,
            float(prev_row['MACD']),
            float(prev_row['RSI']),
            float(prev_row['CCI']),
            float(prev_row['ADX']),
            float(prev_row['VWAP']),
            float(prev_row['CORR_FX'])
        ]
        states, _, _ = mdp.simulate(test_series, state_init, pi_dqn, True)
        portfolio_history = [s[1] + s[0] * s[2] for s in states]
        
        all_final_profits.append(final_profit)
        all_learning_curves.append(learning_curve)
        all_portfolio_histories.append(portfolio_history)
    
    # Calculate statistics
    final_profit = np.mean(all_final_profits)
    learning_curve = np.mean(all_learning_curves, axis=0)
    std_final_profit = np.std(all_final_profits)
    std_learning_curve = np.std(all_learning_curves, axis=0)
    portfolio_history = np.mean(all_portfolio_histories, axis=0)
    
    # Calculate ROI
    initial_investment = mdp.balance_init
    final_portfolio = initial_investment + final_profit
    roi = ((final_portfolio - initial_investment) / initial_investment) * 100
    
    # Calculate financial metrics across all runs
    all_volatilities = [calculate_volatility(ph) for ph in all_portfolio_histories]
    all_sharpe_ratios = [calculate_sharpe_ratio(ph) for ph in all_portfolio_histories]
    all_max_drawdowns = [calculate_max_drawdown(ph) for ph in all_portfolio_histories]
    
    volatility = np.mean(all_volatilities)
    sharpe_ratio = np.mean(all_sharpe_ratios)
    max_drawdown = np.mean(all_max_drawdowns)
    
    return {
        'agent': 'DQN',
        'final_profit': final_profit,
        'std_final_profit': std_final_profit,
        'learning_curve': learning_curve,
        'std_learning_curve': std_learning_curve,
        'all_final_profits': all_final_profits,
        'all_learning_curves': all_learning_curves,
        'portfolio_history': portfolio_history,
        'all_portfolio_histories': all_portfolio_histories,
        'roi': roi,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'trained_agent': agent  # Add trained DQN agent
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
        float(test_series.iloc[0]['ADX']),
        float(test_series.iloc[0]['VWAP']),
        float(test_series.iloc[0]['CORR_FX'])
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
    all_portfolio_histories = []
    
    state_dim = 9  # [price, balance, shares, MACD, RSI, CCI, ADX, VWAP, CORR_FX]
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
                float(train_series.iloc[0]['ADX']),
                float(train_series.iloc[0]['VWAP']),
                float(train_series.iloc[0]['CORR_FX'])
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
        
        # Get portfolio history
        def pi_pg(s, greedy=True, eps=0.0):
            action, _ = agent.select_action(s)
            return mdp.A[action]
        
        prev_row = train_series.iloc[-1]
        state_init = [
            float(prev_row['close']), 
            mdp.balance_init, 
            0,
            float(prev_row['MACD']),
            float(prev_row['RSI']),
            float(prev_row['CCI']),
            float(prev_row['ADX']),
            float(prev_row['VWAP']),
            float(prev_row['CORR_FX'])
        ]
        states, _, _ = mdp.simulate(test_series, state_init, pi_pg, True)
        portfolio_history = [s[1] + s[0] * s[2] for s in states]
        
        all_final_profits.append(final_profit)
        all_learning_curves.append(learning_curve)
        all_portfolio_histories.append(portfolio_history)
    
    # Calculate statistics
    final_profit = np.mean(all_final_profits)
    learning_curve = np.mean(all_learning_curves, axis=0) if all_learning_curves else []
    std_final_profit = np.std(all_final_profits)
    std_learning_curve = np.std(all_learning_curves, axis=0) if all_learning_curves else []
    portfolio_history = np.mean(all_portfolio_histories, axis=0)
    
    # Calculate ROI
    initial_investment = mdp.balance_init
    final_portfolio = initial_investment + final_profit
    roi = ((final_portfolio - initial_investment) / initial_investment) * 100
    
    # Calculate financial metrics across all runs
    all_volatilities = [calculate_volatility(ph) for ph in all_portfolio_histories]
    all_sharpe_ratios = [calculate_sharpe_ratio(ph) for ph in all_portfolio_histories]
    all_max_drawdowns = [calculate_max_drawdown(ph) for ph in all_portfolio_histories]
    
    volatility = np.mean(all_volatilities)
    sharpe_ratio = np.mean(all_sharpe_ratios)
    max_drawdown = np.mean(all_max_drawdowns)
    
    return {
        'agent': 'Policy Gradient',
        'final_profit': final_profit,
        'std_final_profit': std_final_profit,
        'learning_curve': learning_curve,
        'std_learning_curve': std_learning_curve,
        'all_final_profits': all_final_profits,
        'all_learning_curves': all_learning_curves,
        'portfolio_history': portfolio_history,
        'all_portfolio_histories': all_portfolio_histories,
        'roi': roi,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'trained_agent': agent  # Add trained Policy Gradient agent
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
        float(test_series.iloc[0]['ADX']),
        float(test_series.iloc[0]['VWAP']),
        float(test_series.iloc[0]['CORR_FX'])
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
        print(f"  • ROI: {res['roi']:.2f}%")
        # Tính annual return rate (giả sử 8 năm cho period train+test)
        final_portfolio = 1000 + res['final_profit']
        annual_return = ((final_portfolio / 1000) ** (1/8) - 1) * 100
        print(f"  • Annual Return Rate: {annual_return:.2f}%")
        if len(res['learning_curve']) > 0:
            print(f"  • Best Training Profit: ${np.max(res['learning_curve']):.2f}")
            print(f"  • Training Stability (std): ${np.mean(res['std_learning_curve']):.2f}")
    
    return results



