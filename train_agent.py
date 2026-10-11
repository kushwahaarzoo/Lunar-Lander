import gymnasium as gym
import numpy as np
import argparse
import os

# Lunar Lander environment parameters
OBSERVATION_SIZE = 8
ACTION_SIZE = 4
PARAM_SIZE = OBSERVATION_SIZE * ACTION_SIZE + ACTION_SIZE  # 8x4 weights + 4 biases


def policy_action(params, observation):
    """Compute action using a linear policy."""
    W = params[:OBSERVATION_SIZE * ACTION_SIZE].reshape(OBSERVATION_SIZE, ACTION_SIZE)
    b = params[OBSERVATION_SIZE * ACTION_SIZE:].reshape(ACTION_SIZE)
    logits = np.dot(observation, W) + b
    return np.argmax(logits)


def evaluate_policy(params, episodes=5, render=False):
    """Evaluate a policy over multiple episodes."""
    total_reward = 0.0
    for _ in range(episodes):
        env = gym.make("LunarLander-v3", render_mode="human" if render else "rgb_array")
        observation, _ = env.reset()
        episode_reward = 0.0
        done = False
        while not done:
            action = policy_action(params, observation)
            observation, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        env.close()
        total_reward += episode_reward
    return total_reward / episodes


def pso_optimize(population_size=200, generations=300, w=0.9, c1=2.0, c2=2.0):
    """Particle Swarm Optimization to find good policy parameters."""
    swarm = np.random.randn(population_size, PARAM_SIZE) * 0.1
    velocity = np.random.randn(population_size, PARAM_SIZE) * 0.1
    personal_best = swarm.copy()
    personal_best_scores = np.array([evaluate_policy(p, episodes=5) for p in swarm])
    global_best = personal_best[np.argmax(personal_best_scores)]
    global_best_score = max(personal_best_scores)

    for gen in range(generations):
        for i in range(population_size):
            r1, r2 = np.random.rand(PARAM_SIZE), np.random.rand(PARAM_SIZE)
            velocity[i] = (
                w * velocity[i]
                + c1 * r1 * (personal_best[i] - swarm[i])
                + c2 * r2 * (global_best - swarm[i])
            )
            swarm[i] += velocity[i]
            score = evaluate_policy(swarm[i], episodes=5)

            if score > personal_best_scores[i]:
                personal_best[i] = swarm[i]
                personal_best_scores[i] = score
            if score > global_best_score:
                global_best = swarm[i]
                global_best_score = score

        print(f"Generation {gen+1}: Best Reward = {global_best_score:.2f}")

    return global_best


def tabu_search(best_params, max_iterations=50, tabu_size=50):
    """Refine the best solution found by PSO using Tabu Search."""
    tabu_list = []
    best_solution = best_params.copy()
    best_score = evaluate_policy(best_solution, episodes=10)

    for _ in range(max_iterations):
        candidates = [
            best_solution + np.random.normal(0, 0.05, size=PARAM_SIZE) for _ in range(20)
        ]
        candidate_scores = [evaluate_policy(c, episodes=5) for c in candidates]

        sorted_candidates = sorted(zip(candidates, candidate_scores), key=lambda x: -x[1])

        for candidate, score in sorted_candidates:
            if tuple(candidate) not in tabu_list:
                best_solution = candidate
                best_score = score
                tabu_list.append(tuple(candidate))
                if len(tabu_list) > tabu_size:
                    tabu_list.pop(0)
                break

    return best_solution


def train_and_save(filename):
    """Train the policy using PSO + Tabu Search and save it."""
    best_pso_params = pso_optimize(population_size=200, generations=300)
    refined_params = tabu_search(best_pso_params, max_iterations=50, tabu_size=50)
    np.save(filename, refined_params)
    print(f"Best policy saved to {filename}")
    return refined_params


def load_policy(filename):
    """Load a trained policy."""
    if not os.path.exists(filename):
        print(f"File {filename} does not exist.")
        return None
    return np.load(filename)


def play_policy(best_params, episodes=10):
    """Load and evaluate a trained policy."""
    test_reward = evaluate_policy(best_params, episodes=episodes, render=True)
    print(f"Average reward of best policy: {test_reward:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or play best policy for Lunar Lander using Ultimate Hybrid PSO + Tabu Search.")
    parser.add_argument("--train", action="store_true", help="Train the policy using PSO + Tabu and save it.")
    parser.add_argument("--play", action="store_true", help="Load the best policy and play.")
    parser.add_argument("--filename", type=str, default="bestpolicy1.npy", help="Filename to save/load the best policy.")
    args = parser.parse_args()

    if args.train:
        best_params = train_and_save(args.filename)
    elif args.play:
        best_params = load_policy(args.filename)
        if best_params is not None:
            play_policy(best_params, episodes=10)
    else:
        print("Please specify --train to train and save a policy, or --play to load and play the best policy.")