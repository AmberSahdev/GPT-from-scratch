"""
Using LunarLander as our environment because
1) it's pretty
2) has more states to learn than other available pretty simulation environments like blackjack

LunarLander has
- 8 observation values
    - [x, y, horizontal_velocity, vertical_velocity, angle, angular_velocity, left_leg_contact, right_leg_contact]
- 4 actions
    - 0: nothing
    - 1: left engine
    - 2: main engine
    - 3: right engine
"""

import gymnasium as gym
from gymnasium.envs.box2d.lunar_lander import heuristic

# Shared simulator settings; other modules reuse make_environment().
ENV_ID = "LunarLander-v3"
CONTINUOUS_ACTIONS = False
ENABLE_WIND = False
DEMO_SEED = 0
DEMO_RENDER_MODE = "human"


# Create a fresh simulator instance. None skips rendering and runs faster. Use "human" for a live window.
def make_environment(render_mode=None):
    return gym.make(
        ENV_ID,
        continuous=CONTINUOUS_ACTIONS,
        enable_wind=ENABLE_WIND,  # Disable wind to keep the first learning task simpler.
        render_mode=render_mode,
    )


def main():
    # env is the simulator object we reset, step, and eventually close.
    env = make_environment(render_mode=DEMO_RENDER_MODE)

    # reset starts a new episode and returns (initial observation, extra info)
    # observation is an 8-number snapshot: [x, y, vx, vy, angle, angular_velocity, left_leg_contact, right_leg_contact]
    observation, info = env.reset(seed=DEMO_SEED)
    env.action_space.seed(DEMO_SEED)

    # This is a performance score, not a training loss or a count of successful landings.
    # LunarLander rewards progress toward landing and penalizes fuel use/crashes.
    total_reward = 0.0

    print("Initial observation:", observation)

    try:
        # Keep advancing this episode until an ending flag tells us to stop.
        while True:
            # action = env.action_space.sample() # random command
            action = heuristic(env, observation) # heuristic is Gymnasium's handwritten controller, not a learned network

            # Apply the action, advance the physics by one simulator step.
            # observation: new state after that action
            # reward: this step's score, which can be positive or negative
            # terminated: an environment ending, e.g. crash, leaving bounds, or coming to rest
            # truncated: an external cutoff, usually the maximum episode length
            # info: extra diagnostics; Gymnasium always returns all five values
            observation, reward, terminated, truncated, info = env.step(action)

            total_reward += reward

            if terminated or truncated:
                break

        print("Episode reward:", total_reward)
    finally:
        env.close()


if __name__ == "__main__":
    main()
