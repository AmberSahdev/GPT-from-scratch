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
from gymnasium.wrappers import FrameStackObservation

ENV_ID = "LunarLander-v3"
CONTINUOUS_ACTIONS = False
ENABLE_WIND = False
DEMO_SEED = 0
DEMO_RENDER_MODE = "human"


# Create a fresh simulator instance. None skips rendering and runs faster.  "human" == live window.
def make_environment(render_mode=None):
    return gym.make(
        ENV_ID,
        continuous=CONTINUOUS_ACTIONS,
        enable_wind=ENABLE_WIND,  # Disable wind to keep the first learning task simpler.
        render_mode=render_mode,
    )


def make_stacked_environment(context_length, render_mode=None):
    """Turn each 8-value state into a history of shape (context_length, 8)."""
    return FrameStackObservation(
        make_environment(render_mode), stack_size=context_length, padding_type="reset"
    )


def run_episode(env, choose, seed, on_step=None):
    """Run one flight; optionally observe the initial state and each completed step.

    on_step(env, steps, done) can record frames. The caller owns and closes env.
    The environment wrapper maintains the observation history for the policy.
    """
    history, _ = env.reset(seed=seed)
    score, steps = 0.0, 0
    if on_step is not None:
        on_step(env, steps, False)

    while True:
        history, reward, terminated, truncated, _ = env.step(choose(history))
        score += reward
        steps += 1
        done = terminated or truncated
        if on_step is not None:
            on_step(env, steps, done)
        if done:
            break

    return {
        "seed": seed,
        "reward": score,
        "steps": steps,
    }


def main():
    # env is the simulator object we reset, step, and eventually close.
    env = make_environment(render_mode=DEMO_RENDER_MODE)

    # reset starts a new episode and returns (initial observation, extra info)
    # observation is an 8-number snapshot: [x, y, vx, vy, angle, angular_velocity, left_leg_contact, right_leg_contact]
    observation, _ = env.reset(seed=DEMO_SEED)

    # This is a performance score, not a training loss or a count of successful landings.
    # LunarLander rewards progress toward landing and penalizes fuel use/crashes.
    total_reward = 0.0

    print("Initial observation:", observation)

    try:
        # Keep advancing this episode until an ending flag tells us to stop.
        while True:
            action = heuristic(env, observation) # heuristic is Gymnasium's handwritten controller, not a learned network

            # Apply the action, advance the physics by one simulator step.
            # observation: new state after that action
            # reward: this step's score, which can be positive or negative
            # terminated: an environment ending, e.g. crash, leaving bounds, or coming to rest
            # truncated: an external cutoff, usually the maximum episode length
            # info: extra diagnostics; Gymnasium always returns all five values
            observation, reward, terminated, truncated, _ = env.step(action)

            total_reward += reward

            if terminated or truncated:
                break

        print("Episode reward:", total_reward)
    finally:
        env.close()


if __name__ == "__main__":
    main()
