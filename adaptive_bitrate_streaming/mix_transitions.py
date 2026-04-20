import argparse
import os
import pickle
import random
from typing import List


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _chunk_indices(total: int, episode_len: int) -> List[List[int]]:
    if episode_len <= 0:
        raise ValueError("episode_len must be a positive integer.")
    if total % episode_len != 0:
        raise ValueError(
            f"Total transitions ({total}) is not divisible by episode_len ({episode_len})."
        )
    num_episodes = total // episode_len
    return [
        list(range(ep_id * episode_len, (ep_id + 1) * episode_len))
        for ep_id in range(num_episodes)
    ]


def shuffle_episodes(pool, episode_len: int, seed: int):
    states = pool.states
    actions = pool.actions
    rewards = pool.rewards
    dones = pool.dones

    total = len(states)
    if not (len(actions) == len(rewards) == len(dones) == total):
        raise ValueError("Inconsistent pool lengths among states/actions/rewards/dones.")

    episodes = _chunk_indices(total, episode_len)
    rng = random.Random(seed)
    rng.shuffle(episodes)

    shuffled_states = []
    shuffled_actions = []
    shuffled_rewards = []
    shuffled_dones = []

    for ep in episodes:
        for idx in ep:
            shuffled_states.append(states[idx])
            shuffled_actions.append(actions[idx])
            shuffled_rewards.append(rewards[idx])
            shuffled_dones.append(dones[idx])

    pool.states = shuffled_states
    pool.actions = shuffled_actions
    pool.rewards = shuffled_rewards
    pool.dones = shuffled_dones

    return total, len(episodes)


def main():
    parser = argparse.ArgumentParser(
        description="Shuffle experience-pool episodes as whole units."
    )
    parser.add_argument(
        "--input",
        default="artifacts/exp_pools/allfccv1_origin_merge_exp_pool.pkl",
        help="Input exp_pool pickle path.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/exp_pools/exp_pool_shuffled.pkl",
        help="Output shuffled pickle path.",
    )
    parser.add_argument(
        "--episode-len",
        type=int,
        default=47,
        help="Transitions per episode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic shuffling.",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite input file (ignores --output).",
    )

    args = parser.parse_args()

    input_path = (
        args.input
        if os.path.isabs(args.input)
        else os.path.join(SCRIPT_DIR, args.input)
    )
    output_arg = args.input if args.inplace else args.output
    output_path = (
        output_arg
        if os.path.isabs(output_arg)
        else os.path.join(SCRIPT_DIR, output_arg)
    )

    with open(input_path, "rb") as f:
        pool = pickle.load(f)

    total, num_episodes = shuffle_episodes(pool, args.episode_len, args.seed)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(pool, f)

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Transitions: {total}")
    print(f"Episodes: {num_episodes}")
    print(f"Episode length: {args.episode_len}")
    print(f"Seed: {args.seed}")


if __name__ == "__main__":
    main()
