import torch

path = "data/traces/train/eval_trajectories/step_571520_split_1_episode_0000.pt"
traj = torch.load(path, map_location="cpu", weights_only=False)

for key in traj.keys():
    val = traj[key]
    if hasattr(val, "batch_size"):  # 嵌套 TensorDict
        print(f"{key}: TensorDict, keys={list(val.keys())}, batch={val.batch_size}")
    elif torch.is_tensor(val):
        print(f"{key}: {tuple(val.shape)} {val.dtype}")
    else:
        print(f"{key}: {type(val)}")