from run.merina_trajectory_tensordict_store import load_merina_trajectory_pt

path = "artifacts/traces/merina_trajectories/fcc-test_video1/0000_fcc-test__fcc-test_000_trace_942154_http---www.ebay.com_0.pt"  # 改成实际文件名

td, meta = load_merina_trajectory_pt(path)
print("=== meta ===")
for k, v in meta.items():
    print(f"  {k}: {v}")
print("\n=== trajectory 字段 ===")
for k in td.keys():
    t = td[k]
    print(f"  {k:16s} shape={tuple(t.shape)} dtype={t.dtype}")
print("\n=== 前 5 步 ===")
print("buffer_size:", td["buffer_size"][:5].tolist())
print("action:     ", td["action"][:5].tolist())
print("reward:     ", td["reward"][:5].tolist())
print("belief_mu[0,:4]:", td["belief_mu"][0, :4].tolist())