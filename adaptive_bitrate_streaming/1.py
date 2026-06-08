import torch
td = torch.load(
    "artifacts/qoe_cdf_columns/video1/fcc-test_mpc.pt",
    map_location="cpu",
    weights_only=False,
)
print(td.keys())                   # dict_keys(['avg_qoe', 'quality', ...])
print(td) 