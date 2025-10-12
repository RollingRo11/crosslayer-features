import argparse
import sys
import torch
from pathlib import Path
from crosscoder.newcrosscoder import cc_config, Crosscoder_Model
import crosscoder.newcrosscoder as newcrosscoder

sys.modules["newcrosscoder"] = newcrosscoder


model = Crosscoder_Model(cc_config)
state_dict = torch.load(
    "./checkpoints/run_8/crosscoder_step_20000.pt",
    map_location="cpu",
    weights_only=False,
)
model.load_state_dict(state_dict["model_state_dict"])
model.eval()

print(model.W_dec.shape)

print(model.W_dec[10000][0].norm())

norms = []
for i in range(1):
    for j in range(12):
        inorm = torch.linalg.norm(model.W_dec[i][j]).item()
        norms.append(inorm)
print(norms)

max_val = max(norms)
print(max_val)
for i in range(len(norms)):
    temp = norms[i]
    norms[i] = temp / max_val

print(norms)
