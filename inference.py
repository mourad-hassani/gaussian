from gauss_model import GaussModel
from parameters import MODEL_NAME, DEVICE
import torch
from utils.evaluator import Evaluator
from execution import Execution

model: GaussModel = GaussModel(MODEL_NAME, True).eval().to(DEVICE)
model.load_state_dict(torch.load('temporal_bert.pth', map_location=torch.device(DEVICE)))
execution = Execution()

evaluator = Evaluator(sim_fn=execution.sim_fn)
output = evaluator.test()
for o in output:
    print(f"{o.sent0}, {o.sent1}, {o.similarity}")