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
for sent0, sent1, score in output["sent0"], output["sent1"], output["score"]:
    print(f"{sent0}, {sent1}, {score}")