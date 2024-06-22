from utils import set_seed
from execution import Execution
from utils.evaluator import Evaluator
import torch
from tqdm import trange, tqdm
from parameters import EPOCHS, DEVICE, DTYPE, TEMPERATURE, NUM_EVAL_STEPS, OUTPUT_DIRECTORY_PATH
from transformers.tokenization_utils import BatchEncoding
from gauss_model import GaussOutput
from utils.similarity import asymmetrical_kl_sim_mat
import torch.nn.functional as F
from utils.save import save_json

def main():
    set_seed()
    execution = Execution()

    evaluator = Evaluator(sim_fn=execution.sim_fn)

    best_dev_score = evaluator.dev()
    best_epoch, best_step = 0, 0
    val_metrics = {
        "epoch": best_epoch,
        "step": best_step,
        "loss": float("inf"),
        "dev_score": best_dev_score,
    }
    execution.log(val_metrics)
    best_state_dict = execution.clone_state_dict()

    scaler = torch.cuda.amp.GradScaler()
    current_step = 0

    for epoch in trange(EPOCHS, leave=False, dynamic_ncols=True, desc="Epoch"):
        train_losses = []
        execution.model.train()

        for batch in tqdm(execution.train_dataloader, total=len(execution.train_dataloader), dynamic_ncols=True, leave=False, desc="Step"):
            current_step += 1
            batch: BatchEncoding = batch.to(DEVICE)

            with torch.cuda.amp.autocast(dtype=DTYPE):
                sent0_out: GaussOutput = execution.model.forward(**batch.sent0)
                sent1_out: GaussOutput = execution.model.forward(**batch.sent1)

            sim_mat: torch.FloatTensor = asymmetrical_kl_sim_mat(sent0_out.mu, sent0_out.std, sent1_out.mu, sent1_out.std)
            sim_mat = sim_mat / TEMPERATURE

            batch_size = sim_mat.size(0)
            labels = torch.arange(batch_size).to(DEVICE, non_blocking=True)
            loss = F.cross_entropy(sim_mat, labels)

            train_losses.append(loss.item())

            execution.optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(execution.optimizer)

            scale = scaler.get_scale()
            scaler.update()

            if scale <= scaler.get_scale():
                execution.lr_scheduler.step()
            
            if current_step % NUM_EVAL_STEPS == 0:
                execution.model.eval()

                dev_score = evaluator.dev()

                if best_dev_score < dev_score:
                    best_dev_score = dev_score
                    best_epoch, best_step = epoch, current_step
                    best_state_dict = execution.clone_state_dict()

                val_metrics = {
                    "epoch": epoch,
                    "step": current_step,
                    "loss": sum(train_losses) / len(train_losses),
                    "dev-auc": dev_score,
                }
                execution.log(val_metrics)
                train_losses = []

                execution.model.train()

    dev_metrics = {
        "best-epoch": best_epoch,
        "best-step": best_step,
        "best-dev-auc": best_dev_score,
    }
    save_json(dev_metrics, OUTPUT_DIRECTORY_PATH / "dev-metrics.json")

    execution.model.load_state_dict(best_state_dict)
    execution.model.eval().to(DEVICE)

    metrics = execution.evaluator.eval()
    save_json(metrics, OUTPUT_DIRECTORY_PATH / "metrics.json")

if __name__ == "__main__":
    main()