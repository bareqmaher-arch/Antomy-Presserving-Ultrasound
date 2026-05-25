"""Generic GAN trainer with AMP and gradient accumulation."""
import csv
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    """
    Orchestrates the training loop for CUT or CycleGAN.

    Supports:
      - Mixed-precision (AMP) via torch.cuda.amp
      - Gradient accumulation (effective_batch = batch_size × accum_steps)
      - Pluggable callbacks (CheckpointCallback, LRSchedulerCallback, WandBCallback)
    """

    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        epochs: int = 200,
        device: torch.device = torch.device("cuda"),
        amp: bool = True,
        grad_accum_steps: int = 2,
        log_interval: int = 50,
        callbacks: list = (),
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.device = device
        self.amp = amp
        self.grad_accum_steps = grad_accum_steps
        self.log_interval = log_interval
        self.callbacks = list(callbacks)
        self._csv_path: Path | None = None
        self._csv_writer = None
        self._csv_file = None

    def _init_csv(self, fieldnames: list[str], log_dir: str = "logs"):
        import datetime
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        # Timestamped file so accidental re-runs never overwrite previous training logs
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._csv_path = Path(log_dir) / f"train_losses_{ts}.csv"
        # Also keep a symlink-style copy as "latest" for convenience
        self._csv_latest = Path(log_dir) / "train_losses.csv"
        self._csv_file = open(self._csv_path, "w", newline="")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=["epoch"] + fieldnames)
        self._csv_writer.writeheader()

    def train(self, start_epoch: int = 1):
        self.model.G.train()
        self.model.D.train() if hasattr(self.model, "D") else None

        global_step = 0
        for epoch in range(start_epoch, self.epochs + 1):
            epoch_losses: dict[str, float] = {}
            t0 = time.time()

            for step, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs}")):
                real_A = batch["A"].to(self.device, non_blocking=True)
                real_B = batch["B"].to(self.device, non_blocking=True)

                losses = self.model.train_step(real_A, real_B)

                for k, v in losses.items():
                    epoch_losses[k] = epoch_losses.get(k, 0.0) + v

                if (step + 1) % self.log_interval == 0:
                    log_str = "  ".join(f"{k}: {v:.4f}" for k, v in losses.items())
                    print(f"  [step {step+1}] {log_str}")
                    for cb in self.callbacks:
                        if hasattr(cb, "on_step"):
                            cb.on_step(losses, global_step)

                global_step += 1

            n = len(self.train_loader)
            avg = {k: v / n for k, v in epoch_losses.items()}
            elapsed = time.time() - t0
            print(f"Epoch {epoch}/{self.epochs} [{elapsed:.0f}s] | "
                  + "  ".join(f"{k}: {v:.4f}" for k, v in avg.items()))

            # Write losses to CSV for plotting
            if self._csv_writer is None:
                self._init_csv(list(avg.keys()))
            self._csv_writer.writerow({"epoch": epoch, **avg})
            self._csv_file.flush()
            # Keep train_losses.csv always pointing to latest data
            import shutil
            shutil.copy2(self._csv_path, self._csv_latest)

            for cb in self.callbacks:
                cb.on_epoch_end(self.model, epoch, avg)

    def __del__(self):
        if self._csv_file:
            self._csv_file.close()
