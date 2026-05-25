"""Training callbacks: checkpointing, LR scheduling, optional WandB logging."""
import os
from pathlib import Path


class CheckpointCallback:
    def __init__(self, save_dir: str, save_interval: int = 10):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_interval = save_interval

    def on_epoch_end(self, model, epoch: int, metrics: dict):
        if epoch % self.save_interval == 0 or epoch == 1:
            path = self.save_dir / f"epoch_{epoch:04d}.pth"
            model.save(str(path), epoch)
            print(f"[Checkpoint] Saved → {path}")


class LRSchedulerCallback:
    """Linear LR decay starting at `decay_start_epoch`."""

    def __init__(self, model, total_epochs: int, decay_start_epoch: int):
        self.model = model
        self.total_epochs = total_epochs
        self.decay_start_epoch = decay_start_epoch
        self._schedulers = []

    def _build_schedulers(self):
        from torch.optim.lr_scheduler import LambdaLR

        def rule(epoch):
            if epoch < self.decay_start_epoch:
                return 1.0
            return max(0.0, 1.0 - (epoch - self.decay_start_epoch) /
                       (self.total_epochs - self.decay_start_epoch))

        opts = []
        if hasattr(self.model, "opt_G") and self.model.opt_G:
            opts.append(self.model.opt_G)
        if hasattr(self.model, "opt_D") and self.model.opt_D:
            opts.append(self.model.opt_D)
        # last_epoch=-1 avoids the "step() before optimizer.step()" warning
        self._schedulers = [LambdaLR(opt, rule, last_epoch=-1) for opt in opts]

    def on_epoch_end(self, model, epoch: int, metrics: dict):
        if not self._schedulers:
            self._build_schedulers()
        for sched in self._schedulers:
            sched.step()


class WandBCallback:
    def __init__(self, project: str, entity: str | None = None, enabled: bool = False):
        self.enabled = enabled
        if enabled:
            import wandb
            wandb.init(project=project, entity=entity)

    def on_step(self, metrics: dict, step: int):
        if self.enabled:
            import wandb
            wandb.log(metrics, step=step)

    def on_epoch_end(self, model, epoch: int, metrics: dict):
        if self.enabled:
            import wandb
            wandb.log({"epoch": epoch, **metrics})
