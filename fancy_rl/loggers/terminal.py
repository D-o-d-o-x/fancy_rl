from typing import Dict, Sequence, Union, Optional
from torch import Tensor

from torchrl.record.loggers.common import Logger

class TerminalLogger(Logger):
    """Logger that prints to the terminal."""

    def __init__(self, exp_name: str, log_dir: str) -> None:
        super().__init__(exp_name, log_dir)

    def _create_experiment(self):
        # No need to create any experiment object for terminal logging
        pass

    def log_scalar(self, name: str, value: float, step: Optional[int] = None) -> None:
        """Logs a scalar value to the terminal."""
        if step is not None:
            print(f"Step {step}: {name} - {value}")
        else:
            print(f"{name}: {value}")

    def log_video(self, name: str, video: Tensor, step: Optional[int] = None, **kwargs) -> None:
        """Logs video information to the terminal."""
        if step is not None:
            print(f"Step {step}: Logging video {name}")
        else:
            print(f"Logging video {name}")

    def log_hparams(self, cfg: Union[Dict, Sequence]) -> None:
        """Logs hyperparameters to the terminal."""
        print("Hyperparameters:")
        for key, value in cfg.items():
            print(f"{key}: {value}")

    def __repr__(self) -> str:
        return "TerminalLogger"

    def log_histogram(self, name: str, data: Sequence, **kwargs) -> None:
        """Logs histogram data to the terminal."""
        print(f"Logging histogram {name}")