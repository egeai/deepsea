"""Command-line interface."""
import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """DeepSea."""

    print("cfg.params.url", cfg.params.url)


if __name__ == "__main__":
    main()
