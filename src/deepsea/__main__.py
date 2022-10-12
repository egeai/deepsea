"""Command-line interface."""
import click
import hydra
from omegaconf import DictConfig, OmegaConf
from .scraper.html import HtmlScraper


@click.command()
@click.version_option()
@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg : DictConfig) -> None:
    """DeepSea."""
    html_scraper = HtmlScraper(url=cfg.params.url, required_content="body")
    html_scraper.pour_soup()


if __name__ == "__main__":
    main(prog_name="deepsea")  # pragma: no cover
