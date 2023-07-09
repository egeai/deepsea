"""Command-line interface."""
import click
import hydra
from omegaconf import DictConfig, OmegaConf

# from scraper.html import HtmlScraper
# from scraper.literal_click_option import LiteralOption


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
# @click.option('--url', prompt='Url to scrape', help='Url argument of scrape process.')
@click.option('--message', '-m', multiple=True)
@click.version_option()
def additional_params(message):   # url: str,

    @hydra.main(version_base=None, config_path="../../conf", config_name="config")
    def main(cfg: DictConfig) -> None:
        """DeepSea."""
        click.echo('\n'.join(message))
        # click.echo("Url, type: {}  value: {}".format(
        #    type(url), url))

        # print("Url to scrape: ", url)
        print("cfg.params.url", cfg.params.url)


        # if url:
        #    print("hey! here...")
        #    url_to_scrape = url
        # elif cfg.params.url:
        #    url_to_scrape = cfg.params.url
        # else:
        #    click.echo(click.style('Find a Url!', fg='green'))
        #    return

        # html_scraper = HtmlScraper(url=cfg.params.url, required_content="body")
        # html_scraper.pour_soup()

        # print(OmegaConf.to_yaml(cfg))
    main()  # pragma: no cover ,prog_name="deepsea"


if __name__ == "__main__":
    # import shlex

    additional_params()
    # additional_params(shlex.split(
    #    '''--option1 '["o11", "o12", "o13"]' --option2 '["o21", "o22", "o23"]' '''))
    # main(prog_name="deepsea")  # pragma: no cover
