import typer
from absl import logging

from kod.test_utils.inspect_sampler import inspect_sampler

app = typer.Typer()

app.command()(inspect_sampler)

if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    app()
