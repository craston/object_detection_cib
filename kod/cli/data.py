from __future__ import annotations

import typer
from absl import logging

from kod.data.builder import gen_cache
from kod.data.builder import do_analysis
from kod.data.builder import make_coco_zipf
from kod.data.builder import make_coco_2017

app = typer.Typer()

app.command()(gen_cache)
app.command()(do_analysis)
app.command()(make_coco_zipf)
app.command()(make_coco_2017)

if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    app()
