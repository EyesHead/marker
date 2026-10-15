import os

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = (
    "1"  # Transformers uses .isin for a simple op, which is not supported on MPS
)

import time
import click

from marker.config.parser import ConfigParser
from marker.config.printer import CustomClickPrinter
from marker.logger import configure_logging, get_logger
from marker.models import create_model_dict
from marker.output import save_output

configure_logging()
logger = get_logger()


@click.command(cls=CustomClickPrinter, help="Convert a single PDF to markdown.")
@click.argument("fpath", type=str)
@ConfigParser.common_options
def convert_single_cli(fpath: str, **kwargs):
    start = time.time()
    config_parser = ConfigParser(kwargs)

    converter_cls = config_parser.get_converter_cls()
    models = (
        create_model_dict()
        if getattr(converter_cls, "requires_model_dict", True)
        else {}
    )
    converter = converter_cls(
        config=config_parser.generate_config_dict(),
        artifact_dict=models,
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
        llm_service=config_parser.get_llm_service(),
    )
    rendered = converter(fpath)
    out_folder = config_parser.get_output_folder(fpath)
    save_output(rendered, out_folder, config_parser.get_base_filename(fpath))

    logger.info(f"Saved markdown to {out_folder}")
    logger.info(f"Total time: {time.time() - start}")

    if not getattr(converter_cls, "requires_model_dict", True):
        status = rendered.metadata.get("cloud_conversion", {}).get("status")
        if status != "complete":
            raise click.exceptions.Exit(1)


if __name__ == "__main__":
    convert_single_cli()
