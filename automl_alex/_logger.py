import sys
from loguru import logger

logger.remove()
logger.add(
    sys.stderr,
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>",
    level=10,
)
logger.add(
    ".automl-alex_tmp/log.log", rotation="1 MB", level="DEBUG", compression="zip"
)


def logger_print_lvl(verbose):
    if verbose > 2:
        lvl = 10
    elif verbose == 2:
        lvl = 20
    elif verbose == 1:
        lvl = 30
    else:
        lvl = 40

    logger.remove()
    logger.add(
        sys.stderr,
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>",
        level=lvl,
    )
    logger.add(
        ".automl-alex_tmp/log.log", rotation="1 MB", level="DEBUG", compression="zip"
    )
