import logging
import sys

from app.core.trace import get_trace_id


APP_LOGGER_NAMES = (
    "app",
    "uvicorn",
    "uvicorn.error",
    "uvicorn.access",
)
LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] trace_id=%(trace_id)s %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_TRACE_ID_LOG_RECORD_FACTORY_INSTALLED = False


def install_trace_id_log_record_factory() -> None:
    global _TRACE_ID_LOG_RECORD_FACTORY_INSTALLED

    if _TRACE_ID_LOG_RECORD_FACTORY_INSTALLED:
        return

    previous_factory = logging.getLogRecordFactory()

    def record_factory(*args: object, **kwargs: object) -> logging.LogRecord:
        record = previous_factory(*args, **kwargs)
        if not hasattr(record, "trace_id"):
            record.trace_id = get_trace_id()
        return record

    logging.setLogRecordFactory(record_factory)
    _TRACE_ID_LOG_RECORD_FACTORY_INSTALLED = True


def get_log_level(log_level: str) -> int:
    level = logging.getLevelName(log_level.upper())
    if not isinstance(level, int):
        raise ValueError(f"Unsupported log level: {log_level}")
    return level


def configure_logging(log_level: str) -> None:
    level = get_log_level(log_level)
    install_trace_id_log_record_factory()
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        stream=sys.stdout,
    )

    for logger_name in APP_LOGGER_NAMES:
        logging.getLogger(logger_name).setLevel(level)
