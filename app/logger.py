import logging
import json
import datetime
import os


class _JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            'timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }
        # Attach any extra fields passed via extra={...}
        for key, val in record.__dict__.items():
            if key not in (
                'args', 'created', 'exc_info', 'exc_text', 'filename',
                'funcName', 'levelname', 'levelno', 'lineno', 'message',
                'module', 'msecs', 'msg', 'name', 'pathname', 'process',
                'processName', 'relativeCreated', 'stack_info', 'thread',
                'threadName',
            ):
                payload[key] = val
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logger(name: str, log_file: str = 'logs/app.log') -> logging.Logger:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.INFO)
    fmt = _JSONFormatter()

    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.propagate = False
    return logger
