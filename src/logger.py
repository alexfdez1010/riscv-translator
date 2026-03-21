import inspect
import sys
from dataclasses import dataclass
from datetime import datetime
from threading import Lock


_LEVELS = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
}


@dataclass(slots=True)
class _LoggerConfig:
    level: int = _LEVELS["INFO"]


_CONFIG = _LoggerConfig()
_OUTPUT_LOCK = Lock()


class TerminalLogger:
    def __init__(self, name: str):
        self.name = name

    def debug(self, message: str, *args):
        self._log("DEBUG", message, *args)

    def info(self, message: str, *args):
        self._log("INFO", message, *args)

    def warning(self, message: str, *args):
        self._log("WARNING", message, *args)

    def error(self, message: str, *args):
        self._log("ERROR", message, *args)

    def exception(self, message: str, *args):
        self._log("ERROR", message, *args)

    def _log(self, level_name: str, message: str, *args):
        level = _LEVELS[level_name]
        if level < _CONFIG.level:
            return

        rendered_message = message % args if args else message
        timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        context = self._caller_context()
        lines = str(rendered_message).splitlines() or [""]
        prefix = f"[{timestamp}] [{level_name:^7}] {context:<30} | "
        continuation_prefix = " " * len(prefix)
        formatted = [f"{prefix}{lines[0]}"]
        formatted.extend(f"{continuation_prefix}{line}" for line in lines[1:])

        with _OUTPUT_LOCK:
            sys.stdout.write("\n".join(formatted) + "\n")
            sys.stdout.flush()

    def _caller_context(self) -> str:
        frame = inspect.currentframe()
        if frame is None:
            return self.name

        try:
            current = frame.f_back
            while current is not None:
                module_name = current.f_globals.get("__name__", "")
                if module_name != __name__:
                    function_name = current.f_code.co_name
                    if function_name == "<module>":
                        return self.name
                    if "self" in current.f_locals:
                        class_name = current.f_locals["self"].__class__.__name__
                        return f"{class_name}.{function_name}"
                    if "cls" in current.f_locals and hasattr(
                        current.f_locals["cls"], "__name__"
                    ):
                        class_name = current.f_locals["cls"].__name__
                        return f"{class_name}.{function_name}"
                    return function_name
                current = current.f_back
            return self.name
        finally:
            del frame


def configure_logging(level: str = "INFO"):
    _CONFIG.level = _LEVELS.get(level.upper(), _LEVELS["INFO"])


def get_logger(name: str) -> TerminalLogger:
    return TerminalLogger(name)
