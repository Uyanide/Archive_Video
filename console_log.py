import sys
import threading

_log_lock = threading.Lock()


def _get_thread_prefix() -> str:
    thread_name = threading.current_thread().name
    if thread_name == "MainThread" and threading.current_thread() is not threading.main_thread():
        return f"[Thread-{threading.get_ident()}] "
    return f"[{thread_name}] "


def log_error(message: str) -> None:
    """Log an error message to stderr."""
    with _log_lock:
        prefix = _get_thread_prefix()
        print(f"\033[31m{prefix}[ERROR]\033[0m ", end="", file=sys.stderr)
        print(message, file=sys.stderr)
        print("", file=sys.stderr)


def log_warning(message: str) -> None:
    """Log a warning message to stderr."""
    with _log_lock:
        prefix = _get_thread_prefix()
        print(f"\033[33m{prefix}[WARN]\033[0m ", end="", file=sys.stderr)
        print(message, file=sys.stderr)
        print("", file=sys.stderr)


def log_success(message: str) -> None:
    """Log a success message to stdout."""
    with _log_lock:
        prefix = _get_thread_prefix()
        print(f"\033[32m{prefix}[SUCCESS]\033[0m ", end="")
        print(message)
        print("")


def log_info(message: str) -> None:
    """Log an informational message to stdout."""
    with _log_lock:
        prefix = _get_thread_prefix()
        print(f"\033[34m{prefix}[INFO]\033[0m ", end="")
        print(message)
        print("")


def log_debug(message: str) -> None:
    """Log a debug message to stdout."""
    with _log_lock:
        prefix = _get_thread_prefix()
        print(f"\033[37m{prefix}[DEBUG]\033[0m ", end="", file=sys.stderr)
        print(message, file=sys.stderr)
        print("", file=sys.stderr)


if __name__ == "__main__":
    log_error("This is an error message.")
    log_warning("This is a warning message.")
    log_success("This is a success message.")
    log_info("This is an informational message.")
    log_debug("This is a debug message.")
