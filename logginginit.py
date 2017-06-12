import logging
import os
import errno
import sys
from collections import namedtuple

console_logger_name = "console_logger"
CONSOLE_LOGGER = logging.getLogger(console_logger_name)
stdout_logger_name = "stdout_logger"
root_logger = logging.getLogger()

LoggingMode = namedtuple("LoggingMode", ["redirect_stdout",  # if True, logs all stdout messages to the file 'stdout.log' (nothing is printed to stdout)
                                         "console_logger_level",  # The level of the CONSOLE_LOGGER. This logger logs to stderr
                                         "stderr_level",  # If not None, registers a handler for the Root-logger that logs to stderr. Does not affect the console_logger (If None, nothing is printed to stderr, except the console_logger)
                                         "debug_file", "info_file", "warn_file", "error_file", "critical_file",  # The filenames of the respective log levels. If None, then the no handler is registered for that level.
                                         "all_file", "info_and_above_file", "warn_and_above_file",  # The filenames for the logs containing all records of the respective level and above. If None, then no such handler is registered
                                         ])

# PREDEFINED LOGGING MODES

DefaultMode = LoggingMode(redirect_stdout=False, console_logger_level=logging.INFO, stderr_level=logging.INFO,
                          debug_file='debug.log', info_file='info.log', warn_file='warn.log', error_file='error.log', critical_file='critical.log',
                          all_file='all.log', info_and_above_file='info_plus.log', warn_and_above_file='warn_error_critical.log')

TogetherMode = DefaultMode._replace(debug_file=None, info_file=None, warn_file=None, error_file=None, critical_file=None)
SeparateMode = DefaultMode._replace(all_file=None, info_and_above_file=None, warn_and_above_file=None)

ExperimentMode = SeparateMode._replace(redirect_stdout=True, console_logger_level=None, stderr_level=None, debug_file=None, info_file=None)  # Only seperate files and no console output
TrainMode = ExperimentMode._replace(redirect_stdout=True)
HumanplayMode = DefaultMode._replace(redirect_stdout=True, stderr_level=logging.INFO)
HumanplayCheatMode = HumanplayMode._replace(console_logger_level=logging.DEBUG)

DebugMode = DefaultMode._replace(console_logger_level=logging.DEBUG, stderr_level=logging.DEBUG)  # Logs everything

RunGameMode = SeparateMode._replace(redirect_stdout=False, stderr_level=logging.INFO)  # only stdout and seperate files.

logging_modes = {"DefaultMode": DefaultMode, "TogetherMode": TogetherMode, "SeparateMode": SeparateMode,
                 "ExperimentMode": ExperimentMode, "TrainMode": TrainMode, "HumanplayMode": HumanplayMode,
                 "HumanplayCheatMode": HumanplayCheatMode, "DebugMode": DebugMode, "RunGameMode": RunGameMode}


class LevelFilter(logging.Filter):
    """
    Filter only allowing records of the spezified level
    """
    def __init__(self, level, name: str=None):
        if name:
            super().__init__(name=name)
        else:
            super().__init__()
        self._level = level

    def filter(self, record):
        return super().filter(record) and record.levelno == self._level

    def __str__(self):
        return "{me.__class__}(only allowing {me._level})".format(me=self)


class IgnoreLoggerFilter(logging.Filter):
    """
    Filter only allowing records that don't belong to the given logger
    """
    def __init__(self, logger_name: str):
        super().__init__()
        self._logger_name = logger_name

    def filter(self, record):
        return record.name != self._logger_name

    def __str__(self):
        return "{me.__class__}(ignoring {me._logger_name})".format(me=self)


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, log_level=logging.DEBUG):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass


def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def register_file_handler(logger: logging.Logger, filename: str, level: int, format_string: str, record_filter: logging.Filter=None, verbose: bool=True)->None:
    """
    
    :param logger: 
    :param filename: 
    :param level: The logging level (eg. logging.INFO)
    :param format_string: 
    :param record_filter:
    :param verbose: 
    :return: None
    """
    handler = logging.FileHandler(filename, "w", encoding=None, delay=True)
    return register_handler(handler=handler, logger=logger, level=level, format_string=format_string, record_filter=record_filter, verbose=verbose)


def register_stream_handler(logger: logging.Logger, level: int, format_string: str, stream=None, record_filter: logging.Filter=None, verbose: bool=True) -> None:
    """

    :param logger: 
    :param level: The logging level (eg. logging.INFO)
    :param format_string: 
    :param stream: Default is stderr
    :param record_filter:
    :param verbose: 
    :return: None
    """
    handler = logging.StreamHandler(stream=stream)
    return register_handler(handler=handler, logger=logger, level=level, format_string=format_string, record_filter=record_filter, verbose=verbose)


def register_handler(handler: logging.Handler, logger: logging.Logger, level: int, format_string: str, record_filter: logging.Filter, verbose: bool = True) -> None:
    handler.setLevel(level)
    formatter = logging.Formatter(format_string)
    handler.setFormatter(formatter)
    if record_filter:
        handler.addFilter(record_filter)
    logger.addHandler(handler)
    if verbose:
        print("------------- New Logging Handler ---------------")
        print("Added handler: {}".format(str(handler)))
        print("To Logger: {}".format(str(logger)))
        if record_filter:
            print("With Filter: {}".format(str(record_filter)))


def initialize_loggers(output_dir: str, logging_mode: LoggingMode=DefaultMode, min_loglevel=logging.DEBUG):

    make_sure_path_exists(output_dir)
    format_string = '%(process)d, (%(name)s) [%(levelname)s]: %(message)s'
    root_logger.setLevel(min_loglevel)
    CONSOLE_LOGGER.setLevel(min_loglevel)

    def register_specific_level_root_file_handler(filename: str, level: int):
        if level >= min_loglevel:
            register_file_handler(logger=root_logger, filename=os.path.join(output_dir, filename),
                                  format_string=format_string, record_filter=LevelFilter(level),
                                  level=level)

    def register_root_file_handler(filename: str, level: int):
        if level >= min_loglevel:
            register_file_handler(logger=root_logger, filename=os.path.join(output_dir, filename),
                                  format_string=format_string, record_filter=None,
                                  level=level)
    # redirect stdout
    if logging_mode.redirect_stdout:
        stdout_logger = logging.getLogger(stdout_logger_name)
        stdout_logger.setLevel(logging.DEBUG)
        register_file_handler(stdout_logger, filename=os.path.join(output_dir, 'console.log'), level=logging.DEBUG, format_string=format_string)
        sl = StreamToLogger(stdout_logger)
        sys.stdout = sl

    # console logger
    if logging_mode.console_logger_level is not None:
        register_stream_handler(CONSOLE_LOGGER, level=logging_mode.console_logger_level, format_string='%(message)s')

    # root logger stderr level
    if logging_mode.stderr_level is not None:
        register_stream_handler(root_logger, logging_mode.stderr_level, format_string=format_string, record_filter=IgnoreLoggerFilter(console_logger_name))

    # Specific Level Handlers
    if logging_mode.debug_file is not None:
        register_specific_level_root_file_handler(logging_mode.debug_file, level=logging.DEBUG)

    if logging_mode.info_file is not None:
        register_specific_level_root_file_handler(logging_mode.info_file, level=logging.INFO)

    if logging_mode.warn_file is not None:
        register_specific_level_root_file_handler(logging_mode.warn_file, level=logging.WARNING)

    if logging_mode.error_file is not None:
        register_specific_level_root_file_handler(logging_mode.error_file, level=logging.ERROR)

    if logging_mode.critical_file is not None:
        register_specific_level_root_file_handler(logging_mode.critical_file, level=logging.CRITICAL)

    # Handler for level and above
    if logging_mode.all_file is not None:
        register_root_file_handler(logging_mode.all_file, level=logging.DEBUG)

    if logging_mode.info_and_above_file is not None:
        register_root_file_handler(logging_mode.info_and_above_file, level=logging.INFO)

    if logging_mode.warn_and_above_file is not None:
        register_root_file_handler(logging_mode.warn_and_above_file, level=logging.WARNING)

