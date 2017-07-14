"""
Based on https://github.com/benley/python-glog/blob/master/glog.py
(c) BSD 2-Clause 2015 Benjamin Staffin

Changelog: 2016/02  Removed gflags dependency.
"""
import logging
import os.path
import time


def format_message(record):
    try:
        record_message = '%s' % (record.msg % record.args)
    except TypeError:
        record_message = record.msg
    return record_message


class GlogFormatter(logging.Formatter):
    LEVEL_MAP = {
        logging.FATAL: 'F',
        logging.ERROR: 'E',
        logging.WARN: 'W',
        logging.INFO: 'I',
        logging.DEBUG: 'D'
    }

    def __init__(self):
        logging.Formatter.__init__(self)

    def format(self, record):
        level = GlogFormatter.LEVEL_MAP.get(record.levelno, '?')

        date = time.localtime(record.created)
        date_usec = (record.created - int(record.created)) * 1e6
        record_message = '%c%02d%02d %02d:%02d:%02d.%06d %s %s:%d] %s' % (
            level, date.tm_mon, date.tm_mday, date.tm_hour, date.tm_min,
            date.tm_sec, date_usec,
            record.process if record.process is not None else '?????',
            record.filename,
            record.lineno,
            format_message(record))
        record.getMessage = lambda: record_message
        return logging.Formatter.format(self, record)


logger = logging.getLogger()


def setLevel(newlevel):
    logger.setLevel(newlevel)
    logger.debug('Log level set to %s', newlevel)


debug = logging.debug
info = logging.info
warning = logging.warning
warn = logging.warning
error = logging.error
exception = logging.exception
fatal = logging.fatal
log = logging.log

DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
WARN = logging.WARN
ERROR = logging.ERROR
FATAL = logging.FATAL

_level_names = {
    DEBUG: 'DEBUG',
    INFO: 'INFO',
    WARN: 'WARN',
    ERROR: 'ERROR',
    FATAL: 'FATAL'
}

_level_letters = [name[0] for name in _level_names.values()]

GLOG_PREFIX_REGEX = (
                        r"""
                        (?x) ^
                        (?P<severity>[%s])
                        (?P<month>\d\d)(?P<day>\d\d)\s
                        (?P<hour>\d\d):(?P<minute>\d\d):(?P<second>\d\d)
                        \.(?P<microsecond>\d{6})\s+
                        (?P<process_id>-?\d+)\s
                        (?P<filename>[a-zA-Z<_][\w._<>-]+):(?P<line>\d+)
                        \]\s
                        """) % ''.join(_level_letters)
"""Regex you can use to parse glog line prefixes."""

# Defaults
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(GlogFormatter())
logger.addHandler(stream_handler)
setLevel(logging.DEBUG)

stream_handler.setLevel(logging.INFO)


def add_file_handler(filename):
    if os.path.isfile(filename):
        info('Appending to an existing log file {}'.format(filename))
    else:
        info('New log file {}'.format(filename))
    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(GlogFormatter())
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

# logging.debug(time.strftime("Local time zone: %Z (%z)", time.localtime()))
