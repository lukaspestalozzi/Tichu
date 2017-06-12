

class TichuError(Exception):
    """
    Parent class for all Error of the Tichu Module
    """
    pass


class LogicError(TichuError):
    """
    Exception raised for LogicErrors.
    Raise this instead of writing 'Should never happen'.
    """
    pass


class IllegalActionError(TichuError):
    """
    Exception raised when a Player makes an illegal Move
    """
    pass


class TichuEnvValueError(TichuError, ValueError):
    """
    A Value Error
    """
    pass


class TichuEnvTypeError(TichuError, TypeError):
    """
    A Type Error
    """
    pass


class NotSupportedError(TichuError):
    """
    Raised when calling a method that is not supported
    """
    pass