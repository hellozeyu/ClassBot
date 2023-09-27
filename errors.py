"""
error handler
"""


class InvalidUsage(Exception):
    """
    Error handler
    """

    def __init__(self, message, status_code=None, payload=None):
        """
        Initialize error
        :param message: error message
        :param status_code: error status code
        :param payload: additional information with error
        """
        Exception.__init__(self)
        self.message = message
        self.status_code = status_code if isinstance(status_code, int) else 400
        self.payload = payload

    def to_dict(self):
        """
        Convert error to dict
        :return: dict
        """
        payload = dict(self.payload or ())
        payload['message'] = self.message
        return payload
