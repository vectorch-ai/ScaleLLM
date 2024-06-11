class ValidationError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__()
        self.code = code
        self.message = message

    def __repr__(self) -> str:
        return super().__repr__() + self.message

    def __str__(self) -> str:
        return super().__str__() + self.message
