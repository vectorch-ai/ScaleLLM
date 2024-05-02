
# TODO: define following classes in c++ code
class SequenceOutput:
    def __init__(self):
        self.index = 0
        self.text = ""
        self.finish_reason = None

    def __str__(self):
        return f"index: {self.index}, text: {self.text}, finish_reason: {self.finish_reason}"


class RequestOutput:
    def __init__(self):
        self.id = ""
        self.sequence_outpus = []
        self.status = None
        self.usage = None


