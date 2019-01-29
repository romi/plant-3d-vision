from abc import ABC, abstractmethod

class ProcessingBlock(ABC):
    def __init__(self, input_filesets, output_filesets, params):
        for fileset in self.get_input_filesets():
            assert(fileset in input_filesets.keys())
        for fileset in self.get_output_filesets():
            assert(fileset in output_filesets.keys())
        self.input_filesets = input_filesets
        self.output_filesets = output_filesets
        self.params = params
        super().__init__()

    @abstractmethod
    def process(self):
        pass

    @abstractmethod
    def get_input_filesets(self):
        pass

    @abstractmethod
    def get_output_filesets(self):
        pass

