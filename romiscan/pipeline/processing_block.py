from abc import ABC, abstractmethod


class ProcessingBlock(ABC):
    @abstractmethod
    def process(self):
        pass
