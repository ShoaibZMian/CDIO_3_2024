from abc import ABC, abstractmethod

class ImageAnalyzerABC(ABC):
    
    @abstractmethod
    def start():
        pass

    @abstractmethod
    def stop():
        pass

class DecisionMakerABC(ABC):

    @abstractmethod
    def start():
        pass

class RobotControllerABC(ABC):

    @abstractmethod
    def start():
        pass