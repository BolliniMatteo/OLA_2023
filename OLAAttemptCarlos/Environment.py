from abc import abstractmethod


class Environment:

    @abstractmethod
    def round(self, pulled_arm):
        pass
