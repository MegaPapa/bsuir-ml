from abc import abstractmethod


class Lab:

    def __init__(self, lab_number):
        self.labNumber = lab_number

    @abstractmethod
    def run_lab(self):
        pass
