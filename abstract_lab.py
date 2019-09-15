from abc import abstractmethod


class Lab:

    def __init__(self, lab_number):
        self.lab_number = lab_number

    @abstractmethod
    def run_lab(self):
        print("Running lab number", self.lab_number)
