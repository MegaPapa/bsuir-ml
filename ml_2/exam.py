class Exam:

    def __init__(self, first_exam, second_exam, was_applyed):
        self.first_exam = first_exam
        self.second_exam = second_exam
        self.was_applyed = was_applyed

    def get_first_exam(self):
        return self.first_exam

    def get_second_exam(self):
        return self.second_exam

    def get_was_applyed(self):
        return self.was_applyed
