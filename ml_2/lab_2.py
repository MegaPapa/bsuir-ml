from abstract_lab import Lab
from ml_2.exam import Exam
from ml_2.pass_exam import ExamToUniversity
from util.matlab_file_reader import read_matlab_file

PATH_TO_UNIVERSITY_PASS_EXAM_DATA = "./ml_2/resources/ex2data1.txt"
PATH_TO_PASS_EXAM_DATA = "./ml_2/resources/ex2data2.txt"
MATLAB_PICS_DATA = "./ml_2/resources/ex2data3.mat"


class SecondLab(Lab):

    def __init__(self):
        self.lab_number = 2
        self.pass_exams_data = []
        self.exams_data = []
        self.matlab_pics = []

    def run_lab(self):
        super().run_lab()
        self.load_data()
        print()

    def load_data(self):
        # ex2data1
        file_descriptior = open(PATH_TO_UNIVERSITY_PASS_EXAM_DATA, "r")
        file_lines = file_descriptior.readlines()
        for line in file_lines:
            line_pieces = line.split()
            if (len(line_pieces) == 3):
                self.pass_exams_data.append(ExamToUniversity(float(line_pieces[0]), float(line_pieces[1]), int(line_pieces[2])))

        # ex2data2
        file_descriptior = open(PATH_TO_PASS_EXAM_DATA, "r")
        file_lines = file_descriptior.readlines()
        for line in file_lines:
            line_pieces = line.split()
            if (len(line_pieces) == 3):
                self.exams_data.append(
                    Exam(float(line_pieces[0]), float(line_pieces[1]), int(line_pieces[2])))

        #ex2data3
        self.matlab_pics = read_matlab_file(MATLAB_PICS_DATA)
