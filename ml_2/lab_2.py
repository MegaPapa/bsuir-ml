import numpy as np

from abstract_lab import Lab
from ml_2.exam import Exam
from ml_2.pass_exam import ExamToUniversity
from util import graph
from util.algorithms import logistic_regression
from util.file.matlab_file_reader import read_matlab_file
from util.logger import LoggerBuilder

PATH_TO_UNIVERSITY_PASS_EXAM_DATA = "./ml_2/resources/ex2data1.txt"
PATH_TO_PASS_EXAM_DATA = "./ml_2/resources/ex2data2.txt"
MATLAB_PICS_DATA = "./ml_2/resources/ex2data3.mat"

ALPHA = 0.0242
LAMBDA = 1

logger = LoggerBuilder().with_name("lab2").build()


class SecondLab(Lab):

    def __init__(self):
        self.lab_number = 2
        self.pass_exams_data = []
        self.exams_data = []
        self.matlab_pics = []

    def run_lab(self):
        super().run_lab()
        self.load_data()
        # self.analyze_exam_pass_data()
        self.analyze_things_exams()


    def analyze_things_exams(self):
        # (7)
        applyed_things = np.asarray([])
        not_applyed_things = np.asarray([])
        for thing in self.exams_data:
            if thing.get_was_applyed() == 1:
                applyed_things = np.append(applyed_things, thing)
            else:
                not_applyed_things = np.append(not_applyed_things, thing)
        # (8) show points by accepted and not accepted students
        graph.show_points_by_classes([applyed_things, not_applyed_things])
        # (9)
        logistic_regression.print_polynomial_with_two_features(6)
        # (10)




    def analyze_exam_pass_data(self):
        # creates arrays of accepted and not accepted students
        accepted_studs = np.asarray([])
        not_accepted_studs = np.asarray([])
        for pass_exam_stud in self.pass_exams_data:
            if pass_exam_stud.get_was_applyed() == 1:
                accepted_studs = np.append(accepted_studs, pass_exam_stud)
            else:
                not_accepted_studs = np.append(not_accepted_studs, pass_exam_stud)
        # show points by accepted and not accepted students (2)
        graph.show_points_by_classes([accepted_studs, not_accepted_studs])
        # prepare special data, as x's and y, containers, etc.
        x1, x2, y = self.transform_ex2data1()
        x = np.concatenate((x1, x2), axis=1)
        thetas_container = []
        costs_container = []
        # (3)
        initial_thetas = np.ones((3, 1))
        thetas = logistic_regression.logistic_gradient(x, y, initial_thetas, thetas_container, costs_container, ALPHA, 1000)
        logger.info("Thetas from gradient descent \n %s", thetas)
        # (4)
        thetas = logistic_regression.compute_with_nelder_mead(x, np.zeros((3, 1)), y)
        logger.info("Thetas from nelder-mead method \n %s", thetas)

        thetas = logistic_regression.compute_with_tnc(x, np.zeros((3, 1)), y)
        logger.info("Thetas from tnc method \n %s", thetas)

        thetas = logistic_regression.compute_with_bfgs(x, np.zeros((3, 1)), y)
        logger.info("Thetas from bfgs method \n %s", thetas)
        # (5)
        students_first_exam = 55.0
        students_second_exam = 70.0
        marks = np.asarray((1, students_first_exam, students_second_exam))
        z = marks @ thetas
        percents_to_apply = logistic_regression.sigmoid(z)
        logger.info("Student with marks %s and %s has %s percents to apply.", students_first_exam, students_second_exam, percents_to_apply[0] * 100)
        # (6)
        graph.draw_decision_boundary_line(thetas, [accepted_studs, not_accepted_studs])
        # (7)


    # returns data from file in ML style, like x1, x2 ... y
    def transform_ex2data1(self):
        pass_exam_data = np.zeros(shape=(len(self.pass_exams_data), 3))
        for i in range(len(self.pass_exams_data)):
            pass_exam_data[i][0] = self.pass_exams_data[i].get_first_exam()
            pass_exam_data[i][1] = self.pass_exams_data[i].get_second_exam()
            pass_exam_data[i][2] = self.pass_exams_data[i].get_was_applyed()
        x1, x2, y = np.hsplit(pass_exam_data, 3)
        return (x1, x2, y)

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
