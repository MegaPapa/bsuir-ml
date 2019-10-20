from abstract_lab import Lab
from ml_1.city_profit import CityProfit
from ml_1.real_estate import RealEstate
from util import graph
from util.algorithms import linear_regression, normal_equation
import numpy as np

PATH_TO_CITY_PROFIT = "./ml_1/resources/ex1data1.txt"
PATH_TO_REAL_ESTATE_DATA = "./ml_1/resources/ex1data2.txt"

STEP_FOR_PREDICTABLE_VALUE = 5
ALPHA = 0.0242


class FirstLab(Lab):

    def __init__(self):
        super().__init__(self)
        self.lab_number = 1
        self.city_profits = []
        self.real_estate_data = []

    def run_lab(self):
        super().run_lab()
        # load initial data
        self.load_data()
        # analyze depending between profit and population
        # self.analyze_profit()
        self.analyze_real_estate()

    def analyze_real_estate(self):
        x1, x2, y, real_estate_data = self.load_and_prepare_ex2()
        x = np.concatenate((x1, x2), axis=1)
        costs = []
        thetas_container = []
        thetas = linear_regression.calc_gradient(x, y, ALPHA, np.ones((3, 1)), costs, thetas_container, 15)
        print(thetas)
        x_normalized = linear_regression.mean_feature_normalization(x, 2)
        thetas = linear_regression.calc_gradient(x_normalized, y, ALPHA, np.ones((3, 1)), costs, thetas_container, 10000)
        print(thetas)
        thetas = normal_equation.calc_normal_equation(x, y)
        print(thetas)

    def analyze_profit(self):
        x, y, profit_data = self.load_and_prepare_ex1()
        # container for cost function values
        costs = []
        thetas_container = []
        thetas = linear_regression.calc_gradient(x, y, ALPHA, np.ones((2, 1)), costs, thetas_container, 300)

        graph.show_3d_plot_for_cost_function(thetas_container, costs, x, y)
        graph.show_contour_plot_for_cost(thetas_container, costs, x, y)
        graph.show_gradient_descent_steps_in_3d(thetas_container, costs)

        # creates empty values which we will predict
        predictable_values = np.zeros(shape=(5, 1))
        for i in range(len(predictable_values)):
            predictable_values[i][0] = (i + 1) * STEP_FOR_PREDICTABLE_VALUE
        # predict them
        predicted = np.dot(np.concatenate((np.ones((len(predictable_values), 1)), predictable_values), axis=1), thetas)

        # prepare predicted values to print
        predict_to_print = np.concatenate((np.ones((len(predicted), 1)), predicted), axis=1)
        for i in range(len(predictable_values)):
            predict_to_print[i][0] = (i + 1) * STEP_FOR_PREDICTABLE_VALUE

        # show predicted values
        graph.show_plot_by_graph_and_points(profit_data, predict_to_print)

    def load_and_prepare_ex2(self):
        real_estate_data = np.zeros(shape=(len(self.real_estate_data), 3))
        for i in range(len(self.real_estate_data)):
            real_estate_data[i][0] = self.real_estate_data[i].get_foots()
            real_estate_data[i][1] = self.real_estate_data[i].get_rooms()
            real_estate_data[i][2] = self.real_estate_data[i].get_value()
        x1, x2, y = np.hsplit(real_estate_data, 3)
        return (x1, x2, y, real_estate_data)

    def load_and_prepare_ex1(self):
        # profit data here, is matrix n * 2
        profit_data = np.zeros(shape=(len(self.city_profits), 2))
        # fill profit data
        for i in range(len(self.city_profits)):
            profit_data[i][0] = self.city_profits[i].get_population()
            profit_data[i][1] = self.city_profits[i].get_profit()
        # show initial points
        graph.show_plot_by_points(profit_data)
        # split to Xs and Ys
        x, y = np.hsplit(profit_data, 2)
        return (x, y, profit_data)

    def load_data(self):
        # load city profits data
        file_descriptior = open(PATH_TO_CITY_PROFIT, "r")
        file_lines = file_descriptior.readlines()
        for line in file_lines:
            line_pieces = line.split()
            if (len(line_pieces) == 2):
                self.city_profits.append(CityProfit(float(line_pieces[0]), float(line_pieces[1])))
        print("City profits:", self.city_profits)

        file_descriptior = open(PATH_TO_REAL_ESTATE_DATA, "r")
        file_lines = file_descriptior.readlines()
        for line in file_lines:
            line_pieces = line.split()
            self.real_estate_data.append(
                RealEstate(float(line_pieces[0]), float(line_pieces[1]), float(line_pieces[2])))
        print("Real estate data:", self.real_estate_data)
