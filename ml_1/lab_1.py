from abstract_lab import Lab
from ml_1.city_profit import CityProfit
from ml_1.real_estate import RealEstate
from util import graph
from util import linear_regression
import numpy as np

PATH_TO_CITY_PROFIT = "./ml_1/resources/ex1data1.txt"
PATH_TO_REAL_ESTATE_DATA = "./ml_1/resources/ex1data2.txt"

STEP_FOR_PREDICTABLE_VALUE = 5
ALPHA = 0.01


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
        self.analyze_profit()

    def analyze_profit(self):
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
        # container for cost function values
        costs = []
        thetas_container = []
        thetas = linear_regression.calc_gradient(x, y, ALPHA, np.ones((2, 1)), costs, thetas_container, 10000)

        graph.show_3d_plot_for_cost_function(thetas_container, costs)
        graph.show_contour_plot_for_cost(thetas_container, costs)

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

    def load_data(self):
        # load city profits data
        file_descriptior = open(PATH_TO_CITY_PROFIT, "r")
        file_lines = file_descriptior.readlines()
        for line in file_lines:
            line_pieces = line.split()
            self.city_profits.append(CityProfit(float(line_pieces[0]), float(line_pieces[1])))
        print("City profits:", self.city_profits)

        file_descriptior = open(PATH_TO_REAL_ESTATE_DATA, "r")
        file_lines = file_descriptior.readlines()
        for line in file_lines:
            line_pieces = line.split()
            self.real_estate_data.append(
                RealEstate(float(line_pieces[0]), float(line_pieces[1]), float(line_pieces[2])))
        print("Real estate data:", self.real_estate_data)
