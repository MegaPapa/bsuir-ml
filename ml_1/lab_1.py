from abstract_lab import Lab
import matplotlib as plot

PATH_TO_CITY_PROFIT = "./ml_1/resources/ex1data1.txt"
PATH_TO_REAL_ESTATE_DATA = "./ml_1/resources/ex1data2.txt"


class CityProfit:

    def __init__(self, population, profit):
        self.population = population
        self.profit = profit

    def get_population(self):
        return self.population

    def get_profit(self):
        return self.profit

    def __str__(self):
        return "Population: {self.population} --- Profit: {self.profit}"


class RealEstate:

    def __init__(self, foots, rooms, value):
        self.foots = foots
        self.rooms = rooms
        self.value = value


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
        self.analyze_profit()

    def analyze_profit(self):
        for city_profit in self.city_profits:
            print(city_profit.get_population())

    def load_data(self):
        # load city profits data
        file_descriptior = open(PATH_TO_CITY_PROFIT, "r")
        file_lines = file_descriptior.readlines()
        for line in file_lines:
            line.split()
            self.city_profits.append(CityProfit(line[0], line[1]))
        print("City profits:", self.city_profits)

        file_descriptior = open(PATH_TO_REAL_ESTATE_DATA, "r")
        file_lines = file_descriptior.readlines()
        for line in file_lines:
            line.split()
            self.real_estate_data.append(RealEstate(line[0], line[1], line[2]))
        print("Real estate data:", self.real_estate_data)
