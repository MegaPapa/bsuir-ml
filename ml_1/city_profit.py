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
