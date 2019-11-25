from ml_1 import lab_1
from ml_2 import lab_2
from ml_3 import lab_3
from ml_4 import lab_4
from ml_5 import lab_5

DEFAULT_RUNNABLE_LAB = "5"


def main():
    if DEFAULT_RUNNABLE_LAB == "-1":
        print("Which ML lab do you want to start?")
        user_input = input()
    else:
        user_input = DEFAULT_RUNNABLE_LAB
    options = {
        "1": lab_1.FirstLab(),
        "2": lab_2.SecondLab(),
        "3": lab_3.ThirdLab(),
        "4": lab_4.FourthLab(),
        "5": lab_5.FifthLab()
    }
    options[user_input].run_lab()


if __name__ == '__main__':
    main()
