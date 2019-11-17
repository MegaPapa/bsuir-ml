from ml_1 import lab_1
from ml_2 import lab_2
from ml_3 import lab_3

DEFAULT_RUNNABLE_LAB = "3"


def main():
    if DEFAULT_RUNNABLE_LAB == "-1":
        print("Which ML lab do you want to start?")
        user_input = input()
    else:
        user_input = DEFAULT_RUNNABLE_LAB
    options = {
        "1": lab_1.FirstLab(),
        "2": lab_2.SecondLab(),
        "3": lab_3.ThirdLab()
    }
    options[user_input].run_lab()


if __name__ == '__main__':
    main()
