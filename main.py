from ml_1 import lab_1
from ml_2 import lab_2


DEFAULT_RUNNABLE_LAB = "2"


def main():
    if DEFAULT_RUNNABLE_LAB == "-1":
        print("Which ML lab do you want to start?")
        user_input = input()
    else:
        user_input = DEFAULT_RUNNABLE_LAB
    options = {
        "1": lab_1.FirstLab(),
        "2": lab_2.SecondLab()
    }
    options[user_input].run_lab()


if __name__ == '__main__':
    main()
