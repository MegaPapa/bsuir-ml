from ml_1 import lab_1
from ml_2 import lab_2
from ml_3 import lab_3
from ml_4 import lab_4
from ml_5 import lab_5
from ml_6 import lab_6
from ml_7 import lab_7
from ml_8 import lab_8
from ml_9 import lab_9
# from ml_10 import lab_10

DEFAULT_RUNNABLE_LAB = "7"


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
        "5": lab_5.FifthLab(),
        "6": lab_6.SixthLab(),
        "7": lab_7.SeventhLab(),
        "8": lab_8.EightLab(),
        "9": lab_9.NinthLab(),
        # "10": lab_10.TenthLab()
    }
    options[user_input].run_lab()


if __name__ == '__main__':
    main()
