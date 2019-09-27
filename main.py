from ml_1 import lab_1


DEFAULT_RUNNABLE_LAB = "1"


def main():
    if DEFAULT_RUNNABLE_LAB == "-1":
        print("Which ML lab you want to start?")
        user_input = input()
    else:
        user_input = DEFAULT_RUNNABLE_LAB
    options = {
        "1": lab_1.FirstLab()
    }
    options[user_input].run_lab()


if __name__ == '__main__':
    main()
