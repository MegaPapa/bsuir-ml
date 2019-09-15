

globalLab = {
    "1": FirstLab()
}

if __name__ == '__main__':
    print("Which ML lab you want to start?")
    userInput = input()
    loadedLab = globalLab[userInput]
    loadedLab.run_lab()
