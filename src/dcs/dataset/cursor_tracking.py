from time import sleep
import pyautogui

if __name__ == '__main__':
    while True:
        print(pyautogui.position())
        sleep(0.1)
