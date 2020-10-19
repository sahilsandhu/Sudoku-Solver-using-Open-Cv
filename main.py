import cv2
import sys
import numpy as np
from time import time
import matplotlib.pyplot as plt
from SudokuExtractor import extract_sudoku
from NumberExtractor import extract_number
from SolveSudoku import sudoku_solver

a = [[0,5,0,9,8,0,0,6,0],
 [2,0,0,0,0,0,0,0,5],
 [0,0,1,0,0,7,0,0,0],
 [5,0,0,2,0,0,9,0,0],
 [4,0,0,0,0,0,0,0,3],
 [0,0,3,0,0,4,0,0,2],
 [0,0,0,7,0,0,3,0,0],
 [8,0,0,0,0,0,0,0,1],
 [0,9,0,0,4,8,0,7,0]]

b = [[0, 5, 0, 4, 4, 0, 0, 6, 0], [2, 0, 0, 0, 0, 0, 0, 0, 5], [0, 0, 1, 0, 0, 7, 0, 0, 0], [5, 0, 0, 2, 0, 0, 4, 0, 0], [4, 0, 0, 0, 0, 0, 0, 0, 3], [0, 0, 3, 0, 0, 4, 0, 0, 2], [0, 0, 0, 7, 0, 0, 3, 0, 0], [8, 0, 0, 0, 0, 0, 4, 2, 1], [0, 9, 0, 0, 4, 8, 0, 7, 0]]




def output(a):
    sys.stdout.write(str(a))

def display_sudoku(sudoku):
    for i in range(9):
        for j in range(9):
            cell=sudoku[i][j]
            if cell ==0 or isinstance(cell,set):
                output('.')
            else:
                output(cell)
            if(j+1)%3 == 0 and j<8:
                output('|')
            if( j!=8):
                output(' ')
        output('\n')
        if(i+1)%3==0 and i<8:
            output("---------------------------------\n")


def show_image(image):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.show()

def main(image):
    img = extract_sudoku(image)
    grid = extract_number(img)
    print('Sudoku:')
    display_sudoku(grid.tolist())
    print(grid.dtype)
    # for i in range(9):
    #    for j in range(9):
    #        print(grid[i][j])

    print("Normal grid", grid)

    print("Grid to list", grid.tolist())


    solution = sudoku_solver(b)
    print(solution)
    print('solution:')
    display_sudoku(solution.tolist())

def convert_sec_to_hms(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return "%d:%02d:%08d" % (hour, minutes, seconds)

if __name__ == '__main__':

    try:
        start_time = time()
        image = cv2.imread('sudoku3.jpg')
        #cv2.imshow('Img',image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        main(image)

    except:             #    except IndexError:
       pass