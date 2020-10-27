# Real Time Sudokku Solver using Open CV and Keras.
Solves the complex sudoku problem easily and display the output in the console. This repo consists of the python code of solving the sudoku puzzle using the opencv, keras and backtracking.
# Algorithm used
Backtracking is a general algorithm for finding all solutions recursively by building solutions recusrsively,one piece at a time removing those solutions thta fails to satisfy the 
constraint of the problem at any one point of time.
I used the backtracking algorithm for solving the sudoku and filling the empty spaces with appropirate values.
# Brief project technology description
- Get the sudokuy block from the image - OpenCV Python
- Extract the sudoku grid from the image - OpenCV Python
- Extracting the digits from the extracted sudoku grid in same ordered manner - python
- Digit prediction using the MNIST dataset - Keras Python CNN
- Solving the sudoku using the backtracking - Python
 
# Extracting the sudoku from the image
1. apply the Adaptive Thresholding and Gaussian Blur on the image.
2. find the corners of the largest contour of the image using findcontours
3. get the index of the point using the operator.itegetter
4. Now we have 4 corners of the sudoku from the image, nnow we can crop and wrap the rectangular secttion from the image.
# Extracting the digits from the sudoku
1. Now infer 81 cells of equal size from the extraced images
# Predicting the digits
1. Here, we can predict the digits using the keras and MNIST dataset.
2. the accuracy of the model was good with both the training data and testing data(Low bias and Low variance).
3. Prediction can be done in the same order and the number store the predicted value into list
# Solving sudoku
1. use the backtracking algorithm for solving the sudoku grid.
# Working
i have used different images for testing and working one of them is:
1. imput image of sudoku
![sudoku](https://github.com/sahilsandhu/Sudoku-Solver-using-Open-Cv/blob/master/sudoku.jpg)
2. Image of sudoku after thresholding
![Threshold](https://github.com/sahilsandhu/Sudoku-Solver-using-Open-Cv/blob/master/threshold.jpg)
3. Contour corners of sudoku
![Contour Image](https://github.com/sahilsandhu/Sudoku-Solver-using-Open-Cv/blob/master/contour.jpg)
4. Wrap Image
![wrap](https://github.com/sahilsandhu/Sudoku-Solver-using-Open-Cv/blob/master/warp.jpg)
5. Final output of extracted image
![Final](https://github.com/sahilsandhu/Sudoku-Solver-using-Open-Cv/blob/master/final.jpg)
6. Extracted Grid
![Extracted](https://github.com/sahilsandhu/Sudoku-Solver-using-Open-Cv/blob/master/extracted_grid.png)
7. Solved Grid
![Solved Grid](https://github.com/sahilsandhu/Sudoku-Solver-using-Open-Cv/blob/master/solved_grid.png)

 
# learning resouces
1. youtube :- https://www.youtube.com/watch?v=QR66rMS_ZfA
2. blog::- https://towardsdatascience.com/open-cv-based-sudoku-solver-powered-by-rust-df256653d5b3
