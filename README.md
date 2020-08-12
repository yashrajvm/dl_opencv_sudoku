 Automatic sudoku solver with Deep Learning and OpenCV:
 
1. Trained digit recognizer with Keras and Tensorflow using MNIST dataset by implementing a Convolutional Neural Network to OCR the digits
2. Extracted sudoku puzzle from image and identified digits from the cell using OpenCV
3. Output is solved sudoku with final solution masked onto original puzzle

Run the commands in following order.
 1. python3 train_digit_classifier.py --model output/digit_classifier.h5
 2. python3 solve_sudoku.py --model output/digit_classifier.h5 --image sudoku_puzzle2.jpg
