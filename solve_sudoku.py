from pyimagesearch.sudoku import find_puzzle
from pyimagesearch.sudoku import extract_digit
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from sudoku import Sudoku
import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="Path to your trained model")
ap.add_argument("-i", "--image", required=True, help="Path to input image")
ap.add_argument("-d", "--debug", type=int, default=-1, help="decision for visualization")
args = vars(ap.parse_args())

print(">>Beep boop...loading the dl model...")
model = load_model(args["model"])

print(">>Onto image loading...wait a bit...")
image = cv2.imread(args["image"])
image = imutils.resize(image, width=600)

(puzzleImage, warped) = find_puzzle(image, debug=args["debug"]>0)

board = np.zeros((9,9), dtype="int")

stepX = warped.shape[1]//9
stepY = warped.shape[0]//9

cellLocs = []

for y in range(0,9):
    row = []

    for x in range(0,9):
        startX = x*stepX
        startY = y*stepY
        endX = (x+1)*stepX
        endY = (y+1)*stepY
        row.append((startX,startY,endX,endY))

        cell = warped[startY:endY, startX:endX]
        digit = extract_digit(cell, debug=args["debug"]>0)

        if digit is not None:
            roi = cv2.resize(digit, (28,28))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            pred = model.predict(roi).argmax(axis=1)[0]
            board[y, x] = pred
    
    cellLocs.append(row)

print(">> Successfully implemented OCR :-")
puzzle = Sudoku(3, 3, board=board.tolist())
puzzle.show()

print(">> Your answer is getting ready...")
solution = puzzle.solve()
solution.show_full()

for (cellRow, boardRow) in zip(cellLocs, solution.board):
    for (box, digit) in zip(cellRow, boardRow):
        startX, startY, endX, endY = box

        testX = int((endX - startX) * 0.33)
        testY = int((endY - startY) * -0.2)
        testX += startX
        testY += endY

        cv2.putText(puzzleImage, str(digit), (testX,testY), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0,255,255), 2)

cv2.imshow(">>Here's your answer ", puzzleImage)
cv2.waitKey(0)

