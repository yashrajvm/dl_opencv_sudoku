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
ap.add_argument("-d", "--debug", type=int, default=-1, help=decision for visualization)
args = vars(ap.parse_args())

print(">>Beep boop...loading the dl model...")
model = load_model(args["model"])

print(">>Onto image loading...wait a bit...")
image = cv2.imread(args["image"])
image = imutils.resize(image, width=600)

(puzzleImage, warped) = find_puzzle(image, debug=args["debug"]>0)

board = np.zeros((9,9), dtype="int")



