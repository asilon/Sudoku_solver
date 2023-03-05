import cv2
import imutils
import numpy as np
from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
from train import train_model
import argparse


def find_puzzle(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 3)

    thresh = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)

    # find contours
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    puzzleCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            puzzleCnt = approx
            break

    if puzzleCnt is None:
        raise Exception("Could not find the puzzle!")

    puzzle = four_point_transform(image, puzzleCnt.reshape(4, 2))
    warped = four_point_transform(gray, puzzleCnt.reshape(4, 2))

    return (puzzle, warped)


def extract_digit(cell):
    thresh = cv2.threshold(cell, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = clear_border(thresh)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if len(cnts) == 0:
        return None
    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, -1)

    (h, w) = thresh.shape
    percentFilled = cv2.countNonZero(mask) / float(w * h)

    if percentFilled < 0.025:
        return None
    digit = cv2.bitwise_and(thresh, thresh, mask=mask)

    return digit


def solve_sudoku(board):
    # find an empty cell on the board
    row, col = find_empty_cell(board)

    # if all cells are filled, the board is solved
    if row is None:
        return True

    # try placing numbers 1-9 in the empty cell
    for num in range(1, 10):
        if is_valid(board, row, col, num):
            board[row][col] = num
            if solve_sudoku(board):
                return True
            board[row][col] = 0

    # if none of the numbers 1-9 worked, backtrack
    return False


def find_empty_cell(board):
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                return i, j
    return None, None


def is_valid(board, row, col, num):
    # check if the same number appears in the same row
    if num in board[row]:
        return False

    # check if the same number appears in the same column
    if num in board[:, col]:
        return False

    # check if the same number appears in the same 3x3 subgrid
    subgrid_row, subgrid_col = 3 * (row // 3), 3 * (col // 3)
    if num in board[subgrid_row:subgrid_row + 3, subgrid_col:subgrid_col + 3]:
        return False

    return True


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to input image")
    args = vars(ap.parse_args())

    image = cv2.imread(args["image"])
    image = imutils.resize(image, width=600)

    (puzzleImage, warped) = find_puzzle(image)

    board = np.zeros((9, 9), dtype="int")

    stepX = warped.shape[1] // 9
    stepY = warped.shape[0] // 9

    cellLocs = []

    model, scaler = train_model()

    for y in range(0, 9):
        # initialize the current list of cell locations
        row = []
        for x in range(0, 9):
            startX = x * stepX
            startY = y * stepY
            endX = (x + 1) * stepX
            endY = (y + 1) * stepY
            row.append((startX, startY, endX, endY))

            cell = warped[startY:endY, startX:endX]
            digit = extract_digit(cell)
            if digit is not None:
                digit = cv2.resize(digit, (28, 28))
                roi = digit.reshape((1, 28 * 28))
                roi = scaler.transform(roi)
                pred = model.predict(roi)
                board[y, x] = pred
        cellLocs.append(row)

    if solve_sudoku(board):
        print(board)
    else:
        print("Solution not found")

    for (cellRow, boardRow) in zip(cellLocs, board):
        for (box, digit) in zip(cellRow, boardRow):
            startX, startY, endX, endY = box
            textX = int((endX - startX) * 0.33)
            textY = int((endY - startY) * -0.2)
            textX += startX
            textY += endY
            cv2.putText(puzzleImage, str(digit), (textX, textY),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow("Sudoku Result", puzzleImage)
    cv2.waitKey(0)
