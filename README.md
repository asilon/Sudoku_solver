# Sudoku_solver

This is a Sudoku solver project that takes an image of an unsolved Sudoku puzzle as input and returns the solution.

## Installation

```bash
pip install -r requirements.txt
```

## Example

To see an example of how to use the Sudoku solver, run the following command in your terminal:

```bash
python sudoku_solver.py --image example.jpg
```

This will solve the Sudoku puzzle in the example.jpg file and display the solution.

## Limitations

This Sudoku solver is designed to work with images that meet the following criteria:

    * The Sudoku puzzle is centered and aligned with the image.
    * The numbers in the Sudoku puzzle are clear and distinct.

If the image you provide does not meet these criteria, the program may not be able to solve the puzzle correctly.
