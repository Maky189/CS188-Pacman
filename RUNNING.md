RUNNING.md

How to run tests and examples locally (zsh / Linux)

Prerequisites:
- Python 3 installed (this repo uses standard library only).
- From the project root (where `autograder.py` and `pacman.py` live), run the commands below.

Run the autograder for a specific question group (example: q1):

```bash
# From the repository root
cd '/home/marcosg/Documents/Artificial Inteligence/search'
python3 autograder.py -q q1
```

Run multiple question groups (q1, q2, q3):

```bash
cd '/home/marcosg/Documents/Artificial Inteligence/search'
python3 autograder.py -q q1
python3 autograder.py -q q2
python3 autograder.py -q q3
```

Run Pacman with a SearchAgent (example using tinyMaze):

```bash
# Run Pacman with SearchAgent and tinyMaze (tinyMazeSearch)
cd '/home/marcosg/Documents/Artificial Inteligence/search'
python3 pacman.py -l tinyMaze -p SearchAgent -afn=tinyMazeSearch
```

Run Pacman using depth-first search on mediumMaze:

```bash
cd '/home/marcosg/Documents/Artificial Inteligence/search'
python3 pacman.py -l mediumMaze -p SearchAgent -afn=depthFirstSearch
```

Notes and troubleshooting:
- If the display/graphics are not desired, add options to use text display or limit frame time. See `pacman.py` help for more options.
- If you run into Python version issues, ensure `python3` points to a supported Python 3.x interpreter.
- For autograder failures, inspect the test files under `test_cases/` and the `search.py`/`searchAgents.py` implementations.
