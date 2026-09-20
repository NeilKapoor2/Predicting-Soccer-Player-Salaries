# Predicting Soccer Player Salaries

Can a computer guess how much a soccer player gets paid? I wanted to find out, so I built two machine learning projects that try to predict player salaries.

> **Disclaimer:** All of the code in this repo was written by me. I did use AI to generate the initial version of this README.

## What's in here

| File | What it is |
|------|-----------|
| `predicting_player_salaries.py` / `Predicting_Player_Salaries.ipynb` | Project 1: predicting Major League Soccer (MLS) base salaries |
| `performance_based_salaries.py` / `Performance_Based_Salaries.ipynb` | Project 2: predicting salaries from on-the-field stats |
| `requirements.txt` | The Python libraries you need |
| `dfAll.csv` (also zipped in `archive.zip`) | A copy of the player stats + wages data |

The `.py` files and the notebooks are the same code. I originally wrote everything in Google Colab, and the `.py` files are just the Colab notebooks exported. (There's also a second copy of the performance file named `performance_based_salaries (1).py`. That's a leftover download.)

## Project 1: MLS salaries

**Data:** [US Major League Soccer Salaries](https://www.kaggle.com/datasets/crawford/us-major-league-soccer-salaries) from Kaggle. I combined every season's CSV into one big table and dropped rows with missing values.

**Goal:** Predict a player's `base_salary`.

**Inputs:** club, first name, last name, position, and guaranteed compensation.

**How it works:** I split the data 80% for training and 20% for testing. Then I trained a bunch of different models and compared them:

- Linear Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors
- Neural Network (MLP)
- An ensemble that combines the models together

I also tried one round with `guaranteed_compensation` taken out, to see how much the models were leaning on it. At the end there's a section where I plug in a player by hand (Kevin De Bruyne) and see what each model says he should make.

## Project 2: Performance-based salaries

**Data:** [Undervalued Football Players](https://www.kaggle.com/datasets/armaanmartins21/undervalued-football-players) from Kaggle.

**Goal:** Predict a player's `Annual USD` salary using only how they play.

**Inputs:** matches played, starts, minutes, 90s played, goals, assists, goals + assists, expected goals (xG), expected assists (xAG), and progressive carries / passes / receptions.

**How it works:** This one uses a 3-way split: training, validation, and testing (about 64% / 16% / 20%). I used the validation set to try out different hyperparameters and pick the best settings for each model, and then checked the final score on the test set. Models:

- Linear Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (with scaled features)
- Neural Network (MLP)
- A weighted ensemble

For both projects, I compared models using Mean Squared Error (MSE) and percent error.

## How to run it

1. Make sure you have Python 3 installed.
2. Install the libraries:

   ```bash
   pip install -r requirements.txt
   ```

3. Run a project:

   ```bash
   python predicting_player_salaries.py
   ```

   ```bash
   python performance_based_salaries.py
   ```

   Or open the `.ipynb` files in Jupyter or Google Colab.

The scripts download their datasets from Kaggle automatically using `kagglehub`, so you might need to be logged in to Kaggle the first time.

## Things to know

- The train/test split is random and I didn't set a seed, so your numbers will change a little every time you run it.
- Player names are used as inputs in Project 1. That helps the model memorize players it has seen, but it doesn't really "understand" salaries, so it's not a great sign of how it would do on brand new players.
- Salaries depend on a lot of stuff that isn't in the data (like contracts, fame, and negotiation), so no model here is going to be perfect.

## Built with

Python, pandas, NumPy, scikit-learn, and kagglehub.
