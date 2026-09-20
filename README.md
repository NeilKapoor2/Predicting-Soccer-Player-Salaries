# Predicting Soccer Player Salaries

**I trained a model to predict MLS salaries and got an R² of 0.99. Then I figured out it was cheating.**

## Why I built this

I'm a soccer nerd. I'll happily argue that the best player on the pitch is the one who never shows up in the highlights. Salaries are the number everyone quotes when they say a player is "worth it" or "overpaid," so I wanted to see if a computer could learn that number, and what it would end up learning instead.

## The 0.99 mistake

My first MLS model scored an R² of **0.985** (linear regression) and **0.995** (decision tree). Nothing in soccer is that predictable, so I got suspicious and checked what the model was leaning on.

It was almost all one column: `guaranteed_compensation`. I was predicting **base salary** using a number that is mostly base salary plus bonuses. That's like predicting the final score by reading the scoreboard. It's called data leakage, and it's the ML version of a goal that gets chalked off for offside: it looks amazing until someone checks.

So I took the column out.

## The finding

> **My model wasn't learning who was good. It was learning who already got paid.**

With guaranteed compensation, my ensemble was off by about **16%** on average. Without it, the same setup was off by about **123%**, a jump of roughly **107 percentage points**. Random Forest's test MSE went from about 2.3 billion to about 104 billion.

Salary is a stand-in for skill, and it's a biased one. It reflects contracts, age, where a player came from, and how well his agent negotiated. A model trained on it inherits all of that.

## Project 2: what if I only use how players actually play?

To test this properly, I built a second model that predicts salary **only** from on-field stats (minutes, goals, assists, xG, xAG, progressive carries/passes/receptions) on a dataset of players from Europe's big leagues.

It got much worse, and that's the honest result. My best model (Random Forest) has a test MSE of about 1.6 × 10¹³, which is an error of roughly **$4 million** per player. The stacked model explains about **24%** of the variation in salary. Stats alone don't explain pay.

The misses are fun, though. The model thought Kevin Schade (paid about $681K in the data) should be making $4M or more, and it thought Luke Shaw (about $10.2M) should be making around $3–4M. Some of that is probably the model being wrong, and some is real gaps between pay and production. I can't tell which yet, and I think that's the actual research question.

## What's in here

| File | What it is |
|------|-----------|
| `predicting_player_salaries.py` / `Predicting_Player_Salaries.ipynb` | Project 1: MLS base salaries, including the leakage test |
| `performance_based_salaries.py` / `Performance_Based_Salaries.ipynb` | Project 2: salaries from on-field stats only |
| `requirements.txt` | Python libraries |
| `dfAll.csv` (also in `archive.zip`) | Copy of the player stats + wages data |

The `.py` files are exported from the Colab notebooks, so they're the same code.

**Data:** [US Major League Soccer Salaries](https://www.kaggle.com/datasets/crawford/us-major-league-soccer-salaries) (5,509 player-seasons after dropping missing values) and [Undervalued Football Players](https://www.kaggle.com/datasets/armaanmartins21/undervalued-football-players) (2,831 rows).

**Models (both projects):** Linear Regression, Decision Tree, Random Forest, K-Nearest Neighbors, Neural Network (MLP), and ensembles. Project 2 uses a 64/16/20 train/validation/test split and picks hyperparameters on the validation set.

## How to run it

```bash
pip install -r requirements.txt
python predicting_player_salaries.py
python performance_based_salaries.py
```

Or open the notebooks in Jupyter or Google Colab. The scripts download the data with `kagglehub`, so you may need to be logged in to Kaggle the first time.

## What I'd claim, and what I wouldn't

**I'd claim:**
- The 0.99 came from leakage, and removing `guaranteed_compensation` shows how much the models leaned on it.
- On-field stats alone predict salary poorly, so pay and performance really are different things.

**I wouldn't claim:**
- That either model measures a player's true ability. They predict pay, not skill.
- That the with/without comparison is perfectly controlled. The first run used a random, unseeded split and the "without" run used `random_state=42`, so they were tested on different players. The size of the gap is real, but I'd want to rerun both on the same split before quoting exact numbers.
- That Project 1 works on new players. Names are inputs, and the same player shows up in several seasons, so the model can partly memorize people it has already seen.

**Known issues:**
- Project 2's "meta neural network" has a bug: it predicts salaries of about $100. It looks like the "best" model by average percent error (about 100%), but only because predicting near zero beats the other models' ~360–420% errors, which are inflated by cheap players the models overpay. Percent error is a bad metric here, and I'd trust MSE instead.
- The Kevin De Bruyne demo (he isn't in MLS) shows models disagreeing wildly: linear regression and the neural net said about $17–18M, while the tree-based models and KNN said about $5–6M. Trees can't predict above what they saw in training. That's a useful reminder that a model outside its training range is guessing.
- Some notebook cells were run out of order in Colab, so a few printed numbers may not match a clean top-to-bottom run.

## Why it matters

This started as a soccer question, but the same problem shows up anywhere a number stands in for a person: hiring, rankings, who gets scouted and who gets overlooked. If the number was biased to begin with, a model trained on it will be biased too, just with more confidence. Next I want to try measuring players in a way that doesn't rely on their salary, and see who the numbers have been missing.

## AI note

All of the code in this repo was written by me. I used AI to create and edit my README. The findings and numbers come from my own notebook outputs.

## Built with

Python, pandas, NumPy, scikit-learn, and kagglehub.
