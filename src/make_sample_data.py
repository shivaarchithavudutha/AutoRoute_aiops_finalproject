"""Create a small FAKE complaints CSV in the CFPB format, for dry runs before
you download the real data.   Usage: python src/make_sample_data.py [rows]"""
import os
import random
import sys

import pandas as pd

THEMES = {
    "Credit reporting, credit repair services, or other personal consumer reports":
        ["credit report", "equifax", "transunion", "experian", "score dropped", "incorrect account on my report", "dispute the item"],
    "Debt collection": ["debt collector", "calling me every day", "collection agency", "validate the debt", "harassing phone calls", "owe"],
    "Credit card or prepaid card": ["credit card", "annual fee", "interest rate", "charged twice", "card was declined", "rewards points"],
    "Mortgage": ["mortgage", "escrow", "foreclosure", "loan modification", "monthly payment increased", "lender"],
    "Checking or savings account": ["checking account", "overdraft fee", "savings account", "bank closed my account", "deposit on hold"],
    "Student loan": ["student loan", "servicer", "income driven repayment", "forbearance", "loan forgiveness"],
}
FILLER = ["i", "was", "told", "that", "they", "would", "fix", "this", "but", "nothing", "happened", "please", "help",
          "xxxx", "on", "xx/xx/2020", "company", "never", "responded", "again", "and"]


def make(rows: int, seed: int = 0) -> pd.DataFrame:
    rnd = random.Random(seed)
    names = list(THEMES)
    recs = []
    for i in range(rows):
        product = rnd.choice(names)
        words = rnd.sample(THEMES[product], k=3) + [rnd.choice(FILLER) for _ in range(rnd.randint(15, 40))]
        rnd.shuffle(words)
        recs.append({"Date received": f"2020-{rnd.randint(1, 12):02d}-{rnd.randint(1, 28):02d}",
                     "Product": product, "Consumer complaint narrative": " ".join(words)})
    return pd.DataFrame(recs)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    os.makedirs("data/raw/train", exist_ok=True)
    make(n).to_csv("data/raw/train/sample_train.csv", index=False)
    print(f"Wrote {n} fake rows to data/raw/train/sample_train.csv")
