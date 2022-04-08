import pandas as pd

from sklearn.model_selection import train_test_split


def main():
    df = pd.read_csv('data/train_data.csv',
                     error_bad_lines=False, warn_bad_lines=False)
    train, test = train_test_split(df, train_size=.7, random_state=42)
    val, test = train_test_split(test, test_size=.5, random_state=42)

    train.to_csv('data/train.csv')
    val.to_csv('data/val.csv')
    test.to_csv('data/test.csv')


if __name__ == '__main__':
    main()
