import optuna
import pandas as pd
import numpy as np
import random
import pandas as pd
import numpy as np
import xgboost as xgb

df_train = pd.read_csv('../data/train_processed.csv')
df_train = df_train[df_train.FULLVAL > 0]
df_val = pd.read_csv("../data/val_processed.csv")

train_y = df_train['FULLVAL'].copy()
val_y = df_val['FULLVAL'].copy()

MIN = 80_000
MAX = 1_600_000

# train_y.loc[train_y > MAX] = MAX
# train_y.loc[train_y < MIN] = MIN

# val_y.loc[val_y > MAX] = MAX
# val_y.loc[val_y < MIN] = MIN

cat_names = df_train.columns[19:]
df_train.drop(columns=['FULLVAL'], inplace=True)
df_val.drop(columns=['FULLVAL'], inplace=True)

train_y_log = np.log(train_y + 1)
val_y_log = np.log(val_y + 1)

df_train[cat_names] = df_train[cat_names].astype('category')
df_val[cat_names] = df_val[cat_names].astype('category')

dtrain = xgb.DMatrix(df_train.values, enable_categorical=True,
                     label=train_y, feature_names=df_train.columns)
dval = xgb.DMatrix(df_val.values, enable_categorical=True,
                   label=val_y, feature_names=df_val.columns)


def objective(trial: optuna.Trial):

    param = {'objective': 'reg:squarederror',
             'eval_metric': 'mae',
             "max_depth": trial.suggest_int('max_depth', 1, 12),
             "eta": trial.suggest_float('eta', 0.001, 0.99),
             "gamma": trial.suggest_float('gamma', 0, 50000),
             "subsample": trial.suggest_float('subsample', 0, 1),
             "lambda": trial.suggest_float('lambda', 1, 20),
             "alpha": trial.suggest_float('alpha', 0, 20),
             "colsample_bytree": trial.suggest_float("colsample_bytree", 0, 1),
             "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0, 1),
             "colsample_bynode": trial.suggest_float("colsample_bynode", 0, 1),
             "verbosity": 0}

    n_trees = trial.suggest_int('ntrees', 10, 3000)
    results = {}
    reg = xgb.train(param, dtrain, n_trees, evals=[
                    (dval, 'val')], evals_result=results, early_stopping_rounds=10)
    loss = min(results['val']['mae'])
    trial.set_user_attr('best_ntree', reg.best_ntree_limit)
    return loss


if __name__ == '__main__':
    study = optuna.create_study(
        study_name="xgboost", storage="sqlite:///trials.db",
        load_if_exists=True)
    study.optimize(objective, n_trials=2000)
