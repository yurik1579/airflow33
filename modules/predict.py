import glob
import pandas as pd
import dill
import json
import os

from datetime import datetime


path = os.environ.get('PROJECT_PATH', '.')


def predict():
    list_of_files = glob.glob(f'{path}/data/models/*')
    latest_model = max(list_of_files, key=os.path.getctime)

    with open(latest_model, 'rb') as file:
        model = dill.load(file)

    list_of_files = glob.glob(f'{path}/data/test/*')
    test_list = []

    for test_file in list_of_files:
        with open(test_file) as f:
            test_list.append(json.load(f))

    df_test = pd.DataFrame.from_dict(test_list)
    y = model.predict(df_test)
    new_df = pd.DataFrame()
    new_df['id'] = df_test['id']
    new_df['price category'] = y


    new_df.to_csv(f'{path}/data/predictions/predictions_{datetime.now().strftime("%Y%m%d%H%M")}.csv',
                  index=False)


if __name__ == '__main__':
    predict()

