from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
from tqdm import tqdm

def csv_save(data: dict):
    # save csv file
    in_path = data["path_in"]
    ex_path = data["path_out"]
    event_data = event_accumulator.EventAccumulator(in_path)  # a python interface for loading Event data
    event_data.Reload()  # synchronously loads all of the data written so far b
    # print(event_data.Tags())  # print all tags
    keys = event_data.scalars.Keys()  # get all tags,save in a list
    print("in_path", in_path)
    # print(keys)
    df = pd.DataFrame(columns=keys[1:])  # my first column is training loss per iteration, so I abandon it
    for key in tqdm(keys):
        # print(key)
        df[key] = pd.DataFrame(event_data.Scalars(key)).value
    df.to_csv(ex_path)
    print("Tensorboard data exported successfully")