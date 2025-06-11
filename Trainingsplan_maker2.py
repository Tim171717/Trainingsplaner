import numpy as np
import pandas as pd
import plotly as py
import csv
import ast
import random
from datetime import datetime, timedelta

def read_team(teamname, Saison):
    df = pd.read_csv(teamname + '/<UNK>bungen_' + teamname + '.csv').to_dict(orient="records")
    Settings = pd.read_csv(teamname + '/Settings' + teamname + '.csv').to_dict(orient="records")
    return df, Settings


def numbergen(trainings, length, take, before):
    before_init = before.copy()
    gap = length * 3 // 4
    result = np.zeros((trainings, take), dtype=int)
    for m in range(trainings):
        if len(before) > gap:
            before = before[take:]
        list = [x for x in range(1, length + 1) if x not in before]
        a = sorted(random.sample(list, take))
        result[m] = a
        before = np.append(before, a)

    return result

def Make_plan(trainings, grouplens, takes, befores):
    res = np.zeros((trainings, 0), dtype=int)
    for length, take, before, m in zip(grouplens, takes, befores, range(len(grouplens))):
        res = np.concatenate((res, numbergen(trainings, length, take, before)), axis=1)
    return res

def change_training(date, selection, plan, cat_id):
    df = pd.read_csv(plan)
    df.loc[df['date'] == date, 'selection'] = selection
    df.loc[df['date'] == date, 'catalog'] = cat_id
    df.to_csv(plan, index=False)

def write_plan(year, dates, selections, cat_id, teamname):
    data = []
    plan_name = teamname + '/' + year + '_plan.csv'
    for dt, selection in zip(dates, selections):
        data.append({'date': dt, 'selection': selection, 'catalog': cat_id})
    df = pd.DataFrame(data, columns=['date', 'selection', 'catalog'])
    df.to_csv(plan_name, index=False, header=True)

def make_plan(year, weekdays, teamname):
    df, settings = read_team(teamname, year)


def get_dates(year, weekdays):
    df = pd.read_csv(year + '_info.csv')
    main_start = df.iloc[0]["start"]
    main_end = df.iloc[0]["end"]

    all_dates = pd.date_range(main_start, main_end, freq='D')
    weeknum = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}
    tdays = all_dates[all_dates.weekday.isin([weeknum[w] for w in weekdays])]

    exclude_ranges = df.iloc[1:]
    mask = pd.Series([True] * len(tdays), index=tdays)

    for _, row in exclude_ranges.iterrows():
        start = row["start"]
        end = row["end"] if pd.notnull(row["end"]) else row["start"]
        mask[(mask.index >= start) & (mask.index <= end)] = False

    filtered_dates = mask[mask].index.tolist()
    dates = [datetime.strptime(d.strftime("%Y-%m-%d"), "%Y-%m-%d") for d in filtered_dates]
    return dates



if __name__ == '__main__':
    result = Make_plan(50, [16, 8], [2,1], [np.array([]),np.array([])])
    copy = np.unique(result, axis=0)
    count = 1
    while len(copy) < len(result):
        result = Make_plan(50, [16, 8], [2,1], [np.array([]),np.array([])])
        copy = np.unique(result, axis=0)
        count += 1
        if count == 1000:
            break
    print(result, count)

