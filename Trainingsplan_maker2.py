import numpy as np
import pandas as pd
import os
import re
import plotly as py
import csv
import ast
import random
from datetime import datetime, timedelta

def catnum(directory):
    pattern = re.compile(r'^cat(\d+)$')
    numbers = []
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            numbers.append(int(match.group(1)))
    return max(numbers)

def read_team(teamname, Saison):
    df = pd.read_csv(teamname + '/Catalogs_' + teamname + '/Cat' + f'{catnum:03d}' + '.csv')
    Settings = pd.read_csv(teamname + '/Settings_' + teamname + '.csv')
    if os.path.isfile(teamname + '/Plan_' + teamname + '_' + Saison + '.csv'):
        Plan = pd.read_csv(teamname + '/Plan_' + teamname + '_' + Saison + '.csv').to_dict('records')
        Plan = [{'date': datetime.strptime(t['date'], "%Y-%m-%d"),
                'selection': t['selection'], 'category': t['category'], 'catalog': t['cat_id']} for t in Plan]
    else:
        Plan = None
    return df, Settings, Plan


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


def change_training(date, selection, plan, cat_id):
    df = pd.read_csv(plan)
    df.loc[df['date'] == date, 'selection'] = selection
    df.loc[df['date'] == date, 'catalog'] = cat_id
    df.to_csv(plan, index=False)


def make_plan(year, date, teamname):
    df, settings, plan = read_team(teamname, year)
    weekdays = settings["weekdays"][0]
    days = get_dates(year, weekdays)
    takes = settings["takes"][0]
    groupnames = ['Einlaufen', 'Spielfähigkeit', 'Technik', 'Anderes']
    groups = {n: df[df['Kategorie'] == n] for n in groupnames}
    grouplens  = [len(groups[k]) for k in groups.keys()]
    befores = []
    if plan is not None:
        before_plan = [{'date': t['date'], 'selection': t['selection'],
                        'category': t['category'], 'catalog': t['cat_id']} for t in plan if t['date'] < date]
        for t in before_plan:
            for sel, ca in zip(t['selection'], t['catalog']):
                group = groups[ca]
                befores.append([group.index[group['Name'] == sel].tolist()[0]])
    else: before_plan = []
    tododays = [day for day in days if day >= date]
    trainings = len(tododays)

    res = np.zeros((trainings, 0), dtype=int)
    for length, take, before, m in zip(grouplens, takes, befores, range(len(grouplens))):
        res = np.concatenate((res, numbergen(trainings, length, take, before)), axis=1)
    cat = teamname + '/Catalogs_' + teamname + '/Cat' + f'{catnum:03d}' + '.csv'
    newplan = before_plan.copy()
    for n, r in enumerate(res):
        sel = []
        ca = []
        for m, ta in enumerate(takes):
            group = groups[groups.keys()[m]]
            for t in range(ta):
                pos = r[sum(takes[:m])+t]
                sel.append(group['Name'][pos])
                ca.append(groupnames[m])
        newplan.append({'date': tododays[n], 'selection': sel, 'category': ca, 'catalog': cat})
    plan_name = teamname + '/' + year + '_plan.csv'
    pf = pd.DataFrame(newplan, columns=['date', 'selection', 'category', 'catalog'])
    pf.to_csv(plan_name, index=False, header=True)


def Make_plan(trainings, grouplens, takes, befores):
    res = np.zeros((trainings, 0), dtype=int)
    for length, take, before, m in zip(grouplens, takes, befores, range(len(grouplens))):
        res = np.concatenate((res, numbergen(trainings, length, take, before)), axis=1)
    return res

def get_dates(year, weekdays):
    df = pd.read_csv(year + '_info.csv')
    main_start = df.iloc[0]["start"]
    main_end = df.iloc[0]["end"]

    all_dates = pd.date_range(main_start, main_end, freq='D')
    weeknum = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
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
    result = Make_plan(50, [16, 8,6,6], [2,1,0,0], [np.array([]),np.array([])])
    copy = np.unique(result, axis=0)
    count = 1
    while len(copy) < len(result):
        result = Make_plan(50, [16, 8,6,6], [2,1,0,0], [np.array([]),np.array([])])
        copy = np.unique(result, axis=0)
        count += 1
        if count == 1000:
            break
    print(result, count)

