import numpy as np
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import ast
import random
from datetime import datetime, timedelta

def catnum(directory):
    pattern = re.compile(r'^Cat(\d+)\.csv$')
    numbers = []
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            numbers.append(int(match.group(1)))
    return max(numbers)

def read_team(teamname, Saison):
    catdir = teamname + '/Catalogs_' + teamname + '/'
    cat = teamname + '/Catalogs_' + teamname + '/Cat' + f'{catnum(catdir):03d}' + '.csv'
    df = pd.read_csv(cat)
    Settings = pd.read_csv(teamname + '/Settings_' + teamname + '_' + Saison + '.csv')
    if os.path.isfile(teamname + '/Plan_' + teamname + '_' + Saison + '.csv'):
        Plan = pd.read_csv(teamname + '/Plan_' + teamname + '_' + Saison + '.csv')
        Plan['date'] = pd.to_datetime(Plan['date'])
    else:
        Plan = None
    return df, Settings, Plan


def numbergen(trainings, length, take, before):
    gap = length * 3 // 4
    result = np.zeros((trainings, take), dtype=int)
    for m in range(trainings):
        if len(before) > gap:
            before = before[len(before) - gap:]
        list = [x for x in range(length) if x not in before]
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
    weekdays = settings["weekdays"].apply(ast.literal_eval)[0]
    days = get_dates(year, weekdays)
    takes = settings["takes"].apply(ast.literal_eval)[0]
    groupnames = ['Einlaufen', 'Technik', 'Spielfähigkeit', 'Anderes']
    groups = {n: df[df['Kategorie'] == n].reset_index(drop=True) for n in groupnames}
    grouplens  = [len(groups[k]) for k in groups.keys()]
    befores_dict = {'Einlaufen': [], 'Technik':[], 'Spielfähigkeit': [], 'Anderes': []}
    if plan is not None:
        before_plan = plan[plan['date'] < date].copy()
        before_plan['selection'] = before_plan['selection'].apply(ast.literal_eval)
        before_plan['category'] = before_plan['category'].apply(ast.literal_eval)
        for _, t in before_plan.iterrows():
            for sel, ca in zip(t['selection'], t['category']):
                cat = t['catalog']
                dfcur = pd.read_csv(cat)
                group = dfcur[dfcur['Kategorie'] == ca].reset_index(drop=True)
                befores_dict[ca].append(group.index[group['Name'] == sel].tolist()[0])
        before_plan = before_plan.values.tolist()
    else: before_plan = []
    befores = list(befores_dict.values())
    tododays = [day for day in days if day >= date]
    trainings = len(tododays)

    res = np.zeros((trainings, 0), dtype=int)
    for length, take, before in zip(grouplens, takes, befores):
        res = np.concatenate((res, numbergen(trainings, length, take, before)), axis=1)
    catdir = teamname + '/Catalogs_' + teamname + '/'
    cat = teamname + '/Catalogs_' + teamname + '/Cat' + f'{catnum(catdir):03d}' + '.csv'
    newplan = [[datetime.strftime(b[0], "%Y-%m-%d"), b[1], b[2], b[3]] for b in before_plan]
    for n, r in enumerate(res):
        sel = []
        ca = []
        for m, ta in enumerate(takes):
            group = groups[groupnames[m]]
            for t in range(ta):
                pos = r[sum(takes[:m])+t]
                sel.append(group.loc[pos, 'Name'])
                ca.append(groupnames[m])
        newplan.append([datetime.strftime(tododays[n], "%Y-%m-%d"), sel, ca, cat])
    plan_name = teamname + '/Plan_' + teamname + '_' + year + '.csv'
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
    weeknum = {'Mon': 0, 'Tue': 1, 'Wed': 2, 'Thu': 3, 'Fri': 4, 'Sat': 5, 'Sun': 6}
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

def S(s):
    return s[:2] + '/' + s[2:]

def shorten(s: str, max_length: int = 40) -> str:
    if len(s) <= max_length:
        return s

    words = s.split()
    result = ''
    for word in words:
        if len(result) + len(word) + 4 > max_length:  # +4 for ' ...'
            break
        if result:
            result += ' '
        result += word

    return result + ' ...'

def plot_cross_matrix_with_groups(Team, Saison):
    plan_name = Team + '/Plan_' + Team + '_' + Saison + '.csv'
    plan = pd.read_csv(plan_name)
    selections = plan['selection'].apply(ast.literal_eval).tolist()

    catis = plan['category'].apply(ast.literal_eval).tolist()
    row_tuples = []
    categories = []
    dates = plan['date'].tolist()
    alldates = []
    for date, sels, cats in zip(dates, selections, catis):
        for sel, cat in zip(sels, cats):
            row_tuples.append((cat, shorten(sel)))
            categories.append(cat)
            alldates.append(date)

    categories = list(set(categories))
    priority = ['Einlaufen', 'Technik', 'Spielfähigkeit', 'Anderes']

    # Create sorted list with category headers inserted
    entries_by_cat = {cat: [] for cat in categories}
    for cat, skill in row_tuples:
        entries_by_cat[cat].append(shorten(skill))

    index_tuples = []
    for cat in sorted(categories, key=lambda c: priority.index(c)):
        index_tuples.append(('__HEADER__', cat))  # Add header row
        for skill in sorted(set(entries_by_cat[cat]), key=lambda x: x.lower()):
            index_tuples.append((cat, skill))

    index = pd.MultiIndex.from_tuples(index_tuples, names=["Category", "Skill"])


    # Create empty DataFrame
    df = pd.DataFrame(0, index=index, columns=dates)

    # Set positions of red "x"
    for rt, d in zip(row_tuples, alldates):
        df.loc[rt, d] = 1

    fig, ax = plt.subplots(figsize=(18, 10))

    # Plot empty grid
    ax.imshow(np.zeros(df.shape), cmap="Greys", vmin=0, vmax=1)

    # Draw grid lines
    for x in range(len(dates) + 1):
        ax.axvline(x - 0.5, color='black', linewidth=1)
    for y in range(len(df) + 1):
        ax.axhline(y - 0.5, color='black', linewidth=1)

    # Draw category group lines
    category_boundaries = []
    last_category = None
    for i, (cat, _) in enumerate(df.index):
        if cat != last_category:
            category_boundaries.append(i)
            last_category = cat
    category_boundaries.append(len(df))  # Add bottom line

    for y in category_boundaries:
        ax.axhline(y - 0.5, color='black', linewidth=2)

    # Draw content
    for y, (cat, skill) in enumerate(df.index):
        for x, col in enumerate(dates):
            if cat == '__HEADER__':
                ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color=(54/255, 98/255, 130/255)))
            elif df.iloc[y, x] == 1:
                ax.text(x, y, "x", ha='center', va='center', color='red', fontsize=14, weight='bold')

    # Set ticks
    ax.set_xticks(np.arange(len(dates)))
    ax.set_xticklabels(dates, rotation=45, ha='right')
    ax.set_yticks(np.arange(len(df)))

    # Set y-tick labels, skipping labels for header rows
    yticklabels = []
    for cat, skill in df.index:
        if cat == '__HEADER__':
            yticklabels.append("")  # no label for blue header
        else:
            yticklabels.append(skill)
    ax.set_yticklabels(yticklabels)


    # Draw category names inside blue header rows
    for y, (cat, skill) in enumerate(df.index):
        if cat == '__HEADER__':
            ax.text(-1, y, skill, va='center', ha='right', fontsize=14, fontweight='bold', color=(54/255, 98/255, 130/255))


    # Clean up
    ax.set_xlim(-0.5, len(dates) - 0.5)
    ax.set_ylim(len(df) - 0.5, -0.5)
    ax.set_title(Team + '  -  ' + S(Saison), size=24, fontweight='bold', color=(233/255, 138/255, 6/255))
    ax.tick_params(left=False, bottom=False)
    ax.spines[:].set_visible(False)

    plt.tight_layout()
    return fig




if __name__ == '__main__':
    # result = Make_plan(50, [16, 8,6,6], [2,1,0,0], [np.array([]),np.array([])])
    # copy = np.unique(result, axis=0)
    # count = 1
    # while len(copy) < len(result):
    #     result = Make_plan(50, [16, 8,6,6], [2,1,0,0], [np.array([]),np.array([])])
    #     copy = np.unique(result, axis=0)
    #     count += 1
    #     if count == 1000:
    #         break
    # print(result, count)

    make_plan('2526HR', datetime.strptime('2025-05-11', "%Y-%m-%d"), 'U13A')

