import numpy as np
import pandas as pd
import math
import re
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import ast
import random
from datetime import datetime, timedelta
import streamlit as st
from github import Github
from io import StringIO, BytesIO
import csv
from icalendar import Calendar
import googlemaps
import matplotlib as mpl

g = Github(st.secrets["github_token"])
repo = g.get_repo("Tim171717/Trainingsplaner")

weeknum = {'Montag': 0, 'Dienstag': 1, 'Mittwoch': 2, 'Donnerstag': 3, 'Freitag': 4, 'Samstag': 5, 'Sonntag': 6}
weekday_map = {"Mon": "Mo", "Tue": "Di", "Wed": "Mi", "Thu": "Do", "Fri": "Fr", "Sat": "Sa", "Sun": "So"}
weekdays_de = ['Mo', 'Di', 'Mi', 'Do', 'Fr', 'Sa', 'So']

def catnum(directory):
    pattern = re.compile(r'^Cat(\d+)\.csv$')
    numbers = []
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            numbers.append(int(match.group(1)))
    return max(numbers)

def read_team(teamname, Saison, ids=False):
    catdir = teamname + '/Catalogs_' + teamname + '/'
    cat = catdir + 'Cat' + f'{catnum(catdir):03d}' + '.csv'
    df = pd.read_csv(cat)
    Settings = pd.read_csv(teamname + '/Settings_' + teamname + '/Settings_' + teamname + '_' + Saison + '.csv')
    plan_id = teamname + '/Plans_' + teamname + '/Plan_' + teamname + '_' + Saison + '.csv'
    if os.path.isfile(plan_id):
        Plan = pd.read_csv(plan_id)
        Plan['date'] = pd.to_datetime(Plan['date'])
    else:
        Plan = None
    if ids:
        return df, Settings, Plan, cat, plan_id
    else:
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


def change_training(date, selection, categories, plan, cat_id):
    file = repo.get_contents(plan)
    df = pd.read_csv(plan)
    df['date'] = pd.to_datetime(df['date'])
    df.loc[df['date'] == date, 'selection'] = str(selection)
    df.loc[df['date'] == date, 'category'] = str(categories)
    df.loc[df['date'] == date, 'catalog'] = cat_id
    df.to_csv(plan, index=False)
    buffer = StringIO()
    df.to_csv(buffer, index=False, header=True)
    updated_content = buffer.getvalue()

    repo.update_file(
        path=plan,
        message=f"Updated {plan} from Streamlit app",
        content=updated_content,
        sha=file.sha,
        branch="main"
    )

def change_settings(Team, Saison, newweekdays, newtakes):
    newset = {'weekdays': newweekdays, 'takes': newtakes}
    setname = Team + '/Settings_' + Team + '/Settings_' + Team + '_' + Saison + '.csv'
    df = pd.DataFrame([newset])
    df.to_csv(setname, index=False)
    buffer = StringIO()
    df.to_csv(buffer, index=False, header=True)
    updated_content = buffer.getvalue()

    file = repo.get_contents(setname)
    repo.update_file(
        path=setname,
        message=f"Updated {setname} from Streamlit app",
        content=updated_content,
        sha=file.sha,
        branch="main"
    )


def make_plan(saison, date, teamname, exists=True):
    df, settings, plan = read_team(teamname, saison)
    weekdays = settings["weekdays"].apply(ast.literal_eval)[0]
    days = get_dates(saison, weekdays)
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
                dfcur = df.copy()
                group = dfcur[dfcur['Kategorie'] == ca].reset_index(drop=True)
                mask = group['Name'] == sel
                if mask.any():
                    befores_dict[ca].append(group.index[group['Name'] == sel].tolist()[0])
        before_plan = before_plan.values.tolist()
    else: before_plan = []
    befores = list(befores_dict.values())
    tododays = [day for day in days if day >= date]
    trainings = len(tododays)

    n = 0
    while n < 1000:
        res = np.zeros((trainings, 0), dtype=int)
        for length, take, before in zip(grouplens, takes, befores):
            res = np.concatenate((res, numbergen(trainings, length, take, before)), axis=1)
        copy = np.unique(res, axis = 0)
        if len(copy) == len(res):
            break
        n += 1

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
    plan_name = teamname + '/Plans_' + teamname + '/Plan_' + teamname + '_' + saison + '.csv'
    pf = pd.DataFrame(newplan, columns=['date', 'selection', 'category', 'catalog'])
    pf.to_csv(plan_name, index=False)
    buffer = StringIO()
    pf.to_csv(buffer, index=False, header=True)
    updated_content = buffer.getvalue()
    if exists:
        file = repo.get_contents(plan_name)
        repo.update_file(
            path=plan_name,
            message=f"Updated {plan_name} from Streamlit app",
            content=updated_content,
            sha=file.sha,
            branch="main"
        )
    else:
        repo.create_file(
            path=plan_name,
            message=f"Created {plan_name} from Streamlit app",
            content=updated_content,
            branch="main"
        )

def get_dates(saison, weekdays):
    df = pd.read_csv('Saisoninfos/' + saison + '_info.csv')
    main_start = df.iloc[0]["start"]
    main_end = df.iloc[0]["end"]

    all_dates = pd.date_range(main_start, main_end, freq='D')
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

def nextdate(saison, date, weekdays):
    dates = get_dates(saison, weekdays)
    dateday = date.replace(hour=0, minute=0, second=0, microsecond=0)
    
    future = [d for d in dates if d >= dateday]
    return min(future) if future else dates[-1]

def S(s):
    return s[:2] + '/' + s[2:4] + ' ' + s[4:]

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

def plot_plan(Team, Saison):
    plan_name = Team + '/Plans_' + Team + '/Plan_' + Team + '_' + Saison + '.csv'
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

def get_Gumb(excel_file, Saisons, weekdays=['Mittwoch', 'Freitag'],
             locations=['Goldau', 'Brunnen'], Zeiten=[['17:30', '19:00'], ['17:30', '19:00']]):
    df = pd.read_excel(excel_file, engine='openpyxl').iloc[:-1]
    dates = []
    for s in Saisons:
        dates += get_dates(s, weekdays)
    dates = [d for d in dates if d > datetime.now()]
    for d in dates:
        for n, weekday in enumerate(weekdays):
            if d.weekday() == weeknum[weekday]:
                location = locations[n]
                Zeit = Zeiten[n]
        df.loc[len(df)] = ['Training ' + location, 'Training', location, d.strftime('%d.%m.%Y'), Zeit[0], Zeit[1], '']
    output = StringIO()
    fieldnames = df.columns.tolist()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(df.to_dict(orient="records"))
    return output.getvalue()

def parse_summary(summary, my_club="HSG Mythen"):
    parts = summary.split(" - ")
    if len(parts) < 3:
        return "Unknown", None
    gameid = parts[0]
    home_team = parts[1]
    away_team = parts[2]
    if home_team[-1] == '*':
        home_team = home_team[:-2]
    if away_team[-1] == '*':
        away_team = away_team[:-2]
    return [gameid, home_team[:len(my_club)] == my_club, home_team, away_team]

def get_traveltime(arena, startpoint):
    API_KEY = st.secrets["google_apikey"]
    gmaps = googlemaps.Client(key=API_KEY)

    origins = [startpoint]
    destinations = [arena]

    result = gmaps.distance_matrix(origins, destinations, mode='driving')

    duration = result['rows'][0]['elements'][0]['duration']['text']

    hours = 0
    minutes = 0
    hour_match = re.search(r'(\d+)\s*hour', duration)
    min_match = re.search(r'(\d+)\s*min', duration)
    if hour_match:
        hours = int(hour_match.group(1))
    if min_match:
        minutes = int(min_match.group(1))
    td = timedelta(hours=hours, minutes=minutes)

    total_minutes = td.total_seconds() / 60
    rounded_minutes = math.ceil((total_minutes - 10) / 15) * 15
    return timedelta(minutes=rounded_minutes)

def mergecells(table, ix0, ix1):
    ix0, ix1 = np.asarray(ix0), np.asarray(ix1)
    d = ix1 - ix0
    if not (0 in d and 1 in np.abs(d)):
        raise ValueError("ix0 and ix1 should be the indices of adjacent cells. ix0: %s, ix1: %s" % (ix0, ix1))

    if d[0] == -1:
        edges = ('BRL', 'TRL')
    elif d[0] == 1:
        edges = ('TRL', 'BRL')
    elif d[1] == -1:
        edges = ('BTR', 'BTL')
    else:
        edges = ('BTL', 'BTR')

    # hide the merged edges
    for ix, e in zip((ix0, ix1), edges):
        table[ix[0], ix[1]].visible_edges = e

    txts = [table[ix[0], ix[1]].get_text() for ix in (ix0, ix1)]
    tpos = [np.array(t.get_position()) for t in txts]

    # center the text of the 0th cell between the two merged cells
    trans = (tpos[1] - tpos[0]) / 2
    if trans[0] > 0 and txts[0].get_ha() == 'right':
        # reduce the transform distance in order to center the text
        trans[0] /= 2
    elif trans[0] < 0 and txts[0].get_ha() == 'right':
        # increase the transform distance...
        trans[0] *= 2

    txts[0].set_transform(mpl.transforms.Affine2D().translate(*trans))

    # hide the text in the 1st cell
    txts[1].set_visible(False)

def format_german_date(dt):
    weekday = weekdays_de[dt.weekday()]  # Monday=0, Sunday=6
    return f'{weekday} | {dt.strftime("%d.%m.%Y")}'

def get_Matches(cal, excel_file='Gumb_Vorlage.xlsx', team='U13_A', startpoint='Goldau Berufsbildungszentrum',
                show_teamname=True, printversion=False):
    df = pd.read_excel(excel_file, engine='openpyxl').iloc[:-1]

    spiele = []
    for component in cal.walk():
        if component.name == "VEVENT":
            summary = component.get('summary')
            start = component.get('dtstart').dt
            end = component.get('dtend').dt
            location = component.get('location')
            if location == 'Einsiedeln Brühl': location = 'Einsiedeln Brüel'
            curr = parse_summary(summary) + [start, end, location]
            spiele.append(curr)

    spielgroups = list(set([s[0] for s in spiele]))
    newspiele = []
    for group in spielgroups:
        games = [s for s in spiele if s[0] == group]
        if len(games) == 1:
            newspiele.append(games[1:] + [True])
        else:
            dates = list(set([g[4].date() for g in games]))
            if len(dates) == 1:
                starttime = min([g[4] for g in games])
                endtime = max([g[5] for g in games])
                location = games[0][6].split()[0]
                home = location in ['Goldau', 'Brunnen', 'Schwyz']
                newspiele.append([home, location, None, starttime, endtime, games[0][6], False])
            else:
                for g in games:
                    newspiele.append(g[1:] + [True])
    newspiele = sorted(newspiele, key=lambda x: x[4])
    data = []
    home = []
    turnier = []
    for s in newspiele:
        if s[-1]:
            turnier.append(False)
            if s[0]:
                df.loc[len(df)] = [(team + ' ') * show_teamname + 'Heimspiel gegen ' + s[2], 'Heimspiel', s[5],
                                   s[3].strftime('%d.%m.%Y'), (s[3] - timedelta(hours=1)).strftime('%H:%M'),
                                   s[4].strftime('%H:%M'), 'Anpfiff: ' + s[3].strftime('%H:%M')]
                data.append([format_german_date(s[3]), s[3].strftime('%H:%M'),
                             (s[3] - timedelta(hours=1)).strftime('%H:%M'), s[1], s[2], s[5], "", ""])
                home.append(True)
            else:
                traveltime = get_traveltime(s[5], startpoint)
                df.loc[len(df)] = [(team + ' ') * show_teamname + 'Auswärtsspiel gegen ' + s[1], 'Auswärtsspiel', s[5],
                                   s[3].strftime('%d.%m.%Y'), (s[3] - timedelta(hours=1) - traveltime).strftime('%H:%M'),
                                   s[4].strftime('%H:%M'), 'Anpfiff: ' + s[3].strftime('%H:%M')]
                data.append([format_german_date(s[3]), s[3].strftime('%H:%M'),
                             (s[3] - timedelta(hours=1) - traveltime).strftime('%H:%M'), s[1], s[2], s[5], "", ""])
                home.append(False)
        else:
            turnier.append(True)
            if s[0]:
                df.loc[len(df)] = [(team + ' ') * show_teamname + 'Heimturnier in ' + s[1], 'Heimturnier', s[5],
                                   s[3].strftime('%d.%m.%Y'), (s[3] - timedelta(hours=1)).strftime('%H:%M'),
                                   s[4].strftime('%H:%M'), '']
                data.append([format_german_date(s[3]), s[3].strftime('%H:%M'),
                             (s[3] - timedelta(hours=1)).strftime('%H:%M'), 'Heimturnier', '', s[5], "", ""])
                home.append(True)
            else:
                traveltime = get_traveltime(s[5], startpoint)
                df.loc[len(df)] = [(team + ' ') * show_teamname + 'Auswärtsturnier in ' + s[1], 'Auswärtsturnier', s[5],
                                   s[3].strftime('%d.%m.%Y'), (s[3] - timedelta(hours=1) - traveltime).strftime('%H:%M'),
                                   s[4].strftime('%H:%M'), '']
                data.append([format_german_date(s[3]), s[3].strftime('%H:%M'),
                             (s[3] - timedelta(hours=1) - traveltime).strftime('%H:%M'), 'Auswärtsturnier', '', s[5], "", ""])
                home.append(False)

    output = StringIO()
    fieldnames = df.columns.tolist()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(df.to_dict(orient="records"))
    return output.getvalue(), Fahrerliste(data, home, turnier, dotwo=printversion), #Baspo_output(newspiele)

def Fahrerliste(data, home, Turnier, dotwo=False):
    if False in Turnier:
        columns = ["Datum", "Zeit", "Besammlung/\nAbfahrt", "Heim", "Gast", "Halle",
                   "Fahren\n(Anzahl Plätze\nohne Fahrer)", "Trikots\nwaschen"]
        df = pd.DataFrame(data, columns=columns)
        col_widths = {
            0: 0.09,  # Datum
            1: 0.085,  # Zeit
            2: 0.085,  # Besammlung
            3: 0.17,  # Heim
            4: 0.17,  # Gast
            5: 0.22,  # Halle
            6: 0.1,  # Fahren
            7: 0.08  # Trikots
        }
        fahrenloc = 6
    else:
        columns = ["Datum", "Zeit", "Besammlung/\nAbfahrt", 'Turnier', "Halle", "Fahren\n(Anzahl Plätze\nohne Fahrer)",
                   "Trikots\nwaschen"]
        df = pd.DataFrame([[d[0], d[1], d[2], d[3], d[5], d[6], d[7]] for d in data], columns=columns)
        col_widths = {
            0: 0.09,  # Datum
            1: 0.085,  # Zeit
            2: 0.085,  # Besammlung
            3: 0.16,  # Turnier
            4: 0.26,  # Halle
            5: 0.1,  # Fahren
            6: 0.08  # Trikots
        }
        fahrenloc = 5

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(10)

    table.scale(1, 1.5)
    for (row, col), cell in table.get_celld().items():
        if col in col_widths:
            cell.set_width(col_widths[col])
        if row == 0:
            cell.set_fontsize(10)
            cell.set_text_props(weight='bold')
            cell.set_facecolor("#f2f2f2")
            cell.set_height(0.14)
        if col == fahrenloc and row > 0 and home[row - 1]:
            cell.set_facecolor("black")
            cell.set_text_props(color="white")
        # if not home[row - 1]:
        #     cell.set_height(0.14)
        if row > 0 and not Turnier[row - 1]:
            if col == 3 and row > 0 and home[row - 1]:
                cell.set_text_props(weight='bold')
            if col == 4 and row > 0 and not home[row - 1]:
                cell.set_text_props(weight='bold')
        if row > 0 and col == 3 and False in Turnier:
            if Turnier[row - 1]:
                mergecells(table, (row, 3), (row, 4))
                cell.set_text_props(weight='bold')

    # plt.title("Name:", fontsize=14, weight='bold', pad=-20, loc='left')
    img_buf = BytesIO()
    plt.tight_layout()
    plt.savefig(img_buf, format='png', dpi=600)
    img_buf.seek(0)

    img_buf.seek(0)
    img = mpimg.imread(img_buf)
    if dotwo:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.27, 11.69))
        ax2.axis('off')
        ax2.imshow(img)
    else:
        fig, ax1 = plt.subplots(figsize=(11.69, 8.27))
    ax1.axis('off')
    ax1.imshow(img)
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='pdf', dpi=600)
    buf.seek(0)
    return buf.getvalue()

def Baspo_output(spiele):
    dfba = pd.read_csv('Baspo_vorlage.csv', sep=';')
    for s in spiele:
        dfba.loc[len(dfba)] = ['Wettkampf',
                               s[3].strftime("%d.%m.%Y"),
                               s[3].strftime("%H:%M"),
                               int((s[4]-s[3]).total_seconds()//60),
                               s[5],
                               '']
    output = StringIO()
    fieldnames = dfba.columns.tolist()
    writer = csv.DictWriter(output, fieldnames=fieldnames, delimiter=';')
    writer.writeheader()
    writer.writerows(dfba.to_dict(orient="records"))
    return output.getvalue()


if __name__ == '__main__':
    # make_plan('2526HR', datetime.strptime('2025-05-11', "%Y-%m-%d"), 'U13A')

    # csv_string = get_Gumb('D:/timlf/Tim Daten/Downloads/U13 Mythen Shooters.xlsx', ['2526HR', '2526RR'])
    # csvname = 'Gumb_output.csv'
    # with open(csvname, "w", newline='', encoding='utf-8') as f:
    #     f.write(csv_string)

    with open('D:/timlf/Tim Daten/Downloads/spielplan-hsg-mythen-shooters-1.ics', 'rb') as f:
        cal = Calendar.from_ical(f.read())
    _, _, csvdata = get_Matches(cal)
    csvname = 'baspo_output.csv'
    with open(csvname, "w", newline='', encoding='utf-8') as f:
        f.write(csvdata)