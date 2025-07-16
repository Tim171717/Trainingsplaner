import streamlit as st
import pandas as pd
import numpy as np
import ast
from icalendar import Calendar
import io
from datetime import datetime, timedelta, time
import locale

from Trainingsplan_maker2 import *

accounts = dict(st.secrets["teams"])
Saisons = {'2526HR': 0, '2526RR': 1, '2627HR': 2, '2627RR': 3}
ubs = ['Einlaufen', 'Technik', 'Spielfähigkeit', 'Anderes']
Hallen = {'Goldau': 'Goldau Berufsbildungszentrum',
          'Brunnen': 'Sporthalle Brunnen',
          'Muotathal': 'Sporthalle Muotathal',
          'Schwyz': 'Schwyz Schwyz'}
if st.session_state.get('Saison', None) is None:
    st.session_state['Saison'] = '2526HR'

# --- Title ---
st.title("Trainingsplaner")

# --- Profil selection ---
teams = accounts.keys()
with st.form(key="login_form"):
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        profil = st.selectbox('Team', teams, label_visibility='collapsed')

    with col2:
        password = st.text_input('Password', type="password", label_visibility='collapsed', icon='🗝️')

    with col3:
        login = st.form_submit_button('Login')
    with col4:
        if accounts[profil] == password and login:
            st.session_state['loggedin'] = True
            st.session_state['Team'] = profil
        elif login:
            st.session_state['loggedin'] = False
            st.error('Falsches Passwort')


if st.session_state.get('loggedin', False):
    Team = st.session_state['Team']
    Saison = st.session_state['Saison']
    catalog, settings, plan, cat_id, plan_id = read_team(Team, Saison, ids=True)
    weekdays = settings["weekdays"].apply(ast.literal_eval)[0]
    takes = settings["takes"].apply(ast.literal_eval)[0]
    st.session_state['weekdays'] = weekdays
    ndate = nextdate(Saison, datetime.today(), weekdays)
    if st.session_state.get('selected_date', None) is None:
        st.session_state['selected_date'] = ndate
    if plan is None:
        startdate = get_dates(Saison, weekdays)[0]
        make_plan(Saison, startdate, Team, exists=False)
        plan = read_team(Team, Saison)[2]

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Nächstes Training", "Plan", "Übungen", "Gumb", "Einstellungen"])

    with tab1:
        try:
            locale.setlocale(locale.LC_TIME, 'de_DE.UTF-8')
        except:
            locale.setlocale(locale.LC_TIME, '')

        tab1a, tab2a = st.columns([3, 1])
        with tab1a:
            dates = get_dates(Saison, weekdays)
            selected_date = st.selectbox("📅 Wähle ein Datum", dates, index=dates.index(ndate),
                                         format_func=lambda x: x.strftime('%A, %d. %B %Y'), label_visibility='collapsed')
            if selected_date != st.session_state['selected_date']:
                st.session_state['selected_date'] = selected_date
                st.session_state.newexes = None
                st.session_state.newtakes = None


        row = plan[plan['date'] == selected_date.strftime("%Y-%m-%d")].copy().reset_index(drop=True)
        exercises = row['selection'].apply(ast.literal_eval)[0]
        categories = row['category'].apply(ast.literal_eval)[0]
        thistakes = [categories.count(cat) for cat in ubs]
        categories = [g for g in ubs if g in categories]
        cat = row['catalog'][0]
        catdf = pd.read_csv(cat)

        edit = False
        if not st.session_state.get('show_edit', False):
            with tab2a:
                edit = st.button('Bearbeiten')

        if edit:
            st.session_state.show_edit = True
            st.rerun()

        # Edit interface
        if st.session_state.get('show_edit', False):
            if st.session_state.get('newexes', None) is None:
                st.session_state.newexes = exercises.copy()
            newexes = st.session_state.newexes.copy()
            if st.session_state.get('newtakes', None) is None:
                st.session_state.newtakes = thistakes
            newtakes = st.session_state.newtakes
            if selected_date >= datetime.today().replace(hour=0, minute=0, second=0, microsecond=0):
                with tab2a:
                    if st.button('Remix'):
                        make_plan(Saison, selected_date, Team)
                        st.session_state.newtakes = None
                        st.session_state.newexes = None
                        st.rerun()

            keinebox = """
            <style>
            .custom-box {
                display: inline-block;
                background-color: #f0f2f6;
                color: #262730;
                font-family: "Source Sans Pro", sans-serif;
                font-size: 1rem;
                padding: 0.375rem 0.75rem;
                border-radius: 0.5rem;
                border: none;
                width: 165px;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }
            </style>

            <div class="custom-box">Keine</div>
            """
            otexes = []
            i = 0
            for size in newtakes:
                group = newexes[i:i+size]
                otexes.append(group)
                i += size

            for n, (id, t, otex) in enumerate(zip(ubs, newtakes, otexes)):
                options = catalog[catalog['Kategorie'] == id]['Name'].values.tolist()

                st.markdown(f"#### {id}")
                cols = st.columns(4)
                if t == 0:
                    with cols[0]:
                        st.markdown(keinebox, unsafe_allow_html=True)
                else:
                    for col_idx in range(t):
                        if col_idx < len(otex):
                            exid = options.index(otex[col_idx])
                        else: exid = 0
                        with cols[col_idx]:
                            otexes[n][col_idx] = st.selectbox(f"Auswahl_{n}", options, key=f"Selection{id}{col_idx}",
                                                              label_visibility='collapsed',index=exid)


                with cols[3]:
                    col1, col2 = st.columns([1, 1])
                    if t > 0:
                        with col1:
                            if st.button("➖", key=f"remove{n}"):
                                st.session_state.newtakes[n] -= 1
                                otexes[n].pop()
                                st.session_state.newexes = [item for group in otexes for item in group]
                                st.rerun()
                    if t < 3:
                        with col2:
                            if st.button("➕", key=f"add{n}"):
                                st.session_state.newtakes[n] += 1
                                otexes[n].append(options[0])
                                st.session_state.newexes = [item for group in otexes for item in group]
                                st.rerun()
                if t < 3:
                    with cols[t]:
                        st.markdown(' ')

                st.session_state.newexes = [item for group in otexes for item in group]


            if st.button("Speichern", key="speichern"):
                newexes = [item for group in otexes for item in group]
                newcates = [catalog[catalog['Name'] == ex]['Kategorie'].values[0] for ex in newexes]
                change_training(selected_date, newexes, newcates, plan_id, cat_id)
                afterdate = max(selected_date + timedelta(days=1),
                                datetime.today().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1))
                make_plan(Saison, afterdate, Team)
                st.session_state.newexes = None
                st.session_state.newtakes = None
                st.session_state.show_edit = False
                st.rerun()

            if st.button("Abbrechen", key="abbrechen"):
                st.session_state.newexes = None
                st.session_state.newtakes = None
                st.session_state.show_edit = False
                st.rerun()


        else:
            blocks = {c: {'title': c, 'rows': []} for c in categories}
            for ex in exercises:
                info = catdf[catdf['Name'] == ex].to_dict(orient='records')[0]
                for key in info:
                    if info[key] is np.nan:
                        info[key] = "&nbsp;"
                blocks[info['Kategorie']]['rows'].append((info['Zeit'], info['Name'], info['Ziel'], info['Beschreibung']))
            blocks = list(blocks.values())

            table_style1 = """
            <style>
            .table1 {
                width: 705px;
                border-collapse: collapse;
                background-color: #ffffff;
                color: #31333f;
                margin-bottom: 25px;
                table-layout: fixed;
            }
            .table1 th, .table1 td {
                padding: 8px;
                vertical-align: top;
                word-wrap: break-word;
            }
            .table1 th:nth-child(1), .table1 td:nth-child(1) { width: 10%; }
            .table1 th:nth-child(2), .table1 td:nth-child(2) { width: 20%; }
            .table1 th:nth-child(3), .table1 td:nth-child(3) { width: 20%; }
            .table1 th:nth-child(4), .table1 td:nth-child(4) { width: 50%; }
            .table1 th {
                background-color: #f2f2f2;
                text-align: left;
            }
            .block-title {
                font-weight: bold;
                font-size: 18px;
                margin-top: 30px;
            }
            </style>
            """

            # Build HTML
            html_content1 = table_style1
            for block in blocks:
                html_content1 += f"<div class='block-title'>{block['title']}</div>"
                html_content1 += "<table class='table1'>"
                html_content1 += "<tr><th>Zeit</th><th>Name</th><th>Ziel</th><th>Beschreibung</th></tr>"
                for zeit, inhalt, ziel, beschr in block["rows"]:
                    html_content1 += f"<tr><td>{zeit}</td><td>{inhalt}</td><td>{ziel}</td><td>{beschr}</td></tr>"
                html_content1 += "</table>"

            st.markdown(html_content1, unsafe_allow_html=True)


    with tab2:
        plan_fit = plot_plan(Team, Saison)
        st.pyplot(plan_fit)
        buf = io.BytesIO()
        plan_fit.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        plan_name = 'Plan_' + Team + '_' + Saison + '.png'

        st.download_button(
            label="Download Plot as PNG",
            data=buf,
            file_name=plan_name,
            mime="image/png"
        )

    with tab3:
        if st.session_state.get('createx', False):
            Kategorie = st.selectbox('Kategorie', ubs)
            name = st.text_input('Name')
            Zeit = str(st.number_input('Zeit', min_value=0, max_value=60, step=1)) + 'min'
            Ziel = st.text_input('Ziel')
            Beschreibung = st.text_input('Beschreibung')
            Anmerkungen = st.text_input('Anmerkung')

            if st.button("Speichern"):
                catalog.loc[len(catalog)] = [name, Zeit, Ziel, Beschreibung, Anmerkungen, Kategorie]
                catdir = Team + '/Catalogs_' + Team + '/'
                newcatid = catdir + 'Cat' + f'{catnum(catdir)+1:03d}' + '.csv'
                catalog.to_csv(newcatid, index=False, header=True)
                st.session_state.createx = False
                st.rerun()

            if st.button("Verwerfen"):
                st.session_state.createx = False
                st.rerun()

        elif st.session_state.get('editex', False):
            tab1d, tab2d = st.columns(2)
            with tab1d:
                Kategorie = st.selectbox('Kategorie', ubs)
            with tab2d:
                exes = catalog[catalog['Kategorie'] == Kategorie]
                Ubung = st.selectbox('Übung', [e[0] for e in exes.values])
            infos = exes[exes['Name'] == Ubung].values[0]
            infos = [x if x is not np.nan else '' for x in infos]
            Zeit = str(st.number_input('Zeit', min_value=0, max_value=60, step=1, value=int(infos[1][:-3]))) + 'min'
            Ziel = st.text_input('Ziel', value=infos[2])
            Beschreibung = st.text_input('Beschreibung', value=infos[3])
            Anmerkungen = st.text_input('Anmerkung', value=infos[4])

            if st.button("Speichern"):
                catalog[catalog['Name'] == Ubung] = [Ubung, Zeit, Ziel, Beschreibung, Anmerkungen, Kategorie]
                catdir = Team + '/Catalogs_' + Team + '/'
                newcatid = catdir + 'Cat' + f'{catnum(catdir):03d}' + '.csv'
                catalog.to_csv(newcatid, index=False, header=True)
                st.session_state.editex = False
                st.rerun()

            if st.button("Verwerfen"):
                st.session_state.editex = False
                st.rerun()

        else:
            tab1c, tab2c = st.columns(2)
            with tab1c:
                if st.button('Neue Übung'):
                    st.session_state.createx = True
                    st.rerun()
            with tab2c:
                if st.button('Übung bearbeiten'):
                    st.session_state.editex = True
                    st.rerun()

            blocks = {c: {'title': c, 'rows': []} for c in ubs}
            for c in ubs:
                info = catalog[catalog['Kategorie'] == c].to_dict(orient='records')
                for m in range(len(info)):
                    for key in info[m].keys():
                        if info[m][key] is np.nan:
                            info[m][key] = "&nbsp;"
                    blocks[c]['rows'].append(tuple(info[m].values()))
            blocks = list(blocks.values())

            table_style2 = """
            <style>
            .table2 {
                width: 705px;
                border-collapse: collapse;
                background-color: #ffffff;
                color: #31333f;
                margin-bottom: 25px;
                table-layout: fixed;
            }
            .table2 th, .table2 td {
                padding: 8px;
                vertical-align: top;
                word-wrap: break-word;
            }
            .table2 th:nth-child(1), .table2 td:nth-child(1) { width: 20%; }
            .table2 th:nth-child(2), .table2 td:nth-child(2) { width: 10%; }
            .table2 th:nth-child(3), .table2 td:nth-child(3) { width: 10%; }
            .table2 th:nth-child(4), .table2 td:nth-child(4) { width: 30%; }
            .table2 th:nth-child(5), .table2 td:nth-child(5) { width: 30%; }
            .table2 th {
                background-color: #f2f2f2;
                text-align: left;
            }
            .block-title {
                font-weight: bold;
                font-size: 18px;
                margin-top: 30px;
            }
            </style>
            """


            # Build HTML
            html_content2 = table_style2
            for block in blocks:
                html_content2 += f"<div class='block-title'>{block['title']}</div>"
                html_content2 += "<table class='table2'>"
                html_content2 += "<tr><th>Name</th><th>Zeit</th><th>Ziel</th><th>Beschreibung</th><th>Anmerkungen</th></tr>"
                for name, zeit, ziel, beschr, anm, _ in block["rows"]:
                    html_content2 += f"<tr><td>{name}</td><td>{zeit}</td><td>{ziel}</td><td>{beschr}</td><td>{anm}</td></tr>"
                html_content2 += "</table>"

            st.markdown(html_content2, unsafe_allow_html=True)


    with tab4:
        st.title('Termine für zum Gumb hochladen')
        what = st.selectbox('Gumb', ['Trainings', 'Spiele'], label_visibility='collapsed')
        st.divider()

        if what == 'Trainings':
            times = [[None, None] for w in weekdays]
            hallen = [None for w in weekdays]
            sel_saisons = st.multiselect(f"saisons", Saisons.keys(), label_visibility="collapsed")
            for n, weekday in enumerate(weekdays):
                col1, col2, col3, col4 = st.columns([20, 27, 27, 27])  # You can adjust the width ratio

                with col1:
                    st.markdown(f"**{weekday}**")  # Or dynamically use a weekday name

                with col2:
                    times[n][0] = st.time_input(f"Start_{weekday}", value=time(17, 30),
                                                label_visibility="collapsed").strftime('%H:%M')

                with col3:
                    times[n][1] = st.time_input(f"End_{weekday}", value=time(19, 0),
                                                label_visibility="collapsed").strftime('%H:%M')

                with col4:
                    hallen[n] = st.selectbox(f"Halle_{weekday}", Hallen.keys(), label_visibility="collapsed")


            if st.button("CSV erstellen"):
                csv_data = get_Gumb(
                    'Gumb_Vorlage.xlsx',
                    sel_saisons,
                    weekdays=weekdays,
                    locations=hallen,
                    Zeiten=times
                )
                st.success("CSV wurde erstellt!")

                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f'Gumb_upload.csv',
                    mime='text/csv'
                )

        else:
            treffpunkt = st.selectbox(f"Treffpunkt", Hallen.keys())
            uploaded_icsfile = st.file_uploader("Terminplan von Handball.ch hochladen", type="ics")

            if st.button("CSV erstellen"):
                cal = Calendar.from_ical(uploaded_icsfile.read())
                csv_data = get_Matches(
                    cal,
                    team=Team,
                    startpoint=Hallen[treffpunkt])
                st.success("CSV wurde erstellt!")

                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f'Gumb_upload.csv',
                    mime='text/csv'
                )




    with tab5:
        st.header("Saison")
        row1, row2 = st.columns([1,3])
        with row1:
            saiopt = {S(t): t for t in Saisons.keys()}
            saison_op = saiopt[st.selectbox('saison', list(saiopt.keys()), label_visibility='collapsed',
                                     index=Saisons[st.session_state['Saison']])]
        with row2:
            load = st.button('Laden')

        if load:
            st.session_state['Saison'] = saison_op
            st.rerun()

        st.header('Einstellungen')
        st.subheader("Trainingstage")
        allweekdays = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", 'Samstag', 'Sonntag']
        newweekdays = st.multiselect("Trainingstage", allweekdays, default=weekdays, label_visibility='collapsed')

        st.subheader("Übungen")
        row1, row2, row3, row4 = st.columns(4)
        with row1:
            t1 = int(st.selectbox('Einlaufen', [0,1,2,3,4], index=takes[0]))
        with row2:
            t2 = int(st.selectbox('Technik', [0,1,2,3,4], index=takes[1]))
        with row3:
            t3 = int(st.selectbox('Spielfähigkeit', [0,1,2,3,4], index=takes[2]))
        with row4:
            t4 = int(st.selectbox('Sonstiges', [0,1,2,3,4], index=takes[3]))

        newtakes = [t1, t2, t3, t4]

        if newweekdays != weekdays or newtakes != takes:
            if st.button('Speichern'):
                change_settings(Team, Saison, newweekdays, newtakes)
                startdate = datetime.today()
                make_plan(Saison, startdate, Team)
                st.rerun()
