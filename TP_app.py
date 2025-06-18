import streamlit as st
import pandas as pd
import numpy as np
import ast
import io
from datetime import datetime
import locale

from Trainingsplan_maker2 import *

accounts = {'U13A': '1', 'U13B': '1'}
Saisons = {'2526HR': 0, '2526RR': 1}
ubs = ['Einlaufen', 'Technik', 'Spielfähigkeit', 'Anderes']
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
    catalog, settings, plan = read_team(Team, Saison)
    weekdays = settings["weekdays"].apply(ast.literal_eval)[0]
    takes = settings["takes"].apply(ast.literal_eval)[0]
    st.session_state['weekdays'] = weekdays
    ndate = nextdate(Saison, datetime.today(), weekdays)
    if plan is None:
        startdate = get_dates(Saison, weekdays)[0]
        make_plan(Saison, startdate, Team)
        plan = read_team(Saison, Team)[2]

    tab1, tab2, tab3, tab4 = st.tabs(["Nächstes Training", "Plan", "Übungen", "Einstellungen"])

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

        row = plan[plan['date'] == selected_date.strftime("%Y-%m-%d")].copy().reset_index(drop=True)
        exercises = row['selection'].apply(ast.literal_eval)[0]
        categories = row['category'].apply(ast.literal_eval)[0]
        categories = [g for g in ubs if g in categories]
        cat = row['catalog'][0]
        catdf = pd.read_csv(cat)

        with tab2a:
            edit = st.button('Bearbeiten')

        # Session state for row tracking
        if 'num_rows' not in st.session_state:
            st.session_state.num_rows = 1

        if edit:
            st.session_state.show_edit = True

        # Edit interface
        if st.session_state.get('show_edit', False):
            if st.session_state.get('newtakes', None) is None:
                st.session_state.newtakes = takes.copy()
            newtakes = st.session_state.get('newtakes')

            for n, (id, t) in enumerate(zip(ubs, newtakes)):
                options = catalog[catalog['Kategorie'] == id].reset_index(drop=True)

                st.markdown(f"#### {id}")
                cols = st.columns(5)
                if t == 0:
                    with cols[0]:
                        st.markdown("""
                            <style>
                            .custom-box {
                                display: inline-block;
                                background-color: #f0f2f6; /* light gray background */
                                color: #262730; /* Streamlit's default text color */
                                font-family: "Source Sans Pro", sans-serif;
                                font-size: 1rem;
                                padding: 0.375rem 0.75rem;
                                border-radius: 0.5rem;
                                border: none;
                                width: 130px;
                                white-space: nowrap;
                                overflow: hidden;
                                text-overflow: ellipsis;
                            }
                            </style>

                            <div class="custom-box">Keine</div>
                        """, unsafe_allow_html=True)
                else:
                    for col_idx in range(t):
                        with cols[col_idx]:
                            st.selectbox(f"Auswahl_{n}", options, key=f"Selection{id}{col_idx}", label_visibility='collapsed')


                with cols[4]:
                    col1, col2 = st.columns([1, 1])
                    if t > 0:
                        with col1:
                            if st.button("➖", key=f"remove{n}"):
                                st.session_state['newtakes'][n] -= 1
                                st.rerun()
                    if t < 4:
                        with col2:
                            if st.button("➕", key=f"add{n}"):
                                st.session_state['newtakes'][n] += 1
                                st.rerun()
                if t < 4:
                    with cols[t]:
                        st.markdown(' ')

        if not st.session_state.get('show_edit', False):
            blocks = {c: {'title': c, 'rows': []} for c in categories}
            for ex in exercises:
                info = catdf[catdf['Name'] == ex].to_dict(orient='records')[0]
                for key in info:
                    if info[key] is np.nan:
                        info[key] = ''
                blocks[info['Kategorie']]['rows'].append((info['Zeit'], info['Name'], info['Ziel'], info['Beschreibung']))
            blocks = list(blocks.values())

            table_style = """
            <style>
            table {
                width: 705px;
                border-collapse: collapse;
                margin-bottom: 25px;
                table-layout: fixed;
            }
            th, td {
                border: 1px solid black;
                padding: 8px;
                vertical-align: top;
                word-wrap: break-word;
            }
            th:nth-child(1), td:nth-child(1) { width: 10%; }  /* Zeit */
            th:nth-child(2), td:nth-child(2) { width: 20%; }  /* Name */
            th:nth-child(3), td:nth-child(3) { width: 20%; }  /* Ziel */
            th:nth-child(4), td:nth-child(4) { width: 50%; }  /* Beschreibung */
            th {
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
            html_content = table_style
            for block in blocks:
                html_content += f"<div class='block-title'>{block['title']}</div>"
                html_content += "<table>"
                html_content += "<tr><th>Zeit</th><th>Name</th><th>Ziel</th><th>Beschreibung</th></tr>"
                for zeit, inhalt, ziel, beschr in block["rows"]:
                    html_content += f"<tr><td>{zeit}</td><td>{inhalt}</td><td>{ziel}</td><td>{beschr}</td></tr>"
                html_content += "</table>"

            st.markdown(html_content, unsafe_allow_html=True)


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
        st.write(3)


    with tab4:
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
                newset = {'weekdays': newweekdays, 'takes': newtakes}
                setname = Team + '/Settings_' + Team + '_' + Saison + '.csv'
                pf = pd.DataFrame([newset])
                pf.to_csv(setname, index=False)
                startdate = datetime.today()
                make_plan(Saison, startdate, Team)
                st.rerun()


