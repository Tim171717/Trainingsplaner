import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime
import locale

from Trainingsplan_maker2 import *

accounts = {'U13A': '1', 'U13B': '1'}
Saisons = {'2526HR': 0, '2526RR': 1}
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
        locale.setlocale(locale.LC_TIME, 'de_DE')
        st.markdown(f"### 🤾‍♀️ {ndate.strftime('%A, %d. %B %Y')} 🤾‍♂️")
        blocks = [
            {
                "title": "Aufwärmen",
                "rows": [("20min", "Psychoball", "", "")]
            },
            {
                "title": "Grundübungen",
                "rows": [("20min", "Passen mit Medizinbällen", "Wurftechniken üben",
                          "alle möglichen Arten von Wurfbewegung")]
            },
            {
                "title": "Zonenspiele",
                "rows": [
                    ("20min", "2 gegen 3 mit Kreis", "Kreisspiel erleben", "Alle Kinder einmal Kreis spielen lassen")]
            },
            {
                "title": "Spiel(en)/Gegenstoss",
                "rows": [
                    ("25min", "6 gegen 6", "", ""),
                    ("5min", "Penaltkönig", "", "")
                ]
            }
        ]

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

