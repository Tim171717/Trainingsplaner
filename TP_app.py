import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
from datetime import date, timedelta
from Trainingsplan_maker2 import *

accounts = {'U13A': '1', 'U13B': '2'}
Saisons = ['25/26 HR', '25/26 RR']

# --- Title ---
st.title("Trainingsplaner")

# --- Profil selection ---
teams = accounts.keys()
col1, col2, col3, col4 = st.columns(4)
with col1:
    profil = st.selectbox('Team', teams, label_visibility='collapsed')
with col2:
    password = st.text_input('Password', icon="🗝️", label_visibility='collapsed')
with col3:
    login = st.button('Login')
with col4:
    if accounts[profil] == password and login:
        st.success('Ok cool')
        st.session_state['loggedin'] = True
    elif login:
        st.session_state['loggedin'] = False
        st.error('Falsches Passwort')


if st.session_state.get('loggedin', False):
    tab1, tab2, tab3, tab4 = st.tabs(["Plan", "Nächstes Training", "Übungen", "Einstellungen"])

    with tab1:
        st.write(1)

    with tab2:
        st.write(2)

    with tab3:
        st.write(3)


    with tab4:
        st.header("Saison")
        saison = st.selectbox('Saison', Saisons, label_visibility='collapsed')

        weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        selected_weekdays = st.multiselect("Choose weekdays to include", weekdays, default=weekdays)

        # --- Holiday Selection ---
        st.header("3. Select Holidays")
        holiday_options = [
            "New Year's Day",
            "Independence Day",
            "Thanksgiving",
            "Christmas Day",
            "Labor Day"
        ]
        selected_holidays = st.multiselect("Choose holidays to include", holiday_options)

