import numpy as np
import pandas as pd
import plotly as py

def read_team(teamname, Saison):
    df = pd.read_csv(teamname + '/Übungen_' + teamname + '.csv')
    Settings = pd.read_csv(teamname + '/Settings' + teamname + '.csv')
    Plan = pd.read_csv(teamname + '/Plans' + teamname + '_' + Saison + '.csv')
    return df, Settings, Plan


