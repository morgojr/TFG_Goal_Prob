import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import ast
import math
import time

from mplsoccer import Pitch, VerticalPitch


GOAL_POSITION_POINT = (105,32.5)
PITCH_DIMENSIONS = (105,65)



def plot_in_pitch(x_positions, y_positions):
    pitch = VerticalPitch(pitch_type='wyscout',pitch_color='grass', line_color='white', stripe=True, linewidth=2)
    fig, ax = pitch.draw(figsize=(7,7))
    pitch.scatter(x_positions,y_positions,color='red',ax=ax, marker='x')



def get_all_events_of_a_match(match_id, df):
    events = df[df['matchId'] == match_id]
    return events


def percent_coord_to_points(X,Y):
    X = (X*PITCH_DIMENSIONS[0])/100
    Y = (Y*PITCH_DIMENSIONS[1])/100
    return X,Y



def get_shot_distance(x,y):
    x_scaled, y_scaled = percent_coord_to_points(x,y)
    distance = math.sqrt(((GOAL_POSITION_POINT[0]-x_scaled)**2)+((GOAL_POSITION_POINT[1]-y_scaled)**2))
    return distance


def get_shot_angle (X,Y):
    x_scaled, y_scaled = percent_coord_to_points(X,Y)
    x = abs(y_scaled-GOAL_POSITION_POINT[1])
    d = get_shot_distance(X,Y)    
    angle = round(np.arcsin(x/d)*180/np.pi,2)
    
    return angle




