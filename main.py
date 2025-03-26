import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
import codecs
from config import logger
import numpy as np
import calendar
import base64
from utils import render_stat_card, render_team_category_card, render_opponent_performance, load_stylesheet, render_key_moments, generate_youtube_search_link 

# Constants with tunable parameters
K_BASE = 40          # Base K-factor
K_RATING_DIV = 400   # Standard ELO divisor
HOME_ADVANTAGE = 30  # Home team advantage in ELO points
AWAY_WIN_BONUS = 1.1
GOAL_DIFF_WEIGHT = 0.5  # Impact factor for goal difference
LATE_SEASON_WEIGHT = 1.2
INITIAL_ELO = 1000
INACTIVITY_DECAY = 0.95  # Decay factor for inactive teams
debug_log_path = os.path.join("C:", "Users", "3fold", "Documents", "Prem ELO", "debug_log.txt")



st.set_page_config(
        page_title="Histo - ELO Tracker",
        page_icon="Assets/logo.png",
        layout="wide",
        initial_sidebar_state="expanded"
    )
# Set page configuration with improved layout and theme
def setup_page():
    with open("Assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    
    
def encode_image(image_path):
    """Encodes an image file to base64 for embedding in HTML."""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    except FileNotFoundError:
        return ""

    
# Define team colours
team_colours = {
    # London Teams
    "Arsenal": {"primary": "#EF0107", "secondary": "#FFFFFF"},  # Red and White
    "Chelsea": {"primary": "#034694", "secondary": "#FFFFFF"},  # Blue and White 
    "Tottenham": {"primary": "#132257", "secondary": "#FFFFFF"},  # Navy and White
    "West Ham": {"primary": "#7A263A", "secondary": "#1BB1E7"},  # Claret and Sky Blue
    "Crystal Palace": {"primary": "#1B458F", "secondary": "#FFD700"},  # Red and Blue
    "Fulham": {"primary": "#FFFFFF", "secondary": "#000000"},  # White and Black
    "QPR": {"primary": "#1D5BA4", "secondary": "#FFFFFF"},  # Blue and White
    "Brentford": {"primary": "#E30613", "secondary": "#FFFFFF"},  # Red and White
    "Charlton": {"primary": "#E31B23", "secondary": "#FFFFFF"},  # Red and White
    "Wimbledon": {"primary": "#2E86C1", "secondary": "#FFFFFF"},  # Blue and White

    # Greater Manchester
    "Man United": {"primary": "#DA291C", "secondary": "#FBE122"},  # Red and Yellow
    "Man City": {"primary": "#6CABDD", "secondary": "#FFFFFF"},  # Sky Blue and Yellow
    "Bolton": {"primary": "#AA0000", "secondary": "#FFFFFF"},  # Red and White
    "Wigan": {"primary": "#18B5B0", "secondary": "#FFFFFF"},  # Blue and White
    "Oldham": {"primary": "#004A97", "secondary": "#FFFFFF"},  # Red and White

    # West Midlands
    "Aston Villa": {"primary": "#7B003C", "secondary": "#FFFFFF"},  # Claret and Blue
    "Birmingham": {"primary": "#223B8F", "secondary": "#FFFFFF"},  # Blue and White
    "West Brom": {"primary": "#2E86C1", "secondary": "#FFFFFF"},  # Blue and White
    "Wolves": {"primary": "#F87E23", "secondary": "#000000"},  # Gold and Black
    "Coventry": {"primary": "#4B92DB", "secondary": "#000000"},  # Sky Blue and Black

    # North West
    "Liverpool": {"primary": "#C8102E", "secondary": "#FFFFFF"},  # Red and Teal
    "Everton": {"primary": "#003399", "secondary": "#FFFFFF"},  # Blue and White
    "Blackburn": {"primary": "#009EE0", "secondary": "#FFFFFF"},  # Blue and White
    "Blackpool": {"primary": "#000000", "secondary": "#FFD700"},  # Black and Tangerine
    "Burnley": {"primary": "#6C1D45", "secondary": "#FFFFFF"},  # Claret and Blue
    "Bolton": {"primary": "#AA0000", "secondary": "#FFFFFF"},  # Red and White

    # North East
    "Newcastle": {"primary": "#241F20", "secondary": "#FFFFFF"},  # Black and White
    "Sunderland": {"primary": "#CC0000", "secondary": "#FFFFFF"},  # Red and White
    "Middlesbrough": {"primary": "#CC0000", "secondary": "#FFFFFF"},  # Red and White

    # Yorkshire
    "Leeds": {"primary": "#FFFFFF", "secondary": "#000000"},  # White and Blue
    "Sheffield United": {"primary": "#E31B23", "secondary": "#FFFFFF"},  # Red and White
    "Sheffield Weds": {"primary": "#003399", "secondary": "#FFFFFF"},  # Blue and White
    "Huddersfield": {"primary": "#0E355E", "secondary": "#FFFFFF"},  # Blue and White
    "Bradford": {"primary": "#CC0000", "secondary": "#FFFFFF"},  # Red and White
    "Barnsley": {"primary": "#C8102E", "secondary": "#FFFFFF"},  # Red and White

    # East Midlands
    "Leicester": {"primary": "#003090", "secondary": "#FFFFFF"},  # Blue and White
    "Nott'm Forest": {"primary": "#C8102E", "secondary": "#FFFFFF"},  # Red and White
    "Derby": {"primary": "#FFFFFF", "secondary": "#000000"},  # White and Black

    # East Anglia
    "Norwich": {"primary": "#FFF200", "secondary": "#00A650"},  # Yellow and Green
    "Ipswich": {"primary": "#003C71", "secondary": "#FFFFFF"},  # Blue and White

    # South Coast
    "Southampton": {"primary": "#D71921", "secondary": "#FFFFFF"},  # Red and White
    "Bournemouth": {"primary": "#C8102E", "secondary": "#FFFFFF"},  # Red and Black
    "Portsmouth": {"primary": "#003C6B", "secondary": "#FFFFFF"},  # Blue and White
    "Brighton": {"primary": "#005DAA", "secondary": "#FFFFFF"},  # Blue and White
    "Hull": {"primary": "#000000", "secondary": "#F18A01"},  # Blue and White

    # Wales
    "Cardiff": {"primary": "#00205B", "secondary": "#FFFFFF"},  # Blue and White
    "Swansea": {"primary": "#000000", "secondary": "#FFFFFF"},  # Black and White

    # Other
    "Stoke": {"primary": "#E03A3E", "secondary": "#FFFFFF"},  # Red and White
    "Swindon": {"primary": "#C8102E", "secondary": "#FFFFFF"},  # Red and White
    "Watford": {"primary": "#FBEE23", "secondary": "#E51837"},  # Yellow and Red
    "Reading": {"primary": "#020202", "secondary": "#FFFFFF"},  # Blue and White
    "Luton": {"primary": "#000000", "secondary": "#FFFFFF"}  # Orange and White
}

derby_matrix = {
    # London Derbies (all London clubs are considered local rivals)
    ('Arsenal', 'Chelsea'): True,
    ('Arsenal', 'Tottenham'): True,
    ('Arsenal', 'West Ham'): True,
    ('Arsenal', 'Crystal Palace'): True,
    ('Arsenal', 'Fulham'): True,
    ('Arsenal', 'QPR'): True,
    ('Arsenal', 'Brentford'): True,
    ('Arsenal', 'Charlton'): True,
    ('Arsenal', 'Wimbledon'): True,
    
    ('Chelsea', 'Tottenham'): True,
    ('Chelsea', 'West Ham'): True,
    ('Chelsea', 'Crystal Palace'): True,
    ('Chelsea', 'Fulham'): True,
    ('Chelsea', 'QPR'): True,
    ('Chelsea', 'Brentford'): True,
    ('Chelsea', 'Charlton'): True,
    ('Chelsea', 'Wimbledon'): True,
    
    ('Tottenham', 'West Ham'): True,
    ('Tottenham', 'Crystal Palace'): True,
    ('Tottenham', 'Fulham'): True,
    ('Tottenham', 'QPR'): True,
    ('Tottenham', 'Brentford'): True,
    ('Tottenham', 'Charlton'): True,
    ('Tottenham', 'Wimbledon'): True,
    
    ('West Ham', 'Crystal Palace'): True,
    ('West Ham', 'Fulham'): True,
    ('West Ham', 'QPR'): True,
    ('West Ham', 'Brentford'): True,
    ('West Ham', 'Charlton'): True,
    ('West Ham', 'Wimbledon'): True,
    
    ('Crystal Palace', 'Fulham'): True,
    ('Crystal Palace', 'QPR'): True,
    ('Crystal Palace', 'Brentford'): True,
    ('Crystal Palace', 'Charlton'): True,
    ('Crystal Palace', 'Wimbledon'): True,
    
    ('Fulham', 'QPR'): True,
    ('Fulham', 'Brentford'): True,
    ('Fulham', 'Charlton'): True,
    ('Fulham', 'Wimbledon'): True,
    
    ('QPR', 'Brentford'): True,
    ('QPR', 'Charlton'): True,
    ('QPR', 'Wimbledon'): True,
    
    ('Brentford', 'Charlton'): True,
    ('Brentford', 'Wimbledon'): True,
    
    ('Charlton', 'Wimbledon'): True,
    
    # Greater Manchester Derbies
    ('Man United', 'Man City'): True,
    ('Bolton', 'Wigan'): True,
    ('Bolton', 'Oldham'): True,
    ('Wigan', 'Oldham'): True,
    
    # West Midlands Derbies
    ('Aston Villa', 'Birmingham'): True,
    ('Aston Villa', 'West Brom'): True,
    ('Birmingham', 'West Brom'): True,
    ('Birmingham', 'Coventry'): True,
    ('Wolves', 'Coventry'): True,
    
    # North West Derbies
    ('Liverpool', 'Everton'): True,
    ('Blackburn', 'Blackpool'): True,
    ('Burnley', 'Blackpool'): True,
    
    # North East Derbies
    ('Newcastle', 'Sunderland'): True,
    
    # Yorkshire Derbies
    ('Sheffield United', 'Sheffield Weds'): True,
    ('Leeds', 'Huddersfield'): True,
    ('Bradford', 'Barnsley'): True,
    
    # East Midlands Derbies
    ('Derby', "Nott'm Forest"): True,
    
    # East Anglia Derbies
    ('Norwich', 'Ipswich'): True,
    
    # South Coast Derbies
    ('Southampton', 'Portsmouth'): True,
    
    # Wales Derbies
    ('Cardiff', 'Swansea'): True,
    
    # Other Derbies
    ('Watford', 'Luton'): True
}

# Define team colors with adjustments for readability
# Define team colors with adjustments for readability
def get_team_colors(team):
    default_colors = {"primary": "#808080", "secondary": "#FFFFFF"}  # Gray/White fallback
    colors = team_colours.get(team, default_colors)
    
    primary = colors["primary"]
    secondary = colors["secondary"]

    # Check if the primary color is too light
    def is_light(color):
        r, g, b = [int(color[i:i+2], 16) for i in (1, 3, 5)]
        return (r * 0.299 + g * 0.587 + b * 0.114) > 186

    text_color = "#000000" if is_light(primary) else "#FFFFFF"
    
    return primary, secondary, text_color


# Function to apply team colors to DataFrame styles
def apply_team_colours(df):
    def colour_row(row):
        team = row['Team']
        primary, secondary, text_color = get_team_colors(team)
        
        # Log initial color values
        logger.debug(f"Team: {team} - Initial colors: {primary}, {secondary}")
        
        # Log final color values after text color calculation
        logger.debug(f"Team: {team} - Final colors: primary={primary}, secondary={secondary}, text_color={text_color}")
        
        return [f"background-color: {primary}; color: {text_color}"] * len(row)
    
    return df.style.apply(colour_row, axis=1)

# Function to create color-coded bar charts
# Function to create color-coded bar charts
def plot_season_rankings(rankings, season):
    rankings["primary_color"] = rankings["Team"].apply(lambda t: get_team_colors(t)[0])
    fig = go.Figure(data=[go.Bar(
        x=rankings['Team'],
        y=rankings['ELO'],
        marker=dict(color=rankings['primary_color']),
        text=rankings['ELO'].round(1),
        textposition='auto',
    )])
    fig.update_layout(
    title=f"{season} Final ELO Rankings",
    xaxis=dict(title="Team", showgrid=False, tickfont=dict(size=12, color="#333333")),
    yaxis=dict(title="ELO Rating", showgrid=False),
    font=dict(family="Arial", size=14, color="#333333"),
    template="plotly_white",
    plot_bgcolor="#f8f9fa"
)

    return fig

def plot_season_overview_bubble(rankings, season):
    """
    Create an enhanced bubble chart for the season overview with custom color coding,
    hover template, and axis styling.
    
    Parameters:
      - rankings: DataFrame with columns 'Team', 'ELO', 'Rank', and 'Category'.
      - season: Season identifier (used in the title).
      
    Returns:
      - Plotly Express figure displaying the bubble chart.
    """
    fig = px.scatter(
        rankings,
        x="Rank",
        y="ELO",
        size="ELO",
        color="Category",
        hover_name="Team",
        size_max=80,
        title=f"Season Overview Bubble Chart - {season}",
        labels={
            "Rank": "Team Rank", 
            "ELO": "Final ELO Rating", 
            "Category": "Team Category"
        },
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    
    # Customize hover template to display more insights
    fig.update_traces(
        hovertemplate=
            "<b>%{hovertext}</b><br>" +
            "Rank: %{x}<br>" +
            "ELO: %{y:.1f}<br>" +
            "Category: %{marker.color}<extra></extra>",
        opacity=0.85
    )
    
    # Update layout for better readability
    fig.update_layout(
        xaxis=dict(
            showgrid=True,
            gridcolor="lightgrey",
            zeroline=False,
            title="Team Rank",
            tickfont=dict(size=12, color="#333333")
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="lightgrey",
            zeroline=False,
            title="ELO Rating"
        ),
        font=dict(family="Arial", size=14, color="#333333"),
        template="plotly_white",
        plot_bgcolor="#f8f9fa",
        legend_title_text="Team Category"
    )
    
    return fig

def plot_team_elo_timeline(team, season, team_history, match_history):
    # Check if the season exists for the team
    if season not in team_history[team]:
        return None

    # Get the ELO history for the season
    history = team_history[team][season]  # List of tuples: (match_number, elo)
    
    # If the history has one extra entry (initial value), drop it
    team_matches = match_history[(match_history["Season"] == season) &
                                 ((match_history["HomeTeam"] == team) | (match_history["AwayTeam"] == team))]
    if len(history) == len(team_matches) + 1:
        history = history[1:]
    
    # Extract match numbers and ELO values
    match_numbers = [num for num, elo in history]
    elos = [elo for num, elo in history]

    # Use the 'Date' column from the team's match records.
    # We'll take only as many dates as we have ELO points.
    dates = team_matches["Date"].iloc[:len(match_numbers)].tolist()
    
    # Ensure all arrays are the same length
    min_length = min(len(match_numbers), len(elos), len(dates))
    match_numbers = match_numbers[:min_length]
    elos = elos[:min_length]
    dates = dates[:min_length]
    
    # Build a DataFrame for plotting
    df_plot = pd.DataFrame({
        "Match Number": match_numbers,
        "ELO": elos,
        "Date": dates
    })
    
    # Create a scatter plot with lines connecting markers (timeline view)
    fig = px.scatter(
        df_plot,
        x="Date",
        y="ELO",
        hover_data=["Match Number"],
        title=f"{team} ELO Progression - {season}",
        labels={"Date": "Match Date", "ELO": "ELO Rating"}
    )
    
    
    primary, secondary, text = get_team_colors(team)
    
    fig.update_traces(
        mode='lines+markers',
        marker=dict(size=10, opacity=0.8, color=primary),
        line=dict(color=primary, width=2)
    )
    
    # Update layout for clarity
    fig.update_layout(
        xaxis=dict(showgrid=True, gridcolor="lightgrey", title="Match Date"),
        yaxis=dict(showgrid=True, gridcolor="lightgrey", title="ELO Rating"),
        font=dict(family="Arial", size=14, color="#333333"),
        template="plotly_white",
        plot_bgcolor="#f8f9fa"
    )
    
    return fig


# Function to create color-coded line charts
def plot_elo_progression(team, season, team_history, match_history):
    if season in team_history[team]:
        data = team_history[team][season]
        
        # Extract match dates and ELO values
        season_matches = match_history[(match_history["Season"] == season) & 
                                       ((match_history["HomeTeam"] == team) | (match_history["AwayTeam"] == team))]

        if season_matches.empty:
            return None

        match_dates = season_matches["Date"]
        elos = [elo for _, elo in data]

        # Convert dates to Month-Year format (e.g., "Sep 2023")
        month_labels = match_dates.dt.strftime("%b %Y")

        # Get team color
        primary, _, _ = get_team_colors(team)

        # Create a step chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=month_labels,
            y=elos,
            mode='lines+markers',
            name=team,
            line=dict(shape='hv', width=2, color=primary),  # "hv" makes it a step-line
            marker=dict(size=8)
        ))

        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="ELO Rating",
            title=f"{team} ELO Rating Progression ({season})",
            hovermode="x unified"
        )

        return fig
    return None



def get_team_category(rank, total_teams):
    """Classify teams based on their final ranking"""
    if rank <= 4:
        return "Elite"
    elif 5 <= rank <= 7:
        return "Contender"
    elif rank > total_teams - 3:
        return "Relegation"
    else:
        return "Mid-Table"


def is_derby_match(home_team, away_team):
    return derby_matrix.get((home_team, away_team), False) or derby_matrix.get((away_team, home_team), False)


def format_team_name(team_name):
    """Formats team names to match logo file naming conventions."""
    return team_name.strip()

def get_team_logo_path(team_name):
    formatted_name = format_team_name(team_name)
    logo_path = f"Assets/team_logos/{formatted_name}.png"
    if os.path.exists(logo_path):
        return logo_path
    else:
        return "Assets/team_logos/default.png"

def get_team_background_path(team):
    background_path = f"Assets/team_backgrounds/{team}.jpg"
    if os.path.exists(background_path):
        return background_path
    else:
        return "Assets/team_backgrounds/default.jpg"  # fallback if the team image doesn't exist



# Load the match results data
@st.cache_data
def load_data():
    # Check if the file exists
    if os.path.exists("data/results.csv"):
        with codecs.open("data/results.csv", "r", "utf-8", errors="replace") as f:
            df = pd.read_csv(f)
        
        # Keep only necessary columns
        columns_to_keep = ["Season", "DateTime", "HomeTeam", "AwayTeam", "FTR", "FTHG", "FTAG"]
        df = df[columns_to_keep]
        df.dropna(inplace=True)
        df["DateTime"] = pd.to_datetime(df["DateTime"])
        df.sort_values(by=["Season", "DateTime"], inplace=True)
        return df
    else:
        st.error("Data file not found. Please ensure 'data/results.csv' exists.")
        return pd.DataFrame()

def dynamic_k_factor(rating, matches_played, is_new_team=False):
    """
    Calculate dynamic K-factor based on team's rating and match count.
    Uses default matches_played = 10 and is_new_team = False when data is unavailable.
    """
    k_new = np.where(is_new_team, K_BASE * 1.5, K_BASE)
    k_adjustment = np.maximum(5, np.minimum(matches_played, 30)) / 30
    rating_adjustment = np.maximum(0.8, np.minimum(2000, rating) / 2000)
    return k_new * (1.1 - k_adjustment * 0.3) * (1.2 - rating_adjustment * 0.2)

def get_expected_score(elo_a, elo_b, home_advantage=HOME_ADVANTAGE):
    """
    Calculate expected score for the home team using the ELO formula.
    """
    adjusted_elo_a = elo_a + home_advantage
    return 1 / (1 + 10 ** ((elo_b - adjusted_elo_a) / K_RATING_DIV))

def calculate_margin_factor(goal_diff):
    """
    Calculate a non-linear factor based on goal margin.
    This dampens rating changes in the case of blowout results.
    """
    return np.log1p(np.abs(goal_diff)) * GOAL_DIFF_WEIGHT + 1

def update_elo(home_elo, away_elo, home_goals, away_goals, late_season=False, derby=False):
    # Compute goal difference
    goal_diff = home_goals - away_goals

    # Use default values: assume 10 matches played and teams are not new
    k_home = dynamic_k_factor(home_elo, 10, False)
    k_away = dynamic_k_factor(away_elo, 10, False)

    # Calculate margin factor based on goal difference
    margin_factor = calculate_margin_factor(goal_diff)
    
    # Apply derby multiplier if it's a derby match
    if derby:
        margin_factor *= 1.25  # You can adjust this multiplier as needed

    # Calculate expected scores (home advantage applied for home team)
    expected_home = get_expected_score(home_elo, away_elo, HOME_ADVANTAGE)
    # For the away team, remove home advantage by inverting the formula:
    expected_away = 1 / (1 + 10 ** ((home_elo - (away_elo - HOME_ADVANTAGE)) / K_RATING_DIV))

    # Determine actual match outcomes
    if goal_diff > 0:
        home_result, away_result = 1, 0
    elif goal_diff == 0:
        home_result, away_result = 0.5, 0.5
    else:
        home_result, away_result = 0, 1

    # Compute rating changes
    home_change = k_home * margin_factor * (home_result - expected_home)
    away_change = k_away * margin_factor * (away_result - expected_away)
    
    # Apply away win bonus if the away team wins
    if goal_diff < 0:
        away_change *= AWAY_WIN_BONUS

    new_home_elo = home_elo + home_change
    new_away_elo = away_elo + away_change
    return new_home_elo, new_away_elo


def apply_inactivity_decay(all_teams_elo, active_teams, weeks_inactive=None):
    """
    Apply rating decay for teams not active in the current period.
    """
    if weeks_inactive is None:
        for team in all_teams_elo:
            if team not in active_teams:
                all_teams_elo[team] *= INACTIVITY_DECAY
    else:
        for team, weeks in weeks_inactive.items():
            if weeks > 0:
                all_teams_elo[team] *= (INACTIVITY_DECAY ** (weeks / 8))
    return all_teams_elo



@st.cache_data
def calculate_elo(df):
    logger.debug("calculate_elo function called")
    
    team_history = {}
    season_final_elo = {}
    previous_season_elo = {}
    match_history = []
    team_seasons = {}  # Track which seasons each team has played in

    seasons = sorted(df["Season"].unique())
    for season in seasons:
        season_df = df[df["Season"] == season]
        teams_in_season = pd.concat([season_df["HomeTeam"], season_df["AwayTeam"]]).unique()
        
        # Identify new teams in the season
        new_teams = [team for team in teams_in_season if team not in previous_season_elo]
        
        # Initialize ELO ratings for the season
        for team in teams_in_season:
            if team not in team_history:
                team_history[team] = {}
            # Use previous season ELO if available, otherwise use INITIAL_ELO
            starting_elo = previous_season_elo.get(team, INITIAL_ELO)
            # For new teams, we can also choose to reset to INITIAL_ELO if desired:
            if team in new_teams:
                starting_elo = INITIAL_ELO
            team_history[team][season] = [(0, starting_elo)]
            
            # Track seasons for each team
            if team not in team_seasons:
                team_seasons[team] = []
            team_seasons[team].append(season)

        # Process each match in the season
        season_matches = season_df.reset_index(drop=True)
        total_matches = len(season_matches)
        for i, match in season_matches.iterrows():
            home, away = match["HomeTeam"], match["AwayTeam"]
            home_goals, away_goals = match["FTHG"], match["FTAG"]
            match_date = match["DateTime"]
            late_season = i > (total_matches / 2)
            
            current_home_elo = team_history[home][season][-1][1]
            current_away_elo = team_history[away][season][-1][1]
            
            # Check if the match is a derby
            derby = is_derby_match(home, away)
            
            # Update ELO with the derby flag
            new_home_elo, new_away_elo = update_elo(current_home_elo, current_away_elo, home_goals, away_goals, late_season, derby)
            
            # Save updated ELOs in team history
            team_history[home][season].append((i + 1, new_home_elo))
            team_history[away][season].append((i + 1, new_away_elo))
            
            # Record detailed match history for debugging and analysis
            match_history.append({
                'Season': season,
                'Date': match_date,
                'HomeTeam': home,
                'AwayTeam': away,
                'HomeGoals': home_goals,
                'AwayGoals': away_goals,
                'Result': match["FTR"],
                'HomeElo_Before': current_home_elo,
                'AwayElo_Before': current_away_elo,
                'HomeElo_After': new_home_elo,
                'AwayElo_After': new_away_elo,
                'HomeElo_Change': new_home_elo - current_home_elo,
                'AwayElo_Change': new_away_elo - current_away_elo
            })
        
        # Finalize season ELOs for each team for carry-over
        for team in teams_in_season:
            final_elo = team_history[team][season][-1][1]
            if team not in season_final_elo:
                season_final_elo[team] = {}
            season_final_elo[team][season] = final_elo
            previous_season_elo[team] = final_elo

    return team_history, season_final_elo, pd.DataFrame(match_history), team_seasons


# Get end-of-season rankings
def get_season_overview(season_final_elo, season):
    rankings = []
    for team, elo_data in season_final_elo.items():
        if season in elo_data:
            rankings.append({'Team': team, 'ELO': elo_data[season]})

    df = pd.DataFrame(rankings).sort_values('ELO', ascending=False).reset_index(drop=True)
    df["Rank"] = df.index + 1  # Add rank column
    total_teams = len(df)

    # Assign categories
    df["Category"] = df["Rank"].apply(lambda rank: get_team_category(rank, total_teams))

    # Group teams by category
    elite_teams = df[df["Category"] == "Elite"]["Team"].tolist()
    contender_teams = df[df["Category"] == "Contender"]["Team"].tolist()
    mid_table_teams = df[df["Category"] == "Mid-Table"]["Team"].tolist()
    relegation_teams = df[df["Category"] == "Relegation"]["Team"].tolist()

    return elite_teams, contender_teams, mid_table_teams, relegation_teams, df  # Now returning DataFrame too



# Create a sidebar for navigation and filters
def create_sidebar(team_history, seasons):
    with st.sidebar:
        logo_path = "Assets/logo.png"
        if os.path.exists(logo_path):
            st.image(logo_path, width=500)
        st.title("Navigation")

        # User mode selection
        user_mode = st.radio(
            "Select Mode",
            ["Team Analysis", "Season Overview", "Club Dashboard", "About"],
            index=0
        )

        st.divider()

        # Common filters section
        st.subheader("Filters")
        if user_mode == "Team Analysis":
            selected_team = st.selectbox("Select a Team", sorted(team_history.keys()))
            multi_season = st.checkbox("Compare Multiple Seasons", value=False)

            if multi_season:
                available_seasons = sorted(team_history.get(selected_team, {}).keys())
                selected_seasons = st.multiselect(
                    "Select Seasons", 
                    available_seasons, 
                    default=available_seasons[-2:] if len(available_seasons) >= 2 else available_seasons
                )
            else:
                available_seasons = sorted(team_history.get(selected_team, {}).keys())
                selected_season = st.selectbox("Select a Season", available_seasons)

        elif user_mode == "Season Overview":
            selected_season = st.selectbox("Select Season", seasons)
            display_mode = st.radio(
                "Display Mode",
                ["Chart", "Table", "Both"],
                index=2
            )

        elif user_mode == "Club Dashboard":
            selected_team = st.selectbox("Select a Team", sorted(team_history.keys()))


    # Return all the selected values
    if user_mode == "Team Analysis":
        if multi_season:
            return user_mode, {"team": selected_team, "multi_season": True, "seasons": selected_seasons}
        else:
            return user_mode, {"team": selected_team, "multi_season": False, "season": selected_season}
    elif user_mode == "Season Overview":
        return user_mode, {"season": selected_season, "display_mode": display_mode}
    elif user_mode == "Club Dashboard":
        return user_mode, {"team": selected_team}
    else:
        return user_mode, {}


# Create a modern dashboard header
def create_header(): 
    st.markdown("""
    <style>
    .shine {
    font-size: 5em;
    font-weight: 900;
    color: #123524;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
    background: #222 -webkit-gradient(
        linear,
        left top,
        right top,
        from(#222),
        to(#222),
        color-stop(0.5, #fff)
        ) 0 0 no-repeat;
    background-image: -webkit-linear-gradient(
        -40deg,
        transparent 0%,
        transparent 40%,
        #fff 50%,
        transparent 60%,
        transparent 100%
    );
    -webkit-background-clip: text;
    -webkit-background-size: 100px;
    -webkit-animation: zezzz;
    -webkit-animation-duration: 3s;
    -webkit-animation-iteration-count: infinite;
    }
    @-webkit-keyframes zezzz {
    0%,
    10% {
        background-position: -200px;
    }
    20% {
        background-position: top left;
    }
    100% {
        background-position: 200px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    lt = st.empty()
    
    lt.html("""
    <div class="shine">HISTO</div>
            """)


def calculate_team_stats(team, match_history):
    team_matches = match_history[(match_history["HomeTeam"] == team) | (match_history["AwayTeam"] == team)]

    # Compute Wins, Draws, Losses vs Higher & Lower ELO
    higher_matches = team_matches[((team_matches["HomeTeam"] == team) & (team_matches["HomeElo_Before"] < team_matches["AwayElo_Before"])) |
                                  ((team_matches["AwayTeam"] == team) & (team_matches["AwayElo_Before"] < team_matches["HomeElo_Before"]))]
    lower_matches = team_matches[((team_matches["HomeTeam"] == team) & (team_matches["HomeElo_Before"] > team_matches["AwayElo_Before"])) |
                                 ((team_matches["AwayTeam"] == team) & (team_matches["AwayElo_Before"] > team_matches["HomeElo_Before"]))]

    higher_wins = higher_matches[((higher_matches["HomeTeam"] == team) & (higher_matches["Result"] == "H")) |
                                  ((higher_matches["AwayTeam"] == team) & (higher_matches["Result"] == "A"))].shape[0]
    higher_draws = higher_matches[(higher_matches["Result"] == "D")].shape[0]
    higher_losses = higher_matches.shape[0] - (higher_wins + higher_draws)

    lower_wins = lower_matches[((lower_matches["HomeTeam"] == team) & (lower_matches["Result"] == "H")) |
                                ((lower_matches["AwayTeam"] == team) & (lower_matches["Result"] == "A"))].shape[0]
    lower_draws = lower_matches[(lower_matches["Result"] == "D")].shape[0]
    lower_losses = lower_matches.shape[0] - (lower_wins + lower_draws)

    # Convert to percentages
    higher_total = higher_wins + higher_draws + higher_losses
    lower_total = lower_wins + lower_draws + lower_losses

    higher_win_pct = round((higher_wins / higher_total) * 100, 1) if higher_total > 0 else 0
    higher_draw_pct = round((higher_draws / higher_total) * 100, 1) if higher_total > 0 else 0
    higher_loss_pct = round((higher_losses / higher_total) * 100, 1) if higher_total > 0 else 0

    lower_win_pct = round((lower_wins / lower_total) * 100, 1) if lower_total > 0 else 0
    lower_draw_pct = round((lower_draws / lower_total) * 100, 1) if lower_total > 0 else 0
    lower_loss_pct = round((lower_losses / lower_total) * 100, 1) if lower_total > 0 else 0

    # Find Best/Worst Opponents (Win Rate) - At least 4 matches played
    opponents = pd.concat([team_matches["HomeTeam"], team_matches["AwayTeam"]]).unique()
    opponent_stats = []

    for opp in opponents:
        if opp == team:
            continue
        opp_matches = team_matches[(team_matches["HomeTeam"] == opp) | (team_matches["AwayTeam"] == opp)]
        opp_wins = opp_matches[((opp_matches["HomeTeam"] == team) & (opp_matches["Result"] == "H")) |
                               ((opp_matches["AwayTeam"] == team) & (opp_matches["Result"] == "A"))].shape[0]
        opp_total = opp_matches.shape[0]
        win_rate = round((opp_wins / opp_total) * 100, 1) if opp_total > 0 else 0

        if opp_total >= 2:  # Ensure at least 2 matches played
            opponent_stats.append({"Opponent": opp, "Matches Played": opp_total, "Win Rate": win_rate})

    opponent_stats_df = pd.DataFrame(opponent_stats)
    best_opponent = opponent_stats_df.loc[opponent_stats_df["Win Rate"].idxmax()] if not opponent_stats_df.empty else None
    worst_opponent = opponent_stats_df.loc[opponent_stats_df["Win Rate"].idxmin()] if not opponent_stats_df.empty else None
    most_played_opponent = opponent_stats_df.loc[opponent_stats_df["Matches Played"].idxmax()] if not opponent_stats_df.empty else None
    worst_opponents = opponent_stats_df.sort_values("Win Rate").head(5).to_dict(orient="records")


    return {
        "higher_win_pct": higher_win_pct,
        "higher_draw_pct": higher_draw_pct,
        "higher_loss_pct": higher_loss_pct,
        "lower_win_pct": lower_win_pct,
        "lower_draw_pct": lower_draw_pct,
        "lower_loss_pct": lower_loss_pct,
        "best_opponent": best_opponent,
        "worst_opponent": worst_opponent,
        "most_played_opponent": most_played_opponent,
        'worst_opponents': worst_opponents
    }

# Display team categories in card format
def display_category_cards(elite, contenders, mid_table, relegation):
    category_styles = {
        "Elite": "background-color: #FFD700; color: black;",
        "Contender": "background-color: #1E90FF; color: white;",
        "Mid-Table": "background-color: #31b6b0; color: white;",
        "Relegation": "background-color: #FF4C4C; color: white;"
    }

    categories = {
        "Elite": elite,
        "Contender": contenders,
        "Mid-Table": mid_table,
        "Relegation": relegation
    }

    for category, teams in categories.items():
        with st.container():
            st.markdown(f"""
            <div style="{category_styles[category]} padding: 15px; border-radius: 10px; text-align: left; margin-bottom: 10px;">
                <h4>{category} Teams</h4>
                <div class="team-card">
                    <img src="path/to/team-logo.png" alt="Team Logo">
                    <div class="team-info">
                        <p>{", ".join(teams) if teams else "No teams in this category"}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)



# Enhanced team analysis view
def show_team_analysis(team_history, settings, match_history):
    team = settings["team"]
    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
    multi_season = settings["multi_season"]

    st.markdown(f"<h2 style='text-align:center; color:#FFFFFF;'>{team} ELO Performance</h2>", unsafe_allow_html=True)

    team_logo_path = f"Assets/team_logos/{team}.png"
    if os.path.exists(team_logo_path):
        imzy = encode_image(team_logo_path)
        st.markdown(f"""
        <div style='text-align: center;'>
            <img src="data:image/png;base64,{imzy}" style="width: 150px;">
        </div>
    """, unsafe_allow_html=True)
    else:
        st.warning(f"Image not found: {team_logo_path}")

    # ELO Progression Graph
    if not multi_season:
        season = settings["season"]
        st.markdown(f"<h4 style='text-align:center; color:#FFFFFF;'>ELO Rating in {season}</h4>", unsafe_allow_html=True)
        st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)

        # Use the new timeline function instead of the step chart
        fig = plot_team_elo_timeline(team, season, team_history, match_history)
        
        fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#1E1E1E",
        paper_bgcolor="#1E1E1E",
        font=dict(family="Roboto, sans-serif", size=14, color="#FFFFFF")
        )
        
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    else:
        seasons = settings["seasons"]
        st.markdown(f"<h4 style='color:#FFFFFF;'>ELO Comparison Across Seasons</h4>", unsafe_allow_html=True)
        # Multi-Season Graph (existing code)
        fig = go.Figure()
        for season in seasons:
            if season in team_history[team]:
                match_numbers, elos = zip(*team_history[team][season])
                fig.add_trace(go.Scatter(
                    x=list(range(len(match_numbers))),
                    y=elos,
                    mode='lines',
                    name=season,
                    line=dict(width=2)
                ))
        fig.update_layout(
            xaxis_title="Match Number",
            yaxis_title="ELO Rating",
            hovermode="x unified",
            legend_title="Season",
            template="plotly_white"
        )
        
        fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#1E1E1E",
        paper_bgcolor="#1E1E1E",
        font=dict(family="Roboto, sans-serif", size=14, color="#FFFFFF")
        )

        st.plotly_chart(fig, use_container_width=True)

    # Key Matches (Only if not multi-season)
    if not multi_season:
        season_data = team_history[team][season]
        elos = [elo for _, elo in season_data]

        # Filter the team's matches for this season
        team_matches = match_history[(match_history["Season"] == season) & 
                                     ((match_history["HomeTeam"] == team) | (match_history["AwayTeam"] == team))]

        # Identify biggest ELO swings
        if len(elos) > 1:
            elo_changes = [elos[i] - elos[i - 1] for i in range(1, len(elos))]
            biggest_gain_idx = elo_changes.index(max(elo_changes)) + 1
            biggest_loss_idx = elo_changes.index(min(elo_changes)) + 1

            # Find the corresponding match details
            biggest_gain_match = team_matches.iloc[biggest_gain_idx - 1]
            biggest_loss_match = team_matches.iloc[biggest_loss_idx - 1]

            # Determine opponent, match location, and score
            gain_opponent = biggest_gain_match["AwayTeam"] if biggest_gain_match["HomeTeam"] == team else biggest_gain_match["HomeTeam"]
            gain_location = "Home" if biggest_gain_match["HomeTeam"] == team else "Away"
            gain_date = biggest_gain_match["Date"].strftime('%d %b %Y')
            gain_score = f"{biggest_gain_match['HomeGoals']} - {biggest_gain_match['AwayGoals']}"

            loss_opponent = biggest_loss_match["AwayTeam"] if biggest_loss_match["HomeTeam"] == team else biggest_loss_match["HomeTeam"]
            loss_location = "Home" if biggest_loss_match["HomeTeam"] == team else "Away"
            loss_date = biggest_loss_match["Date"].strftime('%d %b %Y')
            loss_score = f"{biggest_loss_match['HomeGoals']} - {biggest_loss_match['AwayGoals']}"
            
            draws = team_matches[team_matches["Result"] == "D"]
            gain_elo_change = elo_changes[biggest_gain_idx - 1]
            loss_elo_change = elo_changes[biggest_loss_idx - 1]

            if not draws.empty:
                # Determine opponent and their pre-match ELO
                draws["Opponent"] = draws.apply(lambda row: row["AwayTeam"] if row["HomeTeam"] == team else row["HomeTeam"], axis=1)
                draws["OpponentElo"] = draws.apply(lambda row: row["AwayElo_Before"] if row["HomeTeam"] == team else row["HomeElo_Before"], axis=1)
                
                # Find draw vs strongest opponent (highest ELO)
                strongest_draw = draws.loc[draws["OpponentElo"].idxmax()]
                
                draw_opponent = strongest_draw["Opponent"]
                draw_score = f"{strongest_draw['HomeGoals']} - {strongest_draw['AwayGoals']}"
                draw_location = "Home" if strongest_draw["HomeTeam"] == team else "Away"
                draw_date = strongest_draw["Date"].strftime('%d %b %Y')
                draw_logo = get_team_logo_path(draw_opponent)
            else:
                draw_opponent = "N/A"
                draw_score = "0 - 0"
                draw_location = "N/A"
                draw_date = "N/A"
                draw_logo = None

            gain_link = generate_youtube_search_link(team, gain_opponent, gain_score, gain_date)
            loss_link = generate_youtube_search_link(team, loss_opponent, loss_score, loss_date)
            draw_link = generate_youtube_search_link(team, draw_opponent, draw_score, draw_date)  

            render_key_moments(
                gain={
                    "Opponent": gain_opponent,
                    "Score": gain_score,
                    "Location": gain_location,
                    "Date": gain_date,
                    "Logo": get_team_logo_path(gain_opponent),
                    "ELO": elo_changes[biggest_gain_idx - 1],
                    "Link": gain_link
                },
                loss={
                    "Opponent": loss_opponent,
                    "Score": loss_score,
                    "Location": loss_location,
                    "Date": loss_date,
                    "Logo": get_team_logo_path(loss_opponent),
                    "ELO": elo_changes[biggest_loss_idx - 1],
                    "Link": loss_link
                },
                draw={
                    "Opponent": draw_opponent,
                    "Score": draw_score,
                    "Location": draw_location,
                    "Date": draw_date,
                    "Logo": get_team_logo_path(draw_opponent),
                    "ELO": 0,
                    "Link": draw_link
                }
            )





def show_season_overview(season_final_elo, settings):
    season = settings["season"]

    # Get rankings and team categories
    elite_teams, contender_teams, mid_table_teams, relegation_teams, rankings = get_season_overview(season_final_elo, season)

    if rankings.empty:
        st.warning(f"No ranking data available for {season}. Check the data processing pipeline.")
        return

    # Display ELO-based category heading
    st.subheader("🔢 Team Categories Based on ELO Rankings")
    cols = st.columns([1, 1, 1, 1]) if st.session_state.get("screen_width", 1000) > 1000 else st.columns(2)
    
    
    screen_width = st.session_state.get("screen_width", 1000)
    num_cols = 4 if screen_width > 1000 else 2
    cols = st.columns(num_cols)


    # Define categories and their content
    categories = [
    ("🥇 Elite Teams", elite_teams, "gold-red"),
    ("💪 Contender Teams", contender_teams, "blue"),
    ("⚖️ Mid-Table Teams", mid_table_teams, "gray"),
    ("📉 Relegation Teams", relegation_teams, "red")
    ]

    # Display each category in a responsive layout
    for i, (label, teams, colour) in enumerate(categories):
        with cols[i % num_cols]:
            render_team_category_card(label, teams, gradient=colour)
            
    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)

    fig = plot_season_overview_bubble(rankings, season)
    
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#1E1E1E",
        paper_bgcolor="#1E1E1E",
        font=dict(family="Roboto, sans-serif", size=14, color="#FFFFFF")
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)


    # Static bar chart (Final ELO)
    fig = px.bar(
        rankings.sort_values("ELO", ascending=False),
        x="ELO",
        y="Team",
        color="Team",
        text="ELO",
        orientation="h",
        title=f"Final ELO Rankings for {season}",
        color_discrete_map={team: get_team_colors(team)[0] for team in rankings["Team"].unique()}
    )

    # Improve layout
    fig.update_layout(
        xaxis_title="ELO Rating",
        yaxis_title="Team",
        template="plotly_white",
        yaxis=dict(categoryorder="total ascending")  # Keep highest ELO at the top
    )
    
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#1E1E1E",
        paper_bgcolor="#1E1E1E",
        font=dict(family="Roboto, sans-serif", size=14, color="#FFFFFF")
    )


    st.plotly_chart(fig, use_container_width=True)


    # Colored season rankings visualization

def show_club_dashboard(match_history, settings):
    st.header(f"📊 {settings['team']} Performance Dashboard")

    

    team = settings["team"]
    team_matches = match_history[(match_history["HomeTeam"] == team) | (match_history["AwayTeam"] == team)]
    team_logo_path = f"Assets/team_logos/{team}.png"
    
    
    # Display team logo
    if os.path.exists(team_logo_path):
        st.image(team_logo_path, width=150)
    else:
        st.warning(f"Image not found: {team_logo_path}")
    
    stats = calculate_team_stats(team, match_history)
    
    # Display Percentage-Based Performance (Progress Bars)
    st.subheader("⚔ Performance vs Higher & Lower ELO Teams")
    
    st.write("### 🆙 vs Higher ELO Teams")
    st.progress(stats["higher_win_pct"] / 100)
    st.write(f"🟢 Win Rate: **{stats['higher_win_pct']}%**")
    st.progress(stats["higher_draw_pct"] / 100)
    st.write(f"⚪ Draw Rate: **{stats['higher_draw_pct']}%**")
    st.progress(stats["higher_loss_pct"] / 100)
    st.write(f"🔴 Loss Rate: **{stats['higher_loss_pct']}%**")
    
    st.write("### 🆖 vs Lower ELO Teams")
    st.progress(stats["lower_win_pct"] / 100)
    st.write(f"🟢 Win Rate: **{stats['lower_win_pct']}%**")
    st.progress(stats["lower_draw_pct"] / 100)
    st.write(f"⚪ Draw Rate: **{stats['lower_draw_pct']}%**")
    st.progress(stats["lower_loss_pct"] / 100)
    st.write(f"🔴 Loss Rate: **{stats['lower_loss_pct']}%**")
    
    st.markdown("## Worst Enemies😈", unsafe_allow_html=True)
    
    worst_teams = stats["worst_opponents"]  # list of dicts
    cols = st.columns(len(worst_teams))
    for i, opp in enumerate(worst_teams):
        with cols[i]:
            render_stat_card(
                title=opp["Opponent"],
                subtitle=f"Win Rate: {opp['Win Rate']}%",
                gradient="red"
            )
    
    
    # Horizontal Scrolling Cards for Opponent-Based Performance
    st.subheader("🆚 Opponent-Based Performance")
    
    render_opponent_performance(
        best_opp=stats["best_opponent"]["Opponent"],
        best_record=f"{stats['best_opponent']['Win Rate']}% Win Rate",
        worst_opp=stats["worst_opponent"]["Opponent"],
        worst_record=f"{stats['worst_opponent']['Win Rate']}% Win Rate"
    )

    render_stat_card(
        title="🔁 Most Played Opponent",
        subtitle=f"{stats['most_played_opponent']['Opponent']} ({stats['most_played_opponent']['Win Rate']}% Win Rate)",
        gradient="gray"
    )

def show_about():
    st.markdown("""
    <style>
        @media (max-width: 768px) {
            h1 { font-size: 2em !important; }
            h3 { font-size: 1.3em !important; }
            p, li { font-size: 0.95em !important; }
        }
    </style>
    <div style="text-align: center; padding-top: 20px;">
        <h1 style="font-size: 3em; color: #39FF14;">What is HistoPulse™?</h1>
        <p style="color: #CCCCCC; font-size: 1.2em;">
            HistoPulse™ is our in-house performance rating — tracking the rise, fall, and legacy of every Premier League club.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background: linear-gradient(135deg, #1f1f1f, #2a2a2a); padding: 25px; border-radius: 16px; color: white; margin-top: 20px;">
        <h3>📊 How Does the HistoPulse™ System Work?</h3>
        <ul style="font-size: 15px; line-height: 1.8;">
            <li><b>Pulse Rating™:</b> Where a team stands right now.</li>
            <li><b>Pulse Shift™:</b> The rise or fall caused by a single match.</li>
            <li><b>Pulse Momentum™:</b> A team's recent trajectory over time.</li>
            <li><b>Legacy Change™:</b> How far a team has climbed or fallen from its historical highs.</li>
            <li><b>Match Impact™:</b> How influential a match was to a club's HistoPulse journey.</li>
        </ul>
        <p style="margin-top: 20px; font-size: 15px;">
            Our goal is to turn raw numbers into living, breathing stories. Histo isn’t just analytics — it’s football’s emotional rhythm.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top: 40px;">
        <h3 style="color: #39FF14;">📉 What HistoPulse™ is <u>NOT</u>:</h3>
        <ul style="font-size: 15px; color: #CCCCCC;">
            <li>❌ It’s not a betting model</li>
            <li>❌ It doesn’t guarantee future outcomes</li>
            <li>❌ It’s not about spreadsheets — it’s about story arcs</li>
            <li>❌ It doesn’t judge players — it tracks the journey of teams</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top: 40px;">
        <h3 style="color: #39FF14;">📈 Team Example: Arsenal (2022–23)</h3>
        <p style="font-size: 15px; color: #CCCCCC;">
            Arsenal started the season with a Pulse Rating™ of 1650. A winning streak gave them a Pulse Surge™ peaking at 1790 by March. A late-season drop — including a draw to Southampton and a loss to Brighton — triggered a Pulse Drop™ of –28.3. Despite finishing 2nd, their Pulse Rating™ held steady at 1752, showing real momentum and legacy rebuild.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top: 40px;">
        <h3 style="color: #39FF14;">💚 Our Origin Story</h3>
        <p style="font-size: 15px; color: #CCCCCC;">
            Histo was born out of a deep love for football stories — of heroes, champions, friendships, family, expectations, dreams, and banter. We believe that football is more than numbers and tactics. It's an emotional rollercoaster — a pulse. Every rise and fall in a club’s journey should feel like the next page in a story. That’s what HistoPulse™ captures.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.image("Assets/pulse_diagram.png", caption="Sample HistoPulse™ Journey (Visualized)", use_container_width=True)

    st.subheader("📂 Data & Sources")
    st.markdown("""
    - Match results sourced from official and publicly available Premier League data.
    - Ratings based on a football-adapted version of the ELO system.
    - Match impact logic fine-tuned using win margins, upsets, and contextual weight.
    """)

    st.subheader("📘 How to Read This App")
    st.markdown("""
    - Hover on ⚠️ icons or stats for more context.
    - “Pulse Rating™” shows current team performance level.
    - “Pulse Shift™” is the ELO-equivalent rise/fall per match.
    - “Legacy Change™” tracks progress since a defined point.
    - Explore the Key Moments tab for emotional peaks and pitfalls.
    """)

    st.success("Have feedback or want your club featured? Email us at contact@histopulse.com")

    st.markdown("""
    <div style='text-align: center; margin-top: 50px;'>
        <hr style='border: none; border-top: 1px solid #333; margin-bottom: 20px;' />
        <p style='color: #AAAAAA; font-size: 14px;'>Did you find this page helpful?</p>
        <p style='font-size: 20px;'>👍 👎</p>
    </div>
    """, unsafe_allow_html=True)




# Main function with improved structure
def main():
    import streamlit as st
    import streamlit.components.v1 as components
    
    # Set up the page
    setup_page()
    create_header()
    
    if st.session_state.get("screen_width", 1000) < 1000:
        cols = st.columns(2)
    else:
        cols = st.columns(4)

    loader_placeholder = st.empty()
    lp = st.empty()
    loader_placeholder.markdown("""
    <style>
    .loader-card {
        --bg-color: #3a125b;
        background: linear-gradient(135deg, #123524, #284431);
        padding: 1rem 2rem;
        border-radius: 1.25rem;
        box-shadow: 0px 4px 8px rgba(0,0,0,0.2);
        width: 100%;
        max-width: 300px;
    }
    .loader {
        color: rgb(124, 124, 124);
        font-family: "Poppins", sans-serif;
        font-weight: 500;
        font-size: 25px;
        box-sizing: content-box;
        height: 50px;
        padding: 10px 10px;
        display: flex;
        border-radius: 8px;
    }
    .words {
        overflow: hidden;
        position: relative;
        height: 40px;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
    }
    .words::after {
        content: "";
        position: absolute;
        background: var(--bg-color)
        inset: 0;
        z-index: 20;
    }
    .word {
        display: block;
        height: 100%;
        padding-left: 5px;
        color: #FFFFFF;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
        animation: spin_4991 7s infinite ease-in-out;
    }
    @keyframes spin_4991 {
        0%   { transform: translateY(0%); }  /* scroll to ELO */
        25%  { transform: translateY(-100%); }  /* scroll to Teams */
        50%  { transform: translateY(-200%); }  /* scroll to Rivalries */
        75%  { transform: translateY(-300%); }  /* scroll to Wins */
        100%  { transform: translateY(-400%); }  /* scroll to Losses */
    }


    </style>
    """, unsafe_allow_html=True)

    # Show loader
    lp.html("""
    <div class="loader-card">
    <div class="loader">
        <p style="color:#C0C0C0; font-size: 25px; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);">Loading</p>
        <div class="words">
        <span class="word">ELO</span>
        <span class="word">Legends</span>
        <span class="word">Legacy</span>
        <span class="word">Glory</span>
        <span class="word">Drama</span>
        </div>
    </div>
    </div>
    """)
       
    df = load_data()
        
    if df.empty:
        st.error("No data available. Please check the data file.")
        return
        
    team_history, season_final_elo, match_history, team_seasons = calculate_elo(df)
    
    loader_placeholder.empty()
    lp.empty()
        
    # Get unique seasons
    seasons = sorted(df["Season"].unique())
    
        # Single persistent selection above tabs
    if "selected_team_global" not in st.session_state:
        st.session_state["selected_team_global"] = sorted(team_history.keys())[0]

    selected_team = st.selectbox(
        "Team", 
        sorted(team_history.keys()), 
        index=sorted(team_history.keys()).index(st.session_state["selected_team_global"]),
        key="team_selector_main"
    )

    # Save selection persistently
    st.session_state["selected_team_global"] = selected_team

    tab_analysis, tab_season, tab_club, tab_about = st.tabs([
        "📈 Team Analysis",
        "🏆 Season Overview",
        "⚽ Club Performance",
        "ℹ️ About"
    ])

    with tab_analysis:
        available_seasons = sorted(team_history[selected_team].keys())
        selected_season = st.selectbox(
            "Select a Season",
            available_seasons,
            index=len(available_seasons)-1,
            key="analysis_selected_season"
        )
        settings = {"team": selected_team, "season": selected_season, "multi_season": False}
        
        if selected_season in team_history[selected_team]:
            show_team_analysis(team_history, settings, match_history)
        else:
            st.warning(f"{selected_team} did not play in {selected_season}.")

    with tab_season:
        selected_season = st.selectbox(
            "Select Season", seasons, index=len(seasons)-1, key="overview_selected_season"
        )
        display_mode = st.radio(
            "Display Mode", ["Chart", "Table", "Both"], index=2, key="display_mode"
        )
        settings = {"season": selected_season, "display_mode": display_mode}
        show_season_overview(season_final_elo, settings)

    with tab_club:
        settings = {"team": selected_team}
        show_club_dashboard(match_history, settings)

    with tab_about:
        show_about()
    # Add footer with version info
    st.markdown("""
    <div style="text-align: center; margin-top: 30px; padding: 10px; color: #666; font-size: 0.8em;">
        Premier League ELO Tracker v1.0 | Created 2025
    </div>
    """, unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()