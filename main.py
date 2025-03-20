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

# Constants
INITIAL_ELO = 1000
K_FACTOR = 40
AWAY_WIN_BONUS = 1.1
LATE_SEASON_WEIGHT = 1.2
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
# Improved CSS with accessibility considerations
def inject_css():
    st.markdown("""
    <style>
    /* Main styles */
    .main {
        background-color: #f9f9f9;
        color: #333;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Header styles */
    h1, h2, h3 {
        color: #0e1e5b;
        font-weight: 600;
    }
    
    /* Card-like containers */
    .stCard {
        background-color: white;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    
    /* Team cards */
    .team-card {
        padding: 10px;
        border-radius: 5px;
        margin: 5px;
        font-weight: 500;
        text-align: center;
        transition: transform 0.2s;
    }
    .team-card:hover {
        transform: scale(1.05);
    }
    
    /* Focus indicators for accessibility */
    *:focus {
        outline: 3px solid #4b92db !important;
        outline-offset: 2px !important;
    }
    
    /* Custom tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 5px 5px 0px 0px;
        padding: 10px 20px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e0e4ee;
        border-bottom: 3px solid #0e1e5b;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        background-color: #0e1e5b;
        color: white;
        font-weight: 500;
        padding: 10px 15px;
    }
    .stButton>button:hover {
        background-color: #1d3a8a;
    }
    
    /* Color scale for visual consistency */
    .low-elo {
        color: #e74c3c;
    }
    .medium-elo {
        color: #f39c12;
    }
    .high-elo {
        color: #2ecc71;
    }
    
    /* Improved selectbox */
    div[data-baseweb="select"] {
        margin-bottom: 10px;
    }
    
    /* Improved spinner */
    .stSpinner {
        text-align: center;
        margin: 20px 0;
    }
    
    /* High contrast mode toggle */
    .high-contrast-toggle {
        position: fixed;
        top: 10px;
        right: 10px;
        z-index: 1000;
    }
    
    /* High contrast mode styles */
    .high-contrast {
        background-color: #000 !important;
        color: #fff !important;
    }
    .high-contrast h1, .high-contrast h2, .high-contrast h3 {
        color: #fff !important;
    }
    .high-contrast .stCard {
        background-color: #222 !important;
        color: #fff !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
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

# Calculate expected score
def get_expected_score(elo_a, elo_b):
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

# Update ELO ratings
def update_elo(winner_elo, loser_elo, goal_diff, is_away, late_season):
    weight = (goal_diff / 2) + (AWAY_WIN_BONUS if is_away else 1)
    if late_season:
        weight *= LATE_SEASON_WEIGHT
    expected_winner = get_expected_score(winner_elo, loser_elo)
    change = K_FACTOR * weight * (1 - expected_winner)
    return winner_elo + change, loser_elo - change

# Calculate ELO for every match and season
def calculate_elo(df):
    logger.debug("calculate_elo function called")
    
    team_history = {}
    season_final_elo = {}
    previous_season_elo = {}
    match_history = []
    team_seasons = {} # Track which seasons each team has played in

    seasons = sorted(df["Season"].unique())
    for season in seasons:
        season_df = df[df["Season"] == season]
        teams_in_season = pd.concat([season_df["HomeTeam"], season_df["AwayTeam"]]).unique()
        
        # Find teams that are new this season
        new_teams = [team for team in teams_in_season if team not in previous_season_elo]
        
        # Initialize ELO ratings for this season
        for team in teams_in_season:
            if team not in team_history:
                team_history[team] = {}
            starting_elo = previous_season_elo.get(team, INITIAL_ELO) if team not in new_teams else INITIAL_ELO
            team_history[team][season] = [(0, starting_elo)]
            
            # Track team seasons
            if team not in team_seasons:
                team_seasons[team] = []
            team_seasons[team].append(season)  # Add the season to the team's list

        
        # Loop through each match
        season_matches = season_df.reset_index(drop=True)
        total_matches = len(season_matches)
        for i, match in season_matches.iterrows():
            home, away, result = match["HomeTeam"], match["AwayTeam"], match["FTR"]
            home_goals, away_goals = match["FTHG"], match["FTAG"]
            match_date = match["DateTime"]
            late_season = i > (total_matches / 2)
            
            current_home_elo = team_history[home][season][-1][1]
            current_away_elo = team_history[away][season][-1][1]
            
            if result == "H":  # Home win
                goal_diff = home_goals - away_goals
                new_home_elo, new_away_elo = update_elo(current_home_elo, current_away_elo, goal_diff, False, late_season)
            elif result == "A":  # Away win
                goal_diff = away_goals - home_goals
                new_away_elo, new_home_elo = update_elo(current_away_elo, current_home_elo, goal_diff, True, late_season)
            else:  # Draw
                expected_home = get_expected_score(current_home_elo, current_away_elo)
                change_home = K_FACTOR * (0.5 - expected_home)
                new_home_elo = current_home_elo + change_home
                new_away_elo = current_away_elo - change_home
            
            # Save updated ELOs
            team_history[home][season].append((i+1, new_home_elo))
            team_history[away][season].append((i+1, new_away_elo))
            
            # Store detailed match history
            match_history.append({
                'Season': season,
                'Date': match_date,  # Ensure this is in datetime format
                'HomeTeam': home,
                'AwayTeam': away,
                'HomeGoals': home_goals,
                'AwayGoals': away_goals,
                'Result': result,
                'HomeElo_Before': current_home_elo,
                'AwayElo_Before': current_away_elo,
                'HomeElo_After': new_home_elo,
                'AwayElo_After': new_away_elo,
                'HomeElo_Change': new_home_elo - current_home_elo,
                'AwayElo_Change': new_away_elo - current_away_elo
            })

        
        # Store final ELOs for next season
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
    """Creates a structured and aligned header with the Premier League logo and title."""
    col1, col2 = st.columns([1, 5])
    with col1:
        logo_path = "Assets/premier_league_logo.png"
        if os.path.exists(logo_path):
            st.image(logo_path, width=550)  # Adjusted for better balance
    with col2:
        st.markdown(
            f"""
            <div style="display: flex; flex-direction: column; justify-content: center; height: 100%;">
                <h1 style="color:#FFFFFF; margin-bottom: 5px;">Premier League ELO Tracker</h1>
            </div>
            """, unsafe_allow_html=True
        )

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

        if opp_total >= 10:  # Ensure at least 10 matches played
            opponent_stats.append({"Opponent": opp, "Matches Played": opp_total, "Win Rate": win_rate})

    opponent_stats_df = pd.DataFrame(opponent_stats)
    best_opponent = opponent_stats_df.loc[opponent_stats_df["Win Rate"].idxmax()] if not opponent_stats_df.empty else None
    worst_opponent = opponent_stats_df.loc[opponent_stats_df["Win Rate"].idxmin()] if not opponent_stats_df.empty else None
    most_played_opponent = opponent_stats_df.loc[opponent_stats_df["Matches Played"].idxmax()] if not opponent_stats_df.empty else None

    return {
        "higher_win_pct": higher_win_pct,
        "higher_draw_pct": higher_draw_pct,
        "higher_loss_pct": higher_loss_pct,
        "lower_win_pct": lower_win_pct,
        "lower_draw_pct": lower_draw_pct,
        "lower_loss_pct": lower_loss_pct,
        "best_opponent": best_opponent,
        "worst_opponent": worst_opponent,
        "most_played_opponent": most_played_opponent
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
            <div style="{category_styles[category]} padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 10px;">
                <h4>{category} Teams</h4>
                <p>{", ".join(teams) if teams else "No teams in this category"}</p>
            </div>
            """, unsafe_allow_html=True)


# Enhanced team analysis view
def show_team_analysis(team_history, settings, match_history):
    team = settings["team"]
    multi_season = settings["multi_season"]

    st.markdown(
        f"""
        <h2 style="color:#FFFFFF;">{team} ELO Performance</h2>
        """, unsafe_allow_html=True
    )

    # Layout: Display team logo next to team name with a styled background
    col1, col2 = st.columns([1, 4])

    with col1:
        team_logo_path = f"Assets/team_logos/{team}.png"
        if os.path.exists(team_logo_path):
            st.image(team_logo_path, width=150)
        else:
            st.warning(f"Image not found: {team_logo_path}")

    # ELO Progression Graph
    if not multi_season:
        season = settings["season"]
        st.markdown(f"<h4 style='color:#FFFFFF;'>ELO Rating in {season}</h4>", unsafe_allow_html=True)

        fig = plot_elo_progression(team, season, team_history, match_history)  # Step chart function

        if fig:
            st.plotly_chart(fig, use_container_width=True)

    else:
        seasons = settings["seasons"]
        st.markdown(f"<h4 style='color:#FFFFFF;'>ELO Comparison Across Seasons</h4>", unsafe_allow_html=True)

        # Multi-Season Graph
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

        st.plotly_chart(fig, use_container_width=True)

    # Key Matches (Only if not multi-season)
    if not multi_season:
        st.markdown(
            f"""
            <h3 style="color:#FFFFFF;">Key Matches</h3>
            """, unsafe_allow_html=True
        )

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
            biggest_gain_match = team_matches.iloc[biggest_gain_idx - 1]  # Get match details
            biggest_loss_match = team_matches.iloc[biggest_loss_idx - 1]

            # Determine opponent, match location, and score
            gain_opponent = biggest_gain_match["AwayTeam"] if biggest_gain_match["HomeTeam"] == team else biggest_gain_match["HomeTeam"]
            gain_location = "Home vs" if biggest_gain_match["HomeTeam"] == team else "Away at"
            gain_date = biggest_gain_match["Date"].strftime('%d %b %Y')
            gain_score = f"{biggest_gain_match['HomeGoals']} - {biggest_gain_match['AwayGoals']}"

            loss_opponent = biggest_loss_match["AwayTeam"] if biggest_loss_match["HomeTeam"] == team else biggest_loss_match["HomeTeam"]
            loss_location = "Home vs" if biggest_loss_match["HomeTeam"] == team else "Away at"
            loss_date = biggest_loss_match["Date"].strftime('%d %b %Y')
            loss_score = f"{biggest_loss_match['HomeGoals']} - {biggest_loss_match['AwayGoals']}"

            # Display Key Matches in a styled format
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #065F46, #0F766E);
                padding: 20px; 
                border-radius: 12px; 
                box-shadow: 0px 4px 8px rgba(0,0,0,0.2);
                text-align:center;">
                <h4 style="color:#39FF14; margin-bottom:10px;">Biggest ELO Gain</h4>
                <p style="color:#FFFFFF; font-size:16px;"><b>{gain_location} {gain_opponent} ({gain_date})</b></p>
                <p style="color:#FFFFFF; font-size:18px;">Final Score: <b>{gain_score}</b></p>
                <p style="color:#39FF14; font-size:20px; font-weight:bold;">ELO Change: +{round(elos[biggest_gain_idx] - elos[biggest_gain_idx-1], 1)}</p>
            </div>
            """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #7F1D1D, #B91C1C);
                padding: 20px; 
                border-radius: 12px; 
                box-shadow: 0px 4px 8px rgba(0,0,0,0.2);
                text-align:center;">
                <h4 style="color:#FF6B6B; margin-bottom:10px;">Biggest ELO Loss</h4>
                <p style="color:#FFFFFF; font-size:16px;"><b>{loss_location} {loss_opponent} ({loss_date})</b></p>
                <p style="color:#FFFFFF; font-size:18px;">Final Score: <b>{loss_score}</b></p>
                <p style="color:#FF6B6B; font-size:20px; font-weight:bold;">ELO Change: {round(elos[biggest_loss_idx] - elos[biggest_loss_idx-1], 1)}</p>
            </div>
            """, unsafe_allow_html=True)


def show_season_overview(season_final_elo, settings):
    season = settings["season"]

    st.header(f"🏆 Final ELO Rankings: {season}")

    # Get rankings and team categories
    elite_teams, contender_teams, mid_table_teams, relegation_teams, rankings = get_season_overview(season_final_elo, season)

    if rankings.empty:
        st.warning(f"No ranking data available for {season}. Check the data processing pipeline.")
        return

    # Display ELO-based category heading
    st.subheader("🔢 Team Categories Based on ELO Rankings")

    # Display categorized teams as gradient cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #FFD700, #FFC107); padding: 20px; border-radius: 12px; text-align:center; height: 200px; display: flex; flex-direction: column; justify-content: center;">
            <h4 style="color:#000000;">🥇 Elite Teams</h4>
            <p style="color: black; font-size: 14px;">{', '.join(elite_teams) if elite_teams else 'No teams'}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1E90FF, #4682B4); padding: 20px; border-radius: 12px; text-align:center; height: 200px; display: flex; flex-direction: column; justify-content: center;">
            <h4 style="color:#FFFFFF;">💪 Contender Teams</h4>
            <p style="color: white; font-size: 14px;">{', '.join(contender_teams) if contender_teams else 'No teams'}</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #31b6b0, #20B2AA); padding: 20px; border-radius: 12px; text-align:center; height: 200px; display: flex; flex-direction: column; justify-content: center;">
            <h4 style="color:#FFFFFF;">⚖️ Mid-Table Teams</h4>
            <p style="color: white; font-size: 12px;">{', '.join(mid_table_teams) if mid_table_teams else 'No teams'}</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #FF4C4C, #B22222); padding: 20px; border-radius: 12px; text-align:center; height: 200px; display: flex; flex-direction: column; justify-content: center;">
            <h4 style="color:#FFFFFF;">📉 Relegation Teams</h4>
            <p style="color: white; font-size: 14px;">{', '.join(relegation_teams) if relegation_teams else 'No teams'}</p>
        </div>
        """, unsafe_allow_html=True)


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

    st.plotly_chart(fig, use_container_width=True)


    # Colored season rankings visualization

def show_club_dashboard(match_history, settings):
    st.header(f"📊 {settings['team']} Performance Dashboard")

    team = settings["team"]
    team_matches = match_history[(match_history["HomeTeam"] == team) | (match_history["AwayTeam"] == team)]

    team_logo_path = f"Assets/team_logos/{team}.png"
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

    # Display Opponent-Based Performance
    st.subheader("🆚 Opponent-Based Performance")
    
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        best_logo = get_team_logo_path(stats["best_opponent"]["Opponent"])
        best_logo_base64 = encode_image(best_logo)
        st.markdown(f"""
        <div style="background-color: #003366; padding: 15px; border-radius: 8px; text-align: center;">
            <img src="data:image/png;base64,{best_logo_base64}" width="60" style="border-radius: 5px; margin-bottom: 8px;">
            <h4 style="color: #FFD700;">🏆 Best Opponent</h4>
            <p style="color: white; font-size: 18px;"><b>{stats['best_opponent']['Opponent']}</b></p>
            <p style="color: white;">Win Rate: <b>{stats['best_opponent']['Win Rate']}%</b></p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        worst_logo = get_team_logo_path(stats["worst_opponent"]["Opponent"])
        worst_logo_base64 = encode_image(worst_logo)
        st.markdown(f"""
        <div style="background-color: #660000; padding: 15px; border-radius: 8px; text-align: center;">
            <img src="data:image/png;base64,{worst_logo_base64}" width="60" style="border-radius: 5px; margin-bottom: 8px;">
            <h4 style="color: #FFD700;">📉 Worst Opponent</h4>
            <p style="color: white; font-size: 18px;"><b>{stats['worst_opponent']['Opponent']}</b></p>
            <p style="color: white;">Win Rate: <b>{stats['worst_opponent']['Win Rate']}%</b></p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        most_played_logo = get_team_logo_path(stats["most_played_opponent"]["Opponent"])
        most_played_logo_base64 = encode_image(most_played_logo)
        st.markdown(f"""
        <div style="background-color: #444444; padding: 15px; border-radius: 8px; text-align: center;">
            <img src="data:image/png;base64,{most_played_logo_base64}" width="60" style="border-radius: 5px; margin-bottom: 8px;">
            <h4 style="color: #FFD700;">🔄 Most Played Opponent</h4>
            <p style="color: white; font-size: 18px;"><b>{stats['most_played_opponent']['Opponent']}</b></p>
            <p style="color: white;">Matches Played: <b>{stats['most_played_opponent']['Matches Played']}</b></p>
        </div>
        """, unsafe_allow_html=True)


def show_about():
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 15px;">
            <h1 style="margin: 0;">About Histo - Premier League ELO Tracker</h1>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div style="background: linear-gradient(135deg, #1e3c72, #2a5298); padding: 20px; border-radius: 10px; color: white; text-align: center;">
            <h3>Welcome to Histo - The Premier League ELO Tracker!</h3>
            <p>This application provides a detailed historical analysis of Premier League teams using the ELO rating system. 
               The ELO rating is a dynamic metric that evaluates team strength based on match results and opponent strength.</p>
        </div>
    """, unsafe_allow_html=True)

    st.subheader("How the ELO System Works")
    st.markdown("""
        - **Starting Point:** All teams begin with a base ELO rating of 1000.
        - **Match Results:** ELO ratings update after every match, rewarding wins and penalizing losses.
        - **Opponent Strength:** Wins against stronger teams result in larger ELO gains.
        - **Modifiers:** Late-season matches and away wins have added weight.
    """)

    # Features Overview with Gradient Cards
    st.subheader("Application Features")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #065F46, #0F766E); padding: 20px; border-radius: 12px; box-shadow: 0px 4px 8px rgba(0,0,0,0.2); text-align:center;">
            <h4 style="color:#39FF14; margin-bottom:10px;">📊 Team Analysis</h4>
            <p style="color: white; font-size: 14px;">Track ELO progression of individual teams across seasons.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #7F1D1D, #B91C1C); padding: 20px; border-radius: 12px; box-shadow: 0px 4px 8px rgba(0,0,0,0.2); text-align:center;">
            <h4 style="color:#FF6B6B; margin-bottom:10px;">🏆 Season Overview</h4>
            <p style="color: white; font-size: 14px;">View how teams rank at the end of each season based on ELO.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #4B0082, #800080); padding: 20px; border-radius: 12px; box-shadow: 0px 4px 8px rgba(0,0,0,0.2); text-align:center;">
            <h4 style="color:#DDA0DD; margin-bottom:10px;">📜 Club Dashboard</h4>
            <p style="color: white; font-size: 14px;">Analyze your club's Premier League tenure, best and worst opponents, and more.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Accessibility Section
    st.subheader("Accessibility & Customization")
    st.markdown("""
        - **High Contrast Mode:** Improve visibility for users with visual impairments.
        - **Text Size Adjustment:** Customize the interface for better readability.
        - **Keyboard Navigation:** Full keyboard accessibility support.
    """)


# Main function with improved structure
def main():
    # Set up the page
    setup_page()
    
    # Create header
    create_header()
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
        
        if df.empty:
            st.error("No data available. Please check the data file.")
            return
        
        # Calculate ELO
        with st.spinner("Calculating ELO ratings..."):
            team_history, season_final_elo, match_history, team_seasons = calculate_elo(df)
        
        # Get unique seasons
        seasons = sorted(df["Season"].unique())
    
    # Create sidebar and get navigation choices
    mode, settings = create_sidebar(team_history, seasons)
    
    # Display the appropriate view based on mode
    if mode == "Team Analysis":
        show_team_analysis(team_history, settings, match_history)
    elif mode == "Season Overview":
        show_season_overview(season_final_elo, settings)
    elif mode == "Club Dashboard":
        show_club_dashboard(match_history, settings)
    else:  # About
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