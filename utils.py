import streamlit as st
import base64

def load_stylesheet(path="Assets/style.css"):
    with open(path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def generate_youtube_search_link(team1: str, team2: str, score: str, date: str):
    query = f"{team1} vs {team2} {score} {date} Premier League highlights"
    return f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"


def encode_image(path):
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def render_stat_card(title, subtitle, gradient="gold-red", logo_path=None):
    gradients = {
        "gold-red": "linear-gradient(to left, #f7ba2b 0%, #ea5358 100%)",
        "green": "linear-gradient(to left, #4ADE80, #16A34A)",
        "blue": "linear-gradient(to left, #60A5FA, #3B82F6)",
        "gray": "linear-gradient(to left, #6B7280, #374151)",
        "red": "linear-gradient(to left, #f87171, #dc2626)",
        "black": "linear-gradient(to left, #1a1a1a 0%, #2e2e2e 100%)"
    }

    background = gradients.get(gradient, gradients["gold-red"])

    logo_html = ""
    if logo_path:
        try:
            encoded = encode_image(logo_path)
            logo_html = f"<img src='data:image/png;base64,{encoded}' style='width: 40px; margin-bottom: 10px;' />"
        except FileNotFoundError:
            pass

    st.markdown(f"""
    <div class="card {gradient}" style="--background: {background};">
      <div class="card-info">
        {logo_html}
        <div class="title">{title}</div>
        <div class="subtitle" style="text-align: center; color: #CCCCCC; font-size: 15px; line-height: 1.5;">
            {subtitle}
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

def render_highlight_card(opponent, score, date, location, tagline, title="⭐ Highlight Moment", gradient="gold-red", logo_path=None, elo_change=None, match_link="#"):
    gradients = {
        "gold-red": "linear-gradient(to left, #f7ba2b 0%, #ea5358 100%)",
        "green": "linear-gradient(to left, #4ADE80, #16A34A)",
        "blue": "linear-gradient(to left, #60A5FA, #3B82F6)",
        "gray": "linear-gradient(to left, #6B7280, #374151)",
        "red": "linear-gradient(to left, #f87171, #dc2626)",
        "black": "linear-gradient(to left, #1a1a1a 0%, #2e2e2e 100%)"
    }

    border_colors = {
        "gold-red": "#f7ba2b",
        "green": "#22c55e",
        "blue": "#3b82f6",
        "gray": "#9ca3af",
        "red": "#ef4444",
        "black": "#6b7280"
    }

    background = gradients.get(gradient, gradients["gold-red"])
    border_color = border_colors.get(gradient, "#f7ba2b")

    logo_html = ""
    if logo_path:
        try:
            encoded = encode_image(logo_path)
            logo_html = f"<img src='data:image/png;base64,{encoded}' style='width: 48px; margin-bottom: 10px;' />"
        except FileNotFoundError:
            pass

    pulse_html = ""
    if elo_change is not None:
        shift_direction = "Pulse Gain" if elo_change > 0 else "Pulse Drop" if elo_change < 0 else "No Change"
        color = "#4ADE80" if elo_change > 0 else "#ef4444" if elo_change < 0 else "#facc15"
        pulse_html = f"<div style='font-size: 13px; margin-top: 4px; color: {color};'>📊 <strong>{shift_direction}:</strong> {elo_change:+.2f} <span title='HistoPulse™ is our proprietary performance rating system.' style='cursor: help;'>ℹ️</span></div>"

    st.markdown(f"""
    <div class="card {gradient}" style="--background: {background}; padding: 25px 20px; border: 2px solid {border_color}; border-radius: 1rem;">
      <div class="card-info">
        {logo_html}
        <div class="title" style="font-size: 20px; margin-bottom: 10px;">{title}</div>
        <div style="text-align: center; font-size: 16px; font-weight: 600;">vs {opponent} ({location})</div>
        <div style="text-align: center; color: #BBBBBB; margin: 6px 0; font-size: 14px;">
            🗓️ {date} &nbsp;&nbsp;&nbsp; 🎯 {score}
        </div>
        <div class="stat-tag">{tagline}</div>
        {pulse_html}
        <div style='margin-top: 10px;'>
            <a href="{match_link}" target="_blank" style="font-size: 12px; color: #39FF14; text-decoration: underline;">🔍 See Full Match</a>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)





def render_team_category_card(label, teams, gradient="gray"):
    if not teams:
        teams = ["No teams"]
    team_text = ", ".join(teams)
    render_stat_card(title=label, subtitle=team_text, gradient=gradient)

def render_opponent_performance(best_opp, best_record, worst_opp, worst_record):
    cols = st.columns(2)
    with cols[0]:
        render_stat_card("📈 Best Record vs", f"{best_opp} ({best_record})", gradient="green")
    with cols[1]:
        render_stat_card("📉 Worst Record vs", f"{worst_opp} ({worst_record})", gradient="red")

def render_key_moments(gain, loss, draw):
    st.subheader("🏆 Key Moments of the Season")

    render_highlight_card(
        opponent=gain["Opponent"],
        score=gain["Score"],
        date=gain["Date"],
        location=gain["Location"],
        tagline="📈 Biggest Pulse Surge",
        title="💥 Match of the Season",
        gradient="green",
        logo_path=gain.get("Logo"),
        elo_change=gain.get("ELO"),
        match_link=gain.get("Link", "#")
    )

    cols = st.columns(2)
    with cols[0]:
        render_highlight_card(
            opponent=loss["Opponent"],
            score=loss["Score"],
            date=loss["Date"],
            location=loss["Location"],
            tagline="💔 Biggest Pulse Drop",
            title="😖 Toughest Loss",
            gradient="red",
            logo_path=loss.get("Logo"),
            elo_change=loss.get("ELO"),
            match_link=loss.get("Link", "#")
        )

    with cols[1]:
        render_highlight_card(
            opponent=draw["Opponent"],
            score=draw["Score"],
            date=draw["Date"],
            location=draw["Location"],
            tagline="🛡️ Held the Giants",
            title="⚖️ Closest Draw",
            gradient="gray",
            logo_path=draw.get("Logo"),
            elo_change=draw.get("ELO"),
            match_link=draw.get("Link", "#")
        )
