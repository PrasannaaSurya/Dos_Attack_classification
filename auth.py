import streamlit as st
import json
import os

# File paths
USERS_FILE = "users.json"
STATS_FILE = "stats.json"

# Load users from file
def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return {}

# Save users to file
def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

# Load stats from file
def load_stats():
    if os.path.exists(STATS_FILE):
        if os.stat(STATS_FILE).st_size == 0:  # Check if the file is empty
            return {"total_attacks": 0}
        with open(STATS_FILE, "r") as f:
            return json.load(f)
    return {"total_attacks": 0}


# Save stats to file
def save_stats(stats):
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f)

# Set background
def set_background(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
        }}
        .card {{
            background-color: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0px 4px 8px rgba(0,0,0,0.1);
            color: black;
            font-size: 18px;
            font-weight: bold;
            text-align: center;
            margin: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Display Stats (Total Users and Attacks)
def display_stats():
    users = load_users()
    stats = load_stats()

    total_users = len(users)
    total_attacks = stats.get("total_attacks", 0)

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f'<div class="card">Total Registered Users<br>{total_users}</div>', unsafe_allow_html=True)

    with col2:
        st.markdown(f'<div class="card">Total Classified Attacks<br>{total_attacks}</div>', unsafe_allow_html=True)

# Login Page
def login_page():
    set_background("https://img.freepik.com/free-vector/scifi-inspired-health-care-background-protect-heart-with-cardio_1017-57723.jpg?ga=GA1.1.2116394528.1741097605")
    st.title("Login Page")

    display_stats()

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    users = load_users()

    if st.button("Login"):
        if username in users and users[username] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.experimental_set_query_params(page="main")
            st.rerun()
        else:
            st.error("Invalid username or password")

    if st.button("Go to Register"):
        st.experimental_set_query_params(page="register")
        st.rerun()

# Register Page
def register_page():
    set_background("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSVmGOHA5LKyLY2SrWffaKyGjsM4-Lqbptu2w&s")
    st.title("Register Page")



    new_username = st.text_input("New Username")
    new_password = st.text_input("New Password", type="password")
    users = load_users()

    if st.button("Register"):
        if new_username in users:
            st.error("Username already exists!")
        else:
            users[new_username] = new_password
            save_users(users)
            st.success("Registration successful! Please login.")

    if st.button("Back to Login"):
        st.experimental_set_query_params(page="login")
        st.rerun()

# Logout Function
def logout():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.experimental_set_query_params(page="login")
    st.rerun()
