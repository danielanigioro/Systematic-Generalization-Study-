import streamlit as st, json, time
st.title("Commands → Actions (Human Study)")
cmd = st.text("walk left twice and jump")
options = ["LTURN WALK LTURN WALK JUMP", "JUMP WALK", "WALK LTURN WALK", "LOOK"]
choice = st.radio("Pick the correct action sequence:", options)
if st.button("Submit"):
    # TODO: write anonymized CSV row with timestamp
    st.success("Recorded — thank you!")

