"""
app.py  -  Streamlit demo for the panel.
Run:  streamlit run app.py
Three user stories, one tab each:
  1. As a plant manager, I want OEE per station so I can see where efficiency is lost.
  2. As a maintenance engineer, I want a failure-risk score for a machine reading so I can act before it breaks.
  3. As a finance lead, I want to see the rupee value of cutting downtime so I can justify the investment.
"""

import streamlit as st                                  # Streamlit turns a Python script into a web app
from src import kpi                                     # our SQL KPI functions
from src.db import DB_PATH, load_csvs_to_db             # database setup helpers
from src.model import MODEL_PATH, predict_one, train_and_save

st.set_page_config(page_title="Smart Factory Insights", layout="wide")   # browser tab title + full-width layout
st.title("Smart Factory Insights")                                       # big heading at the top


@st.cache_resource                                      # run once per app session, not on every click
def setup():
    """Make sure the database and model exist before any tab uses them."""
    if not DB_PATH.exists():                            # first run: build factory.db from the CSVs
        load_csvs_to_db()
    if not MODEL_PATH.exists():                         # first run: train and save the model
        train_and_save()
    return True


setup()
tab_oee, tab_risk, tab_save = st.tabs(["Line efficiency (OEE)", "Failure risk", "Savings what-if"])

# ---------- User story 1: OEE ----------
with tab_oee:                                           # everything indented here appears inside the first tab
    line = st.selectbox("Line", ["All", "LINE_A", "LINE_B"])            # dropdown
    oee = kpi.oee_by_station(None if line == "All" else line)          # "All" -> None -> no filter in SQL
    worst = oee.loc[oee["oee"].idxmin()]                                # row with the lowest OEE
    bottleneck = kpi.find_bottleneck(oee)
    c1, c2, c3 = st.columns(3)                                          # three KPI tiles side by side
    c1.metric("Average OEE", f"{oee['oee'].mean():.1%}")                # :.1% formats 0.812 as 81.2%
    c2.metric("Lowest-OEE station", f"{worst['station']}", f"{worst['oee']:.1%}")
    c3.metric("Bottleneck (slowest)", ", ".join(bottleneck["station"].unique()))
    st.bar_chart(oee.set_index(oee["line_id"] + " " + oee["station"])["oee"])   # one bar per line+station
    st.dataframe(oee, width="stretch")                         # full table for anyone who wants detail
    st.subheader("Downtime by shift")
    st.dataframe(kpi.downtime_by_shift(), width="stretch")

# ---------- User story 2: failure risk ----------
with tab_risk:
    st.write("Enter a machine reading to get its failure probability.")
    col1, col2 = st.columns(2)
    with col1:
        machine_type = st.selectbox("Machine type", ["L", "M", "H"])
        air = st.number_input("Air temperature (K)", 290.0, 310.0, 300.0)       # (label, min, max, default)
        proc = st.number_input("Process temperature (K)", 300.0, 320.0, 310.0)
    with col2:
        rpm = st.number_input("Rotational speed (rpm)", 1000, 3000, 1500)
        torque = st.number_input("Torque (Nm)", 0.0, 90.0, 40.0)
        wear = st.number_input("Tool wear (min)", 0, 300, 100)
    if proc <= air:                                                      # edge case: invalid physics input
        st.warning("Process temperature is normally above air temperature; please check the reading.")
    if st.button("Predict"):                                             # code below runs only when clicked
        result = predict_one({"machine_type": machine_type, "air_temp_k": air, "process_temp_k": proc,
                              "rpm": rpm, "torque_nm": torque, "tool_wear_min": wear})
        if result["alert"]:
            st.error(f"High risk: {result['failure_probability']:.0%}. Schedule maintenance now.")
        else:
            st.success(f"Normal: {result['failure_probability']:.0%} risk (alert threshold {result['threshold']:.0%}).")

# ---------- User story 3: savings ----------
with tab_save:
    st.write("Estimate monthly savings if downtime is reduced (for example through predictive maintenance).")
    reduction = st.slider("Downtime reduction (%)", 0, 50, 20)                    # slider from 0 to 50, default 20
    cost = st.number_input("Cost of one minute of downtime (Rs, assumption)", 0, 10000, 500)
    s = kpi.downtime_saving(reduction, cost)
    c1, c2, c3 = st.columns(3)
    c1.metric("Downtime this month", f"{s['monthly_downtime_min']:,.0f} min")     # :, adds thousand separators
    c2.metric("Minutes saved", f"{s['minutes_saved']:,.0f}")
    c3.metric("Rupees saved / month", f"Rs {s['rupees_saved']:,.0f}")
    st.caption("Cost per minute is an input assumption; replace it with the plant's real figure.")
