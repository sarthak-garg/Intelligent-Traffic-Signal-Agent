import streamlit as st
import pandas as pd
import plotly.express as px
from simulation import TrafficSimulation, emergency_demo_route

st.set_page_config(page_title="Intelligent Traffic Signal Agent", page_icon="🚦", layout="wide")
st.title("🚦 Intelligent Traffic Signal Agent")
st.caption("AI-based traffic signal simulation | Rule-based control, utility search, emergency priority and A* routing")

with st.sidebar:
    st.header("Simulation settings")
    traffic = st.selectbox("Traffic condition", ["Normal", "Peak"])
    intersections = st.slider("Number of junctions", 1, 6, 3)
    duration = st.slider("Simulation duration (time steps)", 60, 600, 240, 30)
    seed = st.number_input("Random seed", min_value=1, max_value=9999, value=7)
    emergency_rate = st.slider("Emergency vehicle probability", 0.0, 0.08, 0.012, 0.002)
    run = st.button("▶ Run simulation", type="primary", use_container_width=True)

st.markdown("""
The system simulates vehicles arriving from four directions at multiple junctions.
The **fixed-time controller** rotates through directions at a constant interval. The
**intelligent agent** evaluates candidate green signals using queue length, accumulated
waiting time and emergency-vehicle priority, then selects the highest-utility action.
""")

if run or "results" not in st.session_state:
    fixed = TrafficSimulation("fixed", traffic, seed, intersections, duration, emergency_rate)
    intelligent = TrafficSimulation("intelligent", traffic, seed, intersections, duration, emergency_rate)
    fixed_df = fixed.run()
    intelligent_df = intelligent.run()
    st.session_state.results = {
        "fixed_df": fixed_df, "intelligent_df": intelligent_df,
        "fixed_summary": fixed.summary(), "intelligent_summary": intelligent.summary(),
        "traffic": traffic
    }

r = st.session_state.results
fixed_df, intelligent_df = r["fixed_df"], r["intelligent_df"]
fixed_summary, intelligent_summary = r["fixed_summary"], r["intelligent_summary"]

st.subheader(f"Results — {r['traffic']} traffic")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Intelligent avg. wait", intelligent_summary["Average wait (ticks)"])
c2.metric("Fixed-time avg. wait", fixed_summary["Average wait (ticks)"])
improvement = 0
if fixed_summary["Average wait (ticks)"]:
    improvement = (fixed_summary["Average wait (ticks)"] - intelligent_summary["Average wait (ticks)"]) / fixed_summary["Average wait (ticks)"] * 100
c3.metric("Waiting-time improvement", f"{improvement:.1f}%")
c4.metric("Intelligent vehicles served", intelligent_summary["Vehicles served"])

summary_df = pd.DataFrame([
    {"Controller": "Fixed-time", **fixed_summary},
    {"Controller": "Intelligent agent", **intelligent_summary}
])
st.subheader("Controller comparison")
st.dataframe(summary_df, use_container_width=True, hide_index=True)

tab1, tab2, tab3 = st.tabs(["📈 Traffic statistics", "🚦 Junction details", "🚑 Emergency A* route"])
with tab1:
    combined = pd.concat([fixed_df, intelligent_df], ignore_index=True)
    by_time = combined.groupby(["time", "mode"], as_index=False).agg(
        average_queue=("queue_length", "mean"),
        average_waiting=("waiting_total", "mean")
    )
    a, b = st.columns(2)
    with a:
        fig = px.line(by_time, x="time", y="average_queue", color="mode",
                      title="Average queue length over time",
                      labels={"time": "Simulation time step", "average_queue": "Vehicles queued", "mode": "Controller"})
        st.plotly_chart(fig, use_container_width=True)
    with b:
        fig = px.line(by_time, x="time", y="average_waiting", color="mode",
                      title="Accumulated waiting time in queues",
                      labels={"time": "Simulation time step", "average_waiting": "Waiting time (ticks)", "mode": "Controller"})
        st.plotly_chart(fig, use_container_width=True)
    served = combined.groupby(["mode", "junction"], as_index=False)["vehicles_served"].max()
    fig = px.bar(served, x="junction", y="vehicles_served", color="mode", barmode="group",
                 title="Vehicles served by junction")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    selected = st.selectbox("Select controller", ["intelligent", "fixed"])
    data = intelligent_df if selected == "intelligent" else fixed_df
    junction = st.selectbox("Select junction", sorted(data["junction"].unique()))
    jdata = data[data["junction"] == junction]
    latest = jdata.iloc[-1]
    st.write(f"**Current green signal:** {latest['green']}  |  **Vehicles waiting:** {int(latest['queue_length'])}")
    qdf = pd.DataFrame({"Direction": ["North", "East", "South", "West"],
                        "Queue": [int(latest[f"queue_{d}"]) for d in ["North", "East", "South", "West"]]})
    fig = px.bar(qdf, x="Direction", y="Queue", title=f"Final queue lengths — {junction}")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(jdata.tail(12), use_container_width=True, hide_index=True)

with tab3:
    path, cost = emergency_demo_route()
    st.markdown("A* finds a minimum-cost route through the sample road network. The edge weights represent travel-time units.")
    st.success(" → ".join(path) + f"  |  Total route cost: {cost}")
    st.info("When an emergency vehicle is detected in a queue, the intelligent controller gives that approach a strong utility bonus and serves the emergency vehicle ahead of regular vehicles.")
    st.write("A* route output:", path)

st.subheader("How the AI agent works")
st.markdown("""
1. **Perception:** observe queue lengths, vehicle arrival times and emergency flags.
2. **Actions:** keep the current green or switch green to North, East, South or West.
3. **Rule-based safety:** maintain a minimum green duration to prevent unstable switching.
4. **Search:** score each candidate direction and choose the highest-utility direction.
5. **Utility:** queue length + accumulated waiting time + a large emergency priority bonus.
6. **Emergency routing:** A* searches the road graph for a low-cost path to the destination.
""")
st.download_button("Download intelligent simulation CSV", intelligent_df.to_csv(index=False), "intelligent_traffic_results.csv", "text/csv")
st.download_button("Download fixed-time simulation CSV", fixed_df.to_csv(index=False), "fixed_time_traffic_results.csv", "text/csv")
