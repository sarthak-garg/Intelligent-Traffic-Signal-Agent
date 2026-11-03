# Intelligent Traffic Signal Agent

A Python + Streamlit AI subject project that simulates multi-junction traffic and compares fixed-time signals with a utility-based intelligent agent.

## Features
- Four approaches (North, East, South, West) at each junction
- Normal and peak traffic modes
- Fixed-time baseline controller
- Intelligent rule-based agent with candidate-action utility search
- Queue length and accumulated waiting-time scoring
- Emergency vehicle priority
- A* emergency route demonstration on a weighted road graph
- Interactive charts and downloadable CSV results

## Requirements
- Python 3.9 or newer
- pip

## Setup and run (Windows)
1. Extract the ZIP file.
2. Open the extracted `intelligent_traffic_signal_agent` folder in VS Code.
3. Open Terminal in VS Code.
4. Create a virtual environment:

   ```bash
   python -m venv venv
   ```

5. Activate it:

   ```bash
   venv\\Scripts\\activate
   ```

6. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

7. Start the app:

   ```bash
   streamlit run app.py
   ```

8. Your browser should open the dashboard at `http://localhost:8501`.

## How to demonstrate
1. Select **Normal** traffic and click **Run simulation**.
2. Note the average waiting time and queue charts for both controllers.
3. Select **Peak** traffic and run again.
4. Open **Junction details** to show direction-wise queues and signal state.
5. Open **Emergency A* route** to show the selected route and route cost.
6. Download the CSVs if you need to include results in your report.

## Project model
At each time step, vehicles may arrive independently on each approach. The fixed controller changes direction after a fixed green interval. The intelligent controller evaluates the candidate green directions using:

`Utility(direction) = 8 × queue_length + 0.35 × accumulated_wait + emergency_bonus`

An emergency bonus of 1000 is applied when an emergency vehicle is waiting in that direction. The controller also observes minimum and maximum green durations to reduce rapid switching and avoid starving an approach.

A* uses a weighted graph. The current heuristic is a reverse-graph hop-count estimate, which is admissible when edge costs are at least one time unit.

## Notes / limitations
This is an educational simulation, not a production traffic-control system. It models simplified independent arrivals and one vehicle served per green approach per time step. A real deployment would require calibrated traffic data, pedestrian phases, yellow/all-red clearance intervals, conflict matrices, sensors, fail-safe controllers, and safety validation.
