import csv
import random
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

# ---------- page setup ----------
st.set_page_config(page_title="TV Scheduling GA", page_icon="📺", layout="wide")

CSV_PATH = Path(__file__).parent / "program_ratings.csv"


# ---------- data ----------
@st.cache_data
def load_ratings(path):
    program_ratings = {}
    with open(path, newline="") as file:
        reader = csv.reader(file)
        next(reader)  # skip header
        for row in reader:
            program_ratings[row[0]] = [float(x) for x in row[1:]]
    return program_ratings


ratings = load_ratings(CSV_PATH)
all_programs = list(ratings.keys())
all_time_slots = list(range(6, 24))
N_SLOTS = len(all_time_slots)  # 18
MAX_RATING = max(max(r) for r in ratings.values())


# ---------- genetic algorithm ----------
def fitness_function(schedule):
    return sum(ratings[program][slot] for slot, program in enumerate(schedule))


def random_schedule():
    return [random.choice(all_programs) for _ in range(N_SLOTS)]


def crossover(parent1, parent2):
    point = random.randint(1, N_SLOTS - 1)
    return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]


def mutate(schedule):
    schedule = schedule.copy()
    schedule[random.randint(0, N_SLOTS - 1)] = random.choice(all_programs)
    return schedule


def genetic_algorithm(co_r, mut_r, generations, pop_size, elitism):
    population = [random_schedule() for _ in range(pop_size)]
    history = []  # best rating in each generation

    for _ in range(generations):
        population.sort(key=fitness_function, reverse=True)
        history.append(fitness_function(population[0]))
        new_population = population[:elitism]  # keep the best schedules

        while len(new_population) < pop_size:
            p1, p2 = random.choices(population, k=2)  # pick two parents
            if random.random() < co_r:
                c1, c2 = crossover(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()
            if random.random() < mut_r:
                c1 = mutate(c1)
            if random.random() < mut_r:
                c2 = mutate(c2)
            new_population.extend([c1, c2])

        population = new_population[:pop_size]

    best = max(population, key=fitness_function)
    history.append(fitness_function(best))
    return best, history


def run_ga(co_r, mut_r, settings):
    best, history = genetic_algorithm(co_r, mut_r, **settings)
    return {"schedule": best, "history": history, "co_r": co_r, "mut_r": mut_r}


# ---------- display helpers ----------
def build_schedule_df(schedule):
    return pd.DataFrame({
        "Time Slot": [f"{h:02d}:00" for h in all_time_slots],
        "Program": schedule,
        "Rating": [ratings[p][i] for i, p in enumerate(schedule)],
    })


def show_table(df):
    st.dataframe(
        df,
        hide_index=True,
        column_config={
            "Rating": st.column_config.ProgressColumn(
                "Rating", min_value=0.0, max_value=MAX_RATING, format="%.1f"
            )
        },
    )


def show_chart(df):
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Time Slot:N", sort=None, title="Time slot"),
            y=alt.Y("Rating:Q", title="Rating"),
            color=alt.Color("Program:N", title="Program"),
            tooltip=["Time Slot", "Program", "Rating"],
        )
        .properties(height=320)
    )
    st.altair_chart(chart)


def show_result(result):
    schedule = result["schedule"]
    history = result["history"]
    df = build_schedule_df(schedule)
    total = fitness_function(schedule)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total rating", f"{total:.2f}", delta=f"{history[-1] - history[0]:+.2f} vs generation 1")
    c2.metric("Average per hour", f"{total / N_SLOTS:.2f}")
    top = df.loc[df["Rating"].idxmax()]
    c3.metric("Best slot", f"{top['Time Slot']}", help=f"{top['Program']} ({top['Rating']})")
    c4.metric("Parameters", f"CO_R {result['co_r']} | MUT_R {result['mut_r']}")

    left, right = st.columns([1, 1.4])
    with left:
        st.markdown("##### Resulting schedule")
        show_table(df)
    with right:
        st.markdown("##### Rating per time slot")
        show_chart(df)
        st.markdown("##### How the algorithm improved")
        st.line_chart(pd.DataFrame({"Best rating": history}))
        st.caption("X-axis: generation. The curve flattening means the algorithm has converged.")


# ---------- sidebar ----------
st.sidebar.title("⚙️ Parameters")
co_r = st.sidebar.slider(
    "Crossover Rate (CO_R)", 0.0, 0.95, 0.8, 0.05,
    help="Chance that two parents swap parts of their schedules.",
)
mut_r = st.sidebar.slider(
    "Mutation Rate (MUT_R)", 0.01, 0.05, 0.02, 0.01,
    help="Chance that a child schedule gets one random program change.",
)
run_single = st.sidebar.button("🚀 Run Genetic Algorithm", type="primary")

with st.sidebar.expander("Advanced settings"):
    generations = st.number_input("Generations", 10, 1000, 100, 10)
    pop_size = st.number_input("Population size", 10, 500, 50, 10)
    elitism = st.number_input("Elitism size", 1, 10, 2, 1)

settings = {"generations": int(generations), "pop_size": int(pop_size), "elitism": int(elitism)}

# ---------- main page ----------
st.title("📺 TV Scheduling with a Genetic Algorithm")
st.write(
    "Find the program order that gives the highest total audience rating "
    "across 18 hourly slots (06:00 to 23:00)."
)

tab_single, tab_trials, tab_data = st.tabs(["Single run", "3 trials", "Ratings data"])

# --- tab 1: single run ---
with tab_single:
    if run_single:
        with st.spinner("Evolving schedules..."):
            st.session_state["single"] = run_ga(co_r, mut_r, settings)
    if "single" in st.session_state:
        show_result(st.session_state["single"])
    else:
        st.info("Choose your parameters in the sidebar and click **Run Genetic Algorithm**.")

# --- tab 2: three trials ---
with tab_trials:
    st.write("Set a different CO_R and MUT_R for each trial, then run them together.")
    defaults = [(0.5, 0.01), (0.8, 0.02), (0.95, 0.05)]
    cols = st.columns(3)
    params = []
    for i, (col, (d_co, d_mut)) in enumerate(zip(cols, defaults), start=1):
        with col:
            st.markdown(f"**Trial {i}**")
            c = st.slider("Crossover Rate (CO_R)", 0.0, 0.95, d_co, 0.05, key=f"co_{i}")
            m = st.slider("Mutation Rate (MUT_R)", 0.01, 0.05, d_mut, 0.01, key=f"mut_{i}")
            params.append((c, m))

    if st.button("Run all 3 trials", type="primary"):
        with st.spinner("Running trials..."):
            st.session_state["trials"] = [run_ga(c, m, settings) for c, m in params]

    if "trials" in st.session_state:
        trials = st.session_state["trials"]

        summary = pd.DataFrame({
            "Trial": [f"Trial {i}" for i in range(1, 4)],
            "CO_R": [t["co_r"] for t in trials],
            "MUT_R": [t["mut_r"] for t in trials],
            "Total rating": [round(fitness_function(t["schedule"]), 2) for t in trials],
        })
        best_row = summary.loc[summary["Total rating"].idxmax()]
        st.success(f"Highest rating: {best_row['Trial']} with {best_row['Total rating']}")

        s1, s2 = st.columns(2)
        with s1:
            st.markdown("##### Comparison")
            st.dataframe(summary, hide_index=True)
        with s2:
            st.markdown("##### Convergence of each trial")
            st.line_chart(pd.DataFrame({f"Trial {i}": t["history"] for i, t in enumerate(trials, start=1)}))

        st.markdown("##### Schedules")
        tcols = st.columns(3)
        for i, (col, t) in enumerate(zip(tcols, trials), start=1):
            with col:
                st.markdown(f"**Trial {i}**: CO_R {t['co_r']}, MUT_R {t['mut_r']}")
                show_table(build_schedule_df(t["schedule"]))

# --- tab 3: data ---
with tab_data:
    st.write("The (modified) program ratings the algorithm is using. Rows are programs, columns are hours.")
    data_df = pd.DataFrame(ratings, index=[f"{h:02d}:00" for h in all_time_slots]).T
    st.dataframe(data_df)
