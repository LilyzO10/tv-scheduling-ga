import csv
import random
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------- data ----------
def read_csv_to_dict(file_path):
    program_ratings = {}
    with open(file_path, newline="") as file:
        reader = csv.reader(file)
        next(reader)  # skip header
        for row in reader:
            program_ratings[row[0]] = [float(x) for x in row[1:]]
    return program_ratings

CSV_PATH = Path(__file__).parent / "program_ratings.csv"
ratings = read_csv_to_dict(CSV_PATH)
all_programs = list(ratings.keys())
all_time_slots = list(range(6, 24))
N_SLOTS = len(all_time_slots)  # 18

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

def genetic_algorithm(co_r, mut_r, generations=100, pop_size=50, elitism=2):
    population = [random_schedule() for _ in range(pop_size)]

    for _ in range(generations):
        population.sort(key=fitness_function, reverse=True)
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

        population = new_population

    return max(population, key=fitness_function)

# ---------- interface ----------
st.title("TV Scheduling - Genetic Algorithm")

co_r = st.slider("Crossover Rate (CO_R)", min_value=0.0, max_value=0.95, value=0.8, step=0.05)
mut_r = st.slider("Mutation Rate (MUT_R)", min_value=0.01, max_value=0.05, value=0.02, step=0.01)

if st.button("Run Genetic Algorithm"):
    best = genetic_algorithm(co_r, mut_r)

    schedule_df = pd.DataFrame({
        "Time Slot": [f"{hour:02d}:00" for hour in all_time_slots],
        "Program": best,
    })

    st.subheader("Resulting Schedule")
    st.table(schedule_df)
    st.write("Parameters used: CO_R =", co_r, ", MUT_R =", mut_r)
    st.write("Total rating:", round(fitness_function(best), 2))
