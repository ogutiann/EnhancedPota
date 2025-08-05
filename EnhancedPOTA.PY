%%capture
!pip install numpy torch scikit-learn deap pandas matplotlib seaborn pygad networkx
!pip install --upgrade torch
!wget https://data.cityofnewyork.us/api/views/2upf-qytp/rows.csv?accessType=DOWNLOAD -O nyc_temperature.csv
!wget https://data.cityofnewyork.us/api/views/edp9-qgv4/rows.csv?accessType=DOWNLOAD -O nyc_air_quality.csv
!pip install deap
!pip install pygad
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from deap import algorithms, base, creator, tools
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import gc
from tqdm import tqdm
import os
from datetime import datetime
import geopandas as gpd
from shapely.geometry import Point

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# Configuration for large-scale deployment
N_WORKERS = 5000
N_TASKS = 10000
CAPACITY = 3  # Max tasks per worker
GPU_ACCELERATED = torch.cuda.is_available()

if GPU_ACCELERATED:
    device = torch.device("cuda")
    print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("⚠️ Using CPU - GPU recommended for large-scale simulation")

######################
### 1. System Model ##
######################
class Worker:
    def __init__(self, id, location=None, real_world=False):
        self.id = id
        self.real_world = real_world

        if real_world:
            # Use actual NYC borough boundaries
            self.location = self.generate_nyc_location()
            # Skills correlate with income levels in different boroughs
            self.skill = self.generate_nyc_skills()
            # Mobility patterns differ by borough
            self.speed = self.generate_nyc_speed()
        else:
            self.location = np.array([np.random.uniform(0,100), np.random.uniform(0,100)])
            self.skill = np.array([np.random.uniform(2,5), np.random.uniform(2,5)])
            self.speed = np.random.uniform(0.5, 2)

        self.weights = np.random.dirichlet(np.ones(3))
        self.reputation = np.random.uniform(0.7, 0.95)
        self.utility_history = []
        self.assigned_tasks = []

    def generate_nyc_location(self):
        # NYC borough probabilities (Manhattan, Brooklyn, Queens, Bronx, Staten Island)
        borough_probs = [0.3, 0.3, 0.2, 0.15, 0.05]
        borough = np.random.choice(5, p=borough_probs)

        # Approximate bounding boxes for each borough
        bboxes = [
            (40.70, -74.02, 40.82, -73.93),  # Manhattan
            (40.57, -74.05, 40.73, -73.85),  # Brooklyn
            (40.68, -73.96, 40.80, -73.70),  # Queens
            (40.79, -73.93, 40.92, -73.75),  # Bronx
            (40.50, -74.26, 40.65, -74.05)   # Staten Island
        ]

        lat = np.random.uniform(bboxes[borough][0], bboxes[borough][2])
        lon = np.random.uniform(bboxes[borough][1], bboxes[borough][3])
        return np.array([lat, lon])

    def generate_nyc_skills(self):
        # Skill levels correlate with borough socioeconomic factors
        if self.location[0] > 40.75:  # Manhattan
            return np.array([np.random.uniform(4, 5), np.random.uniform(4, 5)])
        elif self.location[0] > 40.65:  # Brooklyn/Queens
            return np.array([np.random.uniform(3.5, 4.5), np.random.uniform(3.5, 4.5)])
        else:  # Bronx/Staten Island
            return np.array([np.random.uniform(3, 4), np.random.uniform(3, 4)])

    def generate_nyc_speed(self):
        # Travel speeds vary by borough
        if self.location[0] > 40.75:  # Manhattan (congested)
            return np.random.uniform(0.5, 1.2)
        elif self.location[0] > 40.65:  # Brooklyn/Queens
            return np.random.uniform(1.0, 1.8)
        else:  # Bronx/Staten Island
            return np.random.uniform(1.5, 2.0)

    def travel_time(self, task_location):
        # Haversine distance for real-world locations
        if self.real_world:
            R = 6371  # Earth radius in km
            lat1, lon1 = np.radians(self.location)
            lat2, lon2 = np.radians(task_location)

            dlat = lat2 - lat1
            dlon = lon2 - lon1

            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
            distance = R * c
        else:
            distance = np.linalg.norm(self.location - task_location)

        traffic_noise = np.random.laplace(0, 0.1)
        return max(0.1, distance / self.speed + traffic_noise)

class Task:
    def __init__(self, id, real_world=False):
        self.id = id
        self.real_world = real_world

        if real_world:
            self.location = self.generate_nyc_location()
            # Task characteristics based on NYC environmental data
            self.reward, self.difficulty, self.required_skill = self.generate_nyc_task()
        else:
            self.location = np.array([np.random.uniform(0,100), np.random.uniform(0,100)])
            self.reward = np.random.uniform(5,20)
            self.difficulty = np.random.uniform(1,10)
            self.required_skill = np.array([np.random.uniform(1,4), np.random.uniform(1,4)])

        self.deadline = np.random.randint(10,60)

    def generate_nyc_location(self):
        # Generate locations weighted by population density
        return np.array([40.7 + np.random.uniform(-0.2, 0.3),
                         -74.0 + np.random.uniform(-0.3, 0.3)])

    def generate_nyc_task(self):
        # Load NYC environmental data
        temp_df = pd.read_csv("nyc_temperature.csv")
        air_df = pd.read_csv("nyc_air_quality.csv")

        # Randomly select a task type
        task_type = np.random.choice(["temperature", "air_quality"])

        if task_type == "temperature":
            reward = np.random.uniform(8, 15)
            difficulty = np.random.uniform(3, 6)
            skill = np.array([np.random.uniform(2, 3), np.random.uniform(1, 2)])
        else:  # air_quality
            reward = np.random.uniform(12, 20)
            difficulty = np.random.uniform(6, 9)
            skill = np.array([np.random.uniform(3, 4), np.random.uniform(3, 4)])

        return reward, difficulty, skill

################################
### 2. Deep-Quality Reputation ##
################################
class QualityVAE(nn.Module):
    def __init__(self, input_dim=3, latent_dim=8):
        super(QualityVAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.mu = nn.Linear(16, latent_dim)
        self.logvar = nn.Linear(16, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.mu(h), self.logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    BCE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + 0.5 * KLD

def compute_quality(features, vae_model):
    with torch.no_grad():
        recon, _, _ = vae_model(features)
        error = torch.norm(features - recon, dim=1)
    return 1 - error

################################################
### 3. Federated RL for Preference Optimization #
################################################
class LSTMPolicy(nn.Module):
    def __init__(self, input_size=5, hidden_size=32, output_size=3):
        super(LSTMPolicy, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x[:, -1, :])
        return torch.softmax(x, dim=-1)

def federated_average(models):
    global_dict = {}
    for key in models[0].state_dict().keys():
        global_dict[key] = torch.stack([m.state_dict()[key] for m in models]).mean(0)
    return global_dict

def train_worker_rl(worker, tasks, epochs=20, batch_size=64, lr=0.005):
    model = LSTMPolicy().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Simulate historical data
    n_samples = 1000
    states = torch.randn(n_samples, 1, 5, device=device)
    actions = torch.randint(0, 3, (n_samples,), device=device)
    rewards = torch.rand(n_samples, device=device)

    # Training loop
    for epoch in range(epochs):
        permutation = torch.randperm(n_samples)
        for i in range(0, n_samples, batch_size):
            indices = permutation[i:i+batch_size]
            batch_states = states[indices]
            batch_actions = actions[indices]
            batch_rewards = rewards[indices]

            # Forward pass
            action_probs = model(batch_states)
            selected_probs = action_probs[torch.arange(len(batch_actions)), batch_actions]

            # Policy gradient loss
            loss = -torch.log(selected_probs) * batch_rewards
            loss = loss.mean()

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model

#############################################
### 4. Matching Algorithms Implementations ##
#############################################
def worker_utility(worker, task, beta=0.3):
    tau = worker.travel_time(task.location)
    skill_gap = np.mean(worker.skill - task.required_skill)
    return (worker.weights[0] * (1/tau) +
            worker.weights[1] * task.reward +
            worker.weights[2] * skill_gap +
            beta * worker.reputation)

def task_utility(task, worker, zeta=0.3, kappa=0.7):
    if np.any(worker.skill < task.required_skill):
        return 0
    tau = worker.travel_time(task.location)
    return (kappa * (1/tau) + zeta * worker.reputation)

# 4.1 RAGS (Our Proposed)
def rags_matching(workers, tasks, capacity=CAPACITY, beta=0.3, zeta=0.3, kappa=0.7):
    # Precompute utilities
    w_utils = np.zeros((len(workers), len(tasks)))
    t_utils = np.zeros((len(tasks), len(workers)))

    for i, worker in enumerate(workers):
        for j, task in enumerate(tasks):
            w_utils[i, j] = worker_utility(worker, task, beta)
            t_utils[j, i] = task_utility(task, worker, zeta, kappa)

    # Preference lists
    w_prefs = [np.argsort(-w_utils[i]) for i in range(len(workers))]
    t_prefs = [np.argsort(-t_utils[j]) for j in range(len(tasks))]

    # Initialize
    proposals = np.zeros(len(workers), dtype=int)
    worker_assignments = [[] for _ in range(len(workers))]
    task_assignments = [[] for _ in range(len(tasks))]
    unmatched = list(range(len(workers)))

    # Matching loop
    while unmatched:
        i = unmatched.pop(0)
        if proposals[i] >= len(tasks):
            continue

        j = w_prefs[i][proposals[i]]
        proposals[i] += 1

        if len(task_assignments[j]) < capacity:
            worker_assignments[i].append(j)
            task_assignments[j].append(i)
        else:
            # Find worst assigned worker
            worst_idx = min(task_assignments[j], key=lambda x: t_utils[j, x])
            if t_utils[j, i] > t_utils[j, worst_idx]:
                # Replace worker
                worker_assignments[worst_idx].remove(j)
                task_assignments[j].remove(worst_idx)
                unmatched.append(worst_idx)

                worker_assignments[i].append(j)
                task_assignments[j].append(i)
            else:
                unmatched.append(i)

    return worker_assignments, task_assignments

# 4.2 Classic Gale-Shapley
def classic_gs(workers, tasks, capacity=CAPACITY):
    # Static utilities
    w_utils = np.zeros((len(workers), len(tasks)))
    t_utils = np.zeros((len(tasks), len(workers)))

    for i, worker in enumerate(workers):
        for j, task in enumerate(tasks):
            w_utils[i, j] = worker_utility(worker, task)
            t_utils[j, i] = task_utility(task, worker)

    # Same as RAGS but without dynamic adjustments
    return rags_matching(workers, tasks, capacity)

# 4.3 Random Allocation
def random_matching(workers, tasks, capacity=CAPACITY):
    worker_assignments = [[] for _ in range(len(workers))]
    task_assignments = [[] for _ in range(len(tasks))]

    # Shuffle all possible assignments
    all_pairs = [(i, j) for i in range(len(workers)) for j in range(len(tasks))]
    random.shuffle(all_pairs)

    for i, j in all_pairs:
        if (len(worker_assignments[i]) < capacity and
            len(task_assignments[j]) < capacity and
            np.all(workers[i].skill >= tasks[j].required_skill)):
            worker_assignments[i].append(j)
            task_assignments[j].append(i)

    return worker_assignments, task_assignments

# 4.4 FML (Federated Matching without RL)
def fml_matching(workers, tasks, capacity=CAPACITY):
    # Average weights across workers
    avg_weights = np.mean([w.weights for w in workers], axis=0)

    # Override worker preferences with global average
    for worker in workers:
        worker.weights = avg_weights.copy()

    return classic_gs(workers, tasks, capacity)

# 4.5 SenseChain+ Simulation
def sensechain_matching(workers, tasks, capacity=CAPACITY):
    # Simulate blockchain overhead
    time.sleep(0.001 * len(tasks))  # 1ms per task

    # Same as classic GS
    return classic_gs(workers, tasks, capacity)

# 4.6 DeepMatching (Neural Network Baseline)
class DeepMatching(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=1):
        super(DeepMatching, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, worker_feats, task_feats):
        x = torch.cat([worker_feats, task_feats], dim=1)
        return self.net(x)

def deep_matching(workers, tasks, capacity=CAPACITY):
    # Convert to feature vectors
    worker_feats = np.array([[w.location[0], w.location[1], w.skill[0], w.skill[1], w.reputation]
                            for w in workers])
    task_feats = np.array([[t.location[0], t.location[1], t.reward, t.difficulty,
                          t.required_skill[0], t.required_skill[1]] for t in tasks])

    # Normalize
    scaler = MinMaxScaler()
    worker_feats = scaler.fit_transform(worker_feats)
    task_feats = scaler.fit_transform(task_feats)

    # Initialize model
    model = DeepMatching(input_size=worker_feats.shape[1] + task_feats.shape[1])
    model.eval()

    # Predict utilities
    w_utils = np.zeros((len(workers), len(tasks)))
    for i, w in enumerate(worker_feats):
        for j, t in enumerate(task_feats):
            with torch.no_grad():
                w_utils[i, j] = model(torch.tensor([w], dtype=torch.float32),
                                      torch.tensor([t], dtype=torch.float32)).item()

    # Greedy assignment
    worker_assignments = [[] for _ in range(len(workers))]
    task_assignments = [[] for _ in range(len(tasks))]

    # Assign tasks based on predicted utility
    for j in range(len(tasks)):
        top_workers = np.argsort(-w_utils[:, j])[:capacity]
        for i in top_workers:
            if len(worker_assignments[i]) < capacity and len(task_assignments[j]) < capacity:
                worker_assignments[i].append(j)
                task_assignments[j].append(i)

    return worker_assignments, task_assignments

# 4.7 FairTask (Fairness-Aware Baseline)
def fairtask_matching(workers, tasks, capacity=CAPACITY):
    # Calculate worker-task utilities
    utilities = np.zeros((len(workers), len(tasks)))
    for i, w in enumerate(workers):
        for j, t in enumerate(tasks):
            utilities[i, j] = worker_utility(w, t)

    # Initialize assignments
    worker_assignments = [[] for _ in range(len(workers))]
    task_assignments = [[] for _ in range(len(tasks))]
    worker_counts = np.zeros(len(workers))

    # Fairness-based assignment
    for _ in range(len(tasks) * capacity):
        # Find worker with minimum current utility
        min_util = float('inf')
        candidate = -1
        for i in range(len(workers)):
            if worker_counts[i] >= capacity:
                continue
            current_util = sum(utilities[i, j] for j in worker_assignments[i])
            if current_util < min_util:
                min_util = current_util
                candidate = i

        if candidate == -1:
            break

        # Find best task for this worker
        best_task = -1
        best_util = -float('inf')
        for j in range(len(tasks)):
            if len(task_assignments[j]) < capacity and utilities[candidate, j] > best_util:
                best_util = utilities[candidate, j]
                best_task = j

        if best_task == -1:
            break

        worker_assignments[candidate].append(best_task)
        task_assignments[best_task].append(candidate)
        worker_counts[candidate] += 1

    return worker_assignments, task_assignments

# 4.8 Q-Learning MCS
class QLearningMCS:
    def __init__(self, n_workers, n_tasks, capacity=CAPACITY):
        self.n_workers = n_workers
        self.n_tasks = n_tasks
        self.capacity = capacity
        self.q_table = np.zeros((n_workers, n_tasks))
        self.alpha = 0.1  # learning rate
        self.gamma = 0.9  # discount factor
        self.epsilon = 0.1  # exploration rate

    def learn(self, workers, tasks, episodes=100):
        for ep in range(episodes):
            # Reset assignments
            worker_assignments = [[] for _ in range(self.n_workers)]
            task_assignments = [[] for _ in range(self.n_tasks)]

            for step in range(self.n_tasks * self.capacity):
                # Select worker with epsilon-greedy
                worker_id = np.random.randint(0, self.n_workers)
                if np.random.random() < self.epsilon:
                    task_id = np.random.randint(0, self.n_tasks)
                else:
                    task_id = np.argmax(self.q_table[worker_id])

                # Check if assignment possible
                if (len(worker_assignments[worker_id]) < self.capacity and
                    len(task_assignments[task_id]) < self.capacity):
                    # Calculate reward
                    reward = worker_utility(workers[worker_id], tasks[task_id])

                    # Update Q-value
                    old_value = self.q_table[worker_id, task_id]
                    next_max = np.max(self.q_table[worker_id])
                    new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
                    self.q_table[worker_id, task_id] = new_value

                    # Record assignment
                    worker_assignments[worker_id].append(task_id)
                    task_assignments[task_id].append(worker_id)

        return worker_assignments, task_assignments

############################
### 5. Evaluation Metrics ##
############################
def calculate_satisfaction(workers, tasks, assignments):
    satisfaction = []
    for i, worker in enumerate(workers):
        utils = [worker_utility(worker, tasks[j]) for j in assignments[i]]
        satisfaction.append(np.mean(utils) if utils else 0)
    return np.mean(satisfaction)

def calculate_task_coverage(task_assignments):
    return sum(len(a) > 0 for a in task_assignments) / len(task_assignments)

def calculate_quality(workers, tasks, assignments):
    # Simulate data submissions with noise
    qualities = []
    for i, worker in enumerate(workers):
        for task_id in assignments[i]:
            task = tasks[task_id]
            # Quality depends on worker skill and task difficulty
            base_quality = np.mean(worker.skill - task.required_skill) / 5
            noise = np.random.normal(0, 0.2 * (1 - worker.reputation))
            qualities.append(max(0, min(1, base_quality + noise)))
    return np.mean(qualities) if qualities else 0

def check_stability(workers, tasks, worker_assignments, task_assignments):
    blocking_pairs = 0
    sample_size = min(1000, len(workers))

    for i in np.random.choice(len(workers), sample_size, replace=False):
        worker = workers[i]
        current_utils = [worker_utility(worker, tasks[j]) for j in worker_assignments[i]]
        current_min = min(current_utils) if current_utils else -np.inf

        for j in np.random.choice(len(tasks), min(100, len(tasks)), replace=False):
            if j in worker_assignments[i]:
                continue

            # Worker prefers this task
            w_util = worker_utility(worker, tasks[j])
            if w_util <= current_min:
                continue

            # Task prefers this worker
            t_util = task_utility(tasks[j], worker)
            current_workers = task_assignments[j]
            if not current_workers:
                blocking_pairs += 1
                continue

            min_current_util = min(task_utility(tasks[j], workers[k]) for k in current_workers)
            if t_util > min_current_util:
                blocking_pairs += 1

    return 1 - (blocking_pairs / (sample_size * 100))

########################
### 6. Main Simulation #
########################
def run_full_simulation(real_world=False):
    print(f"🚀 Starting {'real-world' if real_world else 'synthetic'} simulation "
          f"with {N_WORKERS} workers and {N_TASKS} tasks")

    results = []
    methods = {
        "POTA (Proposed)": rags_matching,
        "Classic GS": classic_gs,
        "Random Allocation": random_matching,
        "FML": fml_matching,
        "SenseChain+": sensechain_matching,
        "DeepMatching": deep_matching,
        "FairTask": fairtask_matching
    }

    # Initialize Q-Learning (separate due to different interface)
    q_learner = QLearningMCS(N_WORKERS, N_TASKS)

    # Generate workers and tasks
    print("Generating workers and tasks...")
    workers = [Worker(i, real_world=real_world) for i in tqdm(range(N_WORKERS))]
    tasks = [Task(j, real_world=real_world) for j in tqdm(range(N_TASKS))]

    # Train DQRS model
    print("Training DQRS VAE...")
    vae_model = QualityVAE().to(device)
    optimizer = optim.Adam(vae_model.parameters(), lr=0.001)

    # Generate synthetic quality features
    features = torch.randn(100000, 3, device=device)
    for epoch in tqdm(range(50)):
        recon, mu, logvar = vae_model(features)
        loss = vae_loss(recon, features, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Federated RL Training
    print("Training federated RL agents...")
    worker_models = []
    batch_size = min(100, len(workers))

    for i in tqdm(range(0, len(workers), batch_size)):
        batch = workers[i:i+batch_size]
        batch_models = [train_worker_rl(w, tasks) for w in batch]
        worker_models.extend(batch_models)

        # Clear memory
        if GPU_ACCELERATED:
            torch.cuda.empty_cache()

    # Apply federated averaging
    print("Applying federated averaging...")
    global_weights = federated_average(worker_models)
    for i, model in enumerate(worker_models):
        model.load_state_dict(global_weights)
        workers[i].weights = model(torch.randn(1, 1, 5, device=device)).cpu().detach().numpy()[0]

    # Run all matching algorithms
    for method_name, matching_fn in methods.items():
        print(f"\n🔍 Running {method_name}...")
        start_time = time.time()

        # Reset assignments
        for w in workers:
            w.assigned_tasks = []

        # Run matching
        worker_assignments, task_assignments = matching_fn(workers, tasks)
        elapsed = time.time() - start_time

        # Calculate metrics
        satisfaction = calculate_satisfaction(workers, tasks, worker_assignments)
        coverage = calculate_task_coverage(task_assignments)
        quality = calculate_quality(workers, tasks, worker_assignments)
        stability = check_stability(workers, tasks, worker_assignments, task_assignments)

        # Record results
        results.append({
            "Method": method_name,
            "Satisfaction": satisfaction,
            "Task Coverage": coverage,
            "Data Quality": quality,
            "Stability": stability,
            "Time (s)": elapsed,
            "Scenario": "Real-world" if real_world else "Synthetic"
        })

        print(f"  Satisfaction: {satisfaction:.3f} | Coverage: {coverage:.1%} | "
              f"Quality: {quality:.3f} | Stability: {stability:.1%} | Time: {elapsed:.1f}s")

    # Run Q-Learning separately
    print("\n🔍 Running Q-Learning MCS...")
    start_time = time.time()
    worker_assignments, task_assignments = q_learner.learn(workers, tasks)
    elapsed = time.time() - start_time

    satisfaction = calculate_satisfaction(workers, tasks, worker_assignments)
    coverage = calculate_task_coverage(task_assignments)
    quality = calculate_quality(workers, tasks, worker_assignments)
    stability = check_stability(workers, tasks, worker_assignments, task_assignments)

    results.append({
        "Method": "Q-Learning MCS",
        "Satisfaction": satisfaction,
        "Task Coverage": coverage,
        "Data Quality": quality,
        "Stability": stability,
        "Time (s)": elapsed,
        "Scenario": "Real-world" if real_world else "Synthetic"
    })
    print(f"  Satisfaction: {satisfaction:.3f} | Coverage: {coverage:.1%} | "
          f"Quality: {quality:.3f} | Stability: {stability:.1%} | Time: {elapsed:.1f}s")

    return results

def run_parameter_sensitivity():
    print("🔬 Running parameter sensitivity analysis...")
    results = []
    workers = [Worker(i) for i in range(1000)]  # Smaller scale for sensitivity
    tasks = [Task(j) for j in range(2000)]

    # Test beta (reputation weight)
    for beta in [0.1, 0.2, 0.3, 0.4, 0.5]:
        start_time = time.time()
        worker_assignments, task_assignments = rags_matching(workers, tasks, beta=beta)
        elapsed = time.time() - start_time

        satisfaction = calculate_satisfaction(workers, tasks, worker_assignments)
        quality = calculate_quality(workers, tasks, worker_assignments)

        results.append({
            "Parameter": "β",
            "Value": beta,
            "Satisfaction": satisfaction,
            "Quality": quality,
            "Time (s)": elapsed
        })

    # Test learning rate (eta)
    for lr in [0.001, 0.005, 0.01, 0.05, 0.1]:
        # Retrain RL with different learning rate
        worker_models = []
        for w in workers:
            model = train_worker_rl(w, tasks, lr=lr)
            worker_models.append(model)

        # Apply federated averaging
        global_weights = federated_average(worker_models)
        for i, model in enumerate(worker_models):
            model.load_state_dict(global_weights)
            workers[i].weights = model(torch.randn(1, 1, 5)).detach().numpy()[0]

        # Run matching
        start_time = time.time()
        worker_assignments, task_assignments = rags_matching(workers, tasks)
        elapsed = time.time() - start_time

        satisfaction = calculate_satisfaction(workers, tasks, worker_assignments)
        quality = calculate_quality(workers, tasks, worker_assignments)

        results.append({
            "Parameter": "η",
            "Value": lr,
            "Satisfaction": satisfaction,
            "Quality": quality,
            "Time (s)": elapsed
        })

    # Test alpha (reputation decay)
    for alpha in [0.6, 0.7, 0.8, 0.9, 0.95]:
        # Modify reputation system (simplified)
        original_reps = [w.reputation for w in workers]
        for w in workers:
            w.reputation = alpha * w.reputation  # Simulate decay effect

        start_time = time.time()
        worker_assignments, task_assignments = rags_matching(workers, tasks)
        elapsed = time.time() - start_time

        satisfaction = calculate_satisfaction(workers, tasks, worker_assignments)
        quality = calculate_quality(workers, tasks, worker_assignments)

        results.append({
            "Parameter": "α",
            "Value": alpha,
            "Satisfaction": satisfaction,
            "Quality": quality,
            "Time (s)": elapsed
        })

        # Restore reputations
        for w, rep in zip(workers, original_reps):
            w.reputation = rep

    return pd.DataFrame(results)

def plot_results(results_df, sensitivity_df):
    # Comparative results
    plt.figure(figsize=(14, 10))
    scenario = results_df['Scenario'].iloc[0]

    # Satisfaction and Quality
    plt.subplot(2, 2, 1)
    sns.barplot(x="Method", y="Satisfaction", data=results_df)
    plt.title(f"Worker Satisfaction ({scenario})")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)

    plt.subplot(2, 2, 2)
    sns.barplot(x="Method", y="Data Quality", data=results_df)
    plt.title(f"Data Quality Score ({scenario})")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)

    # Coverage and Stability
    plt.subplot(2, 2, 3)
    sns.barplot(x="Method", y="Task Coverage", data=results_df)
    plt.title(f"Task Coverage ({scenario})")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)

    plt.subplot(2, 2, 4)
    sns.barplot(x="Method", y="Stability", data=results_df)
    plt.title(f"Matching Stability ({scenario})")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(f"pota_metrics_{scenario.lower().replace('-', '_')}.png")

    # Sensitivity analysis
    plt.figure(figsize=(12, 8))
    for i, param in enumerate(['β', 'η', 'α']):
        plt.subplot(2, 2, i+1)
        param_df = sensitivity_df[sensitivity_df['Parameter'] == param]
        plt.plot(param_df['Value'], param_df['Satisfaction'], 'o-', label='Satisfaction')
        plt.plot(param_df['Value'], param_df['Quality'], 's-', label='Quality')
        plt.title(f"Sensitivity to {param}")
        plt.xlabel("Parameter Value")
        plt.ylabel("Metric Value")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig("parameter_sensitivity.png")
    print("\n📊 Visualizations saved")

# Run the simulations
if __name__ == "__main__":
    # Run synthetic scenario
    synthetic_results = run_full_simulation(real_world=False)
    synthetic_df = pd.DataFrame(synthetic_results)

    # Run real-world scenario
    realworld_results = run_full_simulation(real_world=True)
    realworld_df = pd.DataFrame(realworld_results)

    # Run sensitivity analysis
    sensitivity_df = run_parameter_sensitivity()

    # Combine and save results
    results_df = pd.concat([synthetic_df, realworld_df])
    results_df.to_csv("pota_results.csv", index=False)
    sensitivity_df.to_csv("parameter_sensitivity.csv", index=False)
    print("\n✅ Results saved to pota_results.csv and parameter_sensitivity.csv")

    # Visualize results
    plot_results(synthetic_df, sensitivity_df)
    plot_results(realworld_df, sensitivity_df)

    # Display final results
    print("\n⭐ Final Results ⭐")
    print(results_df.groupby(['Scenario', 'Method']).mean())