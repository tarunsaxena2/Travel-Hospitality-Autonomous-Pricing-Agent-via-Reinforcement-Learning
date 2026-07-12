# Travel & Hospitality — Autonomous Pricing Agent via Reinforcement Learning

## Project Overview
This project builds an autonomous dynamic pricing agent using Reinforcement 
Learning (DQN) for the travel and hospitality industry. The goal is to train 
an agent that outperforms heuristic pricing baselines (fixed, time-based 
discount, demand-based) in mean episodic revenue across simulated booking seasons.

## Team
- Member 1 — Environment & Simulation Engineer
- Member 2 — RL Algorithm Engineer
- Member 3 — Analysis & Policy Evaluation
- Member 4 — Eval & Deploy Lead 

## Tech Stack
Python, Gymnasium, PyTorch, NumPy, Pandas, Matplotlib, Seaborn

## Status
🚧 Project in progress — Week 1: MDP & Environment Design
# 🚀 Week 1 — MDP Design & Gymnasium Environment

> **Sprint Goal**
>
> Formulate the Dynamic Pricing problem as a **Markov Decision Process (MDP)**, develop a custom **Gymnasium environment**, and establish a **Random Agent baseline** for future Reinforcement Learning experiments.

---

# 📅 Week 1 Progress Timeline

| Day       | Theme                                     | Status |
| --------- | ----------------------------------------- | :----: |
| Monday    | Project Initialization & Repository Setup |    ✅   |
| Tuesday   | MDP Design & Environment Skeleton         |    ✅   |
| Wednesday | Environment Implementation & Baseline     |    ✅   |
| Thursday  | Demand Modeling & Evaluation              |    ✅   |
| Friday    | Finalization, Documentation & Integration |    ✅   |

---

# 👥 Team Contributions

---

## 🟦 Member 1 — Environment & Simulation Engineer

**Primary Responsibility**

* Design and implement the custom Gymnasium pricing environment.

### 📌 Monday

* Created project folder structure
* Configured Python virtual environment
* Installed required dependencies
* Organized source code layout

### 📌 Tuesday

* Developed `PricingEnv(gym.Env)` skeleton
* Defined Observation Space
* Defined Action Space
* Implemented environment constructor (`__init__`)

### 📌 Wednesday

* Implemented `reset()`
* Implemented `step(action)`
* Updated inventory
* Managed episode transitions
* Generated environment rewards

### 📌 Thursday

* Designed stochastic customer demand model
* Added price sensitivity
* Added time-to-departure demand behavior
* Implemented logistic demand curve

### 📌 Friday

* Added `render()` method
* Performed stability testing
* Executed 100 random episodes
* Verified environment consistency
* Eliminated invalid state transitions

**✅ Total Commits:** **5**

---

## 🟩 Member 2 — RL Algorithm Engineer

**Primary Responsibility**

* Define Reinforcement Learning formulation and environment logic.

### 📌 Monday

* Configured Jupyter Notebook environment
* Installed `nbstripout`
* Created environment design notebook

### 📌 Tuesday

* Formalized complete MDP
* Defined:

  * State Space
  * Action Space
  * Transition Function
  * Reward Function
  * Episode Horizon

### 📌 Wednesday

* Added action masking
* Restricted invalid pricing actions
* Tested edge cases

  * Zero inventory
  * Zero remaining days

### 📌 Thursday

* Developed automated PyTest unit tests
* Tested:

  * reset()
  * step()
  * reward calculation
  * terminal conditions

### 📌 Friday

* Completed environment documentation
* Added mathematical explanation
* Documented reward function
* Documented demand model
* Updated notebook

**✅ Total Commits:** **5**

---

## 🟨 Member 3 — Analysis & Policy Evaluation

**Primary Responsibility**

* Evaluate baseline performance and generate analytics.

### 📌 Monday

* Created Random Agent notebook
* Reviewed Gymnasium API
* Studied MDP references

### 📌 Tuesday

* Implemented Random Agent
* Connected agent with Pricing Environment

### 📌 Wednesday

* Ran Random Agent
* Executed **500 Episodes**
* Recorded episodic revenue

### 📌 Thursday

* Generated revenue histogram
* Calculated:

  * Mean Revenue
  * Standard Deviation
* Evaluated baseline statistics

### 📌 Friday

* Visualized episode trajectory
* Plotted:

  * Inventory over time
  * Selected price levels
* Cleared notebook outputs

**✅ Total Commits:** **5**

---

## 🟥 Member 4 — Evaluation & Deployment Lead

**Primary Responsibility**

* Repository management, documentation and project coordination.

### 📌 Monday

* Created GitHub Repository
* Added `.gitignore`
* Wrote project README
* Created Kanban Board
* Created all 20 GitHub Issues
* Assigned issues to team members

### 📌 Tuesday

* Created development branch
* Added `AGENTS.md`
* Defined all team roles
* Pushed branch to GitHub

### 📌 Wednesday

* Added project rules
* Created Brain Memory directory
* Documented project context
* Added MDP definitions

### 📌 Thursday

* Created team progress tracker
* Reviewed Pull Requests
* Merged completed work into main branch

### 📌 Friday

* Closed Week 1 GitHub Issues
* Updated Kanban Board
* Wrote Week 1 Summary
* Cleaned notebook outputs
* Final repository synchronization

**✅ Total Commits:** **5**

---

# 📊 Week 1 Sprint Statistics

| Metric                 |       Value |
| ---------------------- | ----------: |
| Sprint Duration        |      5 Days |
| Team Members           |           4 |
| Git Commits            |      **20** |
| GitHub Issues          |      **20** |
| Pull Requests Reviewed |    Multiple |
| Gym Environment        | ✅ Completed |
| MDP Formulation        | ✅ Completed |
| Random Agent Baseline  | ✅ Completed |
| Environment Testing    | ✅ Completed |
| Documentation          | ✅ Completed |

---

# 🏆 Week 1 Deliverables

✅ Project Repository Initialized

✅ Professional Folder Structure

✅ GitHub Kanban Workflow

✅ Issue Tracking

✅ Team Role Definition

✅ Complete Markov Decision Process (MDP)

✅ Custom Gymnasium Environment

✅ Reward Function

✅ Stochastic Demand Simulation

✅ Random Agent Baseline

✅ 500 Episode Evaluation

✅ Revenue Distribution Analysis

✅ Episode Visualization

✅ Automated Unit Tests

✅ Complete Documentation

---

# 🎯 Week 1 Outcome

At the end of Week 1, the project successfully established a complete Reinforcement Learning foundation by designing the Dynamic Pricing problem as a Markov Decision Process (MDP), implementing a custom Gymnasium environment, and validating its behavior using a Random Agent baseline. The environment now supports realistic inventory dynamics, stochastic customer demand, reward computation, and simulation-based experimentation, providing a robust platform for implementing advanced RL algorithms in subsequent development phases.
