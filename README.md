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

---

# 📅 Week 2 Progress Timeline

| Day       | Theme                                          | Status |
| --------- | ---------------------------------------------- | :----: |
| Monday    | Baseline Agent Development & Q-Learning Setup  |    ✅   |
| Tuesday   | Bellman Learning & Baseline Evaluation         |    ✅   |
| Wednesday | Q-Learning Training & Performance Benchmarking |    ✅   |
| Thursday  | Hyperparameter Optimization & Result Analysis  |    ✅   |
| Friday    | Finalization, Documentation & Integration      |    ✅   |

---

# 👥 Team Contributions

---

## 🟦 Member 1 — Environment & Simulation Engineer

**Primary Responsibility**

* Develop and validate heuristic pricing strategies for comparison with Reinforcement Learning models.

### 📌 Monday

* Implemented **FixedPriceAgent**
* Created `baseline_agents.py`
* Added constant pricing strategy
* Integrated agent with Pricing Environment

### 📌 Tuesday

* Developed **TimeBasedDiscountAgent**
* Implemented automatic 10% daily price reduction
* Tested pricing strategy in simulation environment
* Verified seasonal pricing behavior

### 📌 Wednesday

* Implemented **DemandBasedAgent**
* Designed inventory-to-time pricing strategy
* Tested stability over multiple simulation episodes
* Validated adaptive pricing logic

### 📌 Thursday

* Finalized all heuristic pricing agents
* Added comprehensive documentation
* Standardized code structure
* Verified functionality of all baseline strategies

### 📌 Friday

* Completed Week 2 Baseline Notebook
* Added introduction, methodology and conclusion
* Cleared notebook outputs
* Prepared final notebook for GitHub

**✅ Total Commits:** **5**

---

## 🟩 Member 2 — RL Algorithm Engineer

**Primary Responsibility**

* Build and optimize the Tabular Q-Learning algorithm.

### 📌 Monday

* Discretized state space
* Created Inventory Buckets
* Created Days Remaining Buckets
* Initialized Q-table

### 📌 Tuesday

* Implemented Bellman Update Equation
* Added learning rate
* Added discount factor
* Implemented epsilon-greedy exploration
* Configured epsilon decay

### 📌 Wednesday

* Trained Q-Learning agent
* Executed **5,000 training episodes**
* Logged reward curve
* Monitored training convergence

### 📌 Thursday

* Tuned hyperparameters
* Evaluated multiple learning rates
* Compared discount factors
* Selected optimal epsilon decay schedule

### 📌 Friday

* Finalized `q_learning_agent.py`
* Evaluated best model on **500 unseen episodes**
* Cleaned notebook outputs
* Documented final implementation

**✅ Total Commits:** **5**

---

## 🟨 Member 3 — Analysis & Policy Evaluation

**Primary Responsibility**

* Evaluate heuristic strategies and compare them with Q-Learning.

### 📌 Monday

* Built common evaluation framework
* Implemented reusable simulation helper
* Standardized evaluation metrics

### 📌 Tuesday

* Evaluated Random Agent
* Evaluated Fixed Price Agent
* Evaluated Time-Based Discount Agent
* Generated revenue comparison plots

### 📌 Wednesday

* Compared Q-Learning against all heuristic agents
* Calculated:

  * Mean Revenue
  * Standard Deviation
  * Sell-through Rate
* Recorded benchmarking results

### 📌 Thursday

* Computed revenue improvement percentage
* Compared Q-Learning with best-performing heuristic
* Documented experimental findings

### 📌 Friday

* Finalized comparison notebook
* Added complete performance summary table
* Verified evaluation metrics
* Cleared notebook outputs

**✅ Total Commits:** **5**

---

## 🟥 Member 4 — Evaluation & Deployment Lead

**Primary Responsibility**

* Coordinate evaluation, documentation and repository integration.

### 📌 Monday

* Created `results_comparison.md`
* Designed result documentation structure
* Added comparison section templates

### 📌 Tuesday

* Reviewed Q-Learning implementation
* Verified Bellman update correctness
* Reviewed Pull Requests
* Added technical review comments

### 📌 Wednesday

* Created baseline comparison tables
* Organized Week 2 evaluation notebook
* Documented performance metrics

### 📌 Thursday

* Wrote Week 2 findings
* Explained improvements achieved by Q-Learning
* Documented comparison with heuristic pricing strategies

### 📌 Friday

* Updated GitHub Kanban Board
* Closed Week 2 Issues
* Updated README
* Reviewed and merged all Pull Requests
* Completed Week 2 repository synchronization

**✅ Total Commits:** **5**

---

# 📊 Week 2 Sprint Statistics

| Metric                   |                 Value |
| ------------------------ | --------------------: |
| Sprint Duration          |                5 Days |
| Team Members             |                     4 |
| Git Commits              |                **20** |
| GitHub Issues Completed  | **5** (Issues #6–#10) |
| Pull Requests Reviewed   |              Multiple |
| Heuristic Pricing Agents |           ✅ Completed |
| Tabular Q-Learning       |           ✅ Completed |
| Hyperparameter Tuning    |           ✅ Completed |
| Agent Benchmarking       |           ✅ Completed |
| Documentation            |           ✅ Completed |

---

# 🏆 Week 2 Deliverables

✅ Fixed Price Agent

✅ Time-Based Discount Agent

✅ Demand-Based Pricing Agent

✅ Shared Evaluation Framework

✅ State Space Discretization

✅ Tabular Q-Learning Implementation

✅ Bellman Learning Algorithm

✅ Hyperparameter Optimization

✅ 5,000 Episode Training

✅ 500 Episode Testing

✅ Baseline vs Q-Learning Comparison

✅ Revenue Performance Analysis

✅ Sell-through Rate Evaluation

✅ Week 2 Documentation

✅ Updated GitHub Repository

---

# 🎯 Week 2 Outcome

By the end of Week 2, the project successfully established a strong Reinforcement Learning baseline by implementing multiple heuristic pricing strategies and developing a complete Tabular Q-Learning agent. The Q-Learning model was trained, optimized through hyperparameter tuning, and rigorously evaluated against heuristic approaches using extensive simulation experiments. Performance metrics such as mean revenue, sell-through rate, and revenue improvement demonstrated that the learned policy consistently outperformed static pricing strategies, providing a solid foundation for transitioning to a Deep Q-Network (DQN) architecture in the next phase of the project. 
