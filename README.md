# 📶 Telecom Handover Simulation System

A **Dockerized**, **scalable simulation** of mobile handovers among cell-tower nodes, storing recent data in **ArangoDB** and visualizing the **handover graph**. Includes automated data generation, retention cleanup, and graph visualization.

---

## 📚 Table of Contents

- [Project Overview](#project-overview)  
- [Architecture](#architecture)  
- [Prerequisites](#prerequisites)  
- [Getting Started](#getting-started)  
  - [1. Clone Repository](#1-clone-repository)  
  - [2. Configure Environment](#2-configure-environment)  
  - [3. Build & Run with Docker Compose](#3-build--run-with-docker-compose)  
- [Components](#components)  
  - [Data Generator](#data-generator)  
  - [ArangoDB Setup](#arangodb-setup)  
  - [Retention Cleanup Script](#retention-cleanup-script)  
  - [Graph Visualization](#graph-visualization)  
- [Configuration](#configuration)  
- [Usage](#usage)  
- [Extending the Simulation](#extending-the-simulation)  
- [License](#license)  
- [Contact](#contact)

---

## 📌 Project Overview

This project simulates mobile phone **handovers** as clients move between cell-tower nodes.  
It generates synthetic data every **30 seconds**, stores the **last 1 hour of events** in **ArangoDB**, and includes a cleanup script for old data. A visualization component builds a **handover graph** for analysis.

---

## 🏗️ Architecture

![Architecture](https://github.com/user-attachments/assets/f62d3faa-5a7a-49ad-9d86-3e616c815e09)

---

## Process Block Diagram 

![Process Block](https://github.com/user-attachments/assets/e60c0502-a256-41b4-8f76-ab6cfdaa6896)

---

## 🧰 Prerequisites

- Docker & Docker Compose  
- Python 3.8+  
- Python packages:  
  \`\`\`
  pip install pandas openpyxl python-arango networkx matplotlib
  \`\`\`

---

## 🚀 Getting Started

### 1. Clone Repository

\`\`\`bash
git clone https://github.com/yourusername/telecom-handover-sim.git
cd telecom-handover-sim
\`\`\`

### 2. Configure Environment

Copy the example file and edit values:

\`\`\`bash
cp .env.example .env
\`\`\`

Modify as needed:

\`\`\`dotenv
ARANGO_ROOT_PASSWORD=ChangeMe!
DB_NAME=handover_db
EVENT_COLLECTION=events
GRAPH_NAME=handoverGraph
DATA_INTERVAL=30
RETENTION_WINDOW=3600
\`\`\`

### 3. Build & Run with Docker Compose

\`\`\`bash
docker-compose up --build -d
\`\`\`

Monitor logs:

\`\`\`bash
docker-compose logs -f generator
docker-compose logs -f cleanup
\`\`\`

---

## 🧩 Components

### 📈 Data Generator

- **Location:** \`generator/generate.py\`  
- Simulates 20 towers and random handovers every 30s.  
- Stores data in \`data/handover.xlsx\` and inserts into ArangoDB.

### 🗄️ ArangoDB Setup

- Initialized using \`init-arango.js\`  
- Creates the database, graph, and collections  
- Data is persisted in \`./arangodata\`

### 🧹 Retention Cleanup Script

- **Location:** \`cleanup/cleanup.py\`  
- Deletes events older than the defined retention window (e.g., 1 hour)  
- Can be run via cron or as a background service

### 📊 Graph Visualization

- **Location:** \`visualizer/visualize.py\`  
- Uses \`networkx\` and \`matplotlib\` to render the graph to PNG or HTML

---
## DashBoard Python 

![Dashboard](https://github.com/user-attachments/assets/bd28b06f-da83-40ad-8d09-398afa6e2ada)

---

## ⚙️ Configuration

Define via \`.env\` file or override with \`docker-compose.override.yml\`.

| Variable             | Description                          | Default        |
|----------------------|--------------------------------------|----------------|
| \`ARANGO_ROOT_PASSWORD\` | Admin password for ArangoDB        | \`ChangeMe!\`    |
| \`DB_NAME\`              | ArangoDB database name             | \`handover_db\`  |
| \`EVENT_COLLECTION\`     | Collection for handover events     | \`events\`       |
| \`GRAPH_NAME\`           | Name of the graph in ArangoDB      | \`handoverGraph\`|
| \`DATA_INTERVAL\`        | Interval (sec) between handovers   | \`30\`           |
| \`RETENTION_WINDOW\`     | Time window (sec) for retention    | \`3600\`         |

---

## ▶️ Usage

1. Start all services:

\`\`\`bash
docker-compose up -d
\`\`\`

2. Watch generated data:

\`\`\`bash
tail -f data/handover.xlsx
\`\`\`

3. Access ArangoDB UI:

\`\`\`
http://localhost:8529
\`\`\`

4. Run visualization:

\`\`\`bash
python visualizer/visualize.py --output latest_graph.png
\`\`\`

---

## 🔧 Extending the Simulation

- **Add Nodes:**  
  Modify \`generator/config.yml\` to include more towers.

- **Change Interval:**  
  Edit \`DATA_INTERVAL\` in \`.env\`.

- **Modify Retention Logic:**  
  Update \`RETENTION_WINDOW\` in \`.env\`.

- **Visual Enhancements:**  
  Integrate \`Plotly\`, \`D3.js\`, or web-based dashboards.

---

## 📄 License

This project is licensed under the MIT License.

---

## 📬 Contact

For questions or collaboration, contact me at:  
📧 **mokshhyadav.024@gmail.com**
