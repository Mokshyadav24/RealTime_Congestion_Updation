import random
from datetime import datetime
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv
from arango import ArangoClient
from arango.exceptions import DocumentUpdateError

# --- ArangoDB config ---
ARANGO_URL = "http://localhost:8529"
DB_NAME = "_system"
USERNAME = "root"
PASSWORD = "yourpassword"

TRAFFIC_COLLECTION = "traffic_data"
EDGE_COLLECTION_DEVICE = "cell_edges"

# --- Connect to ArangoDB ---
client = ArangoClient(hosts=ARANGO_URL)
db = client.db(DB_NAME, username=USERNAME, password=PASSWORD)
traffic_col = db.collection(TRAFFIC_COLLECTION)

if not db.has_collection(EDGE_COLLECTION_DEVICE):
    raise Exception(f"Collection {EDGE_COLLECTION_DEVICE} not found! Ensure your edge generation has been run.")
edge_device_col = db.collection(EDGE_COLLECTION_DEVICE)

# --- Build graph data ---
device_features = []
devicekpi_features = []
edge_index_kpi = [[], []]
edge_index_device = [[], []]
labels = []
device_keys = []
cell_id_to_index = {}

kpi_keys = [
    'uplink_traffic_MB', 'downlink_traffic_MB', 'active_users', 'call_drop_rate',
    'latency_ms', 'throughput_Mbps', 'signal_strength_dBm', 'resource_utilization',
    'handover_success_rate', 'packet_loss_rate', 'jitter_ms'
]

for i, doc in enumerate(traffic_col.all()):
    cell_id = str(int(doc["cell_id"]))
    device_keys.append(cell_id)
    cell_id_to_index[f"traffic_data/{cell_id}"] = i

    x_device = torch.tensor([random.random() for _ in range(11)], dtype=torch.float)
    device_features.append(x_device)

    kpi_values = [float(doc.get(k, 0)) for k in kpi_keys]
    x_kpi = torch.tensor(kpi_values, dtype=torch.float)
    devicekpi_features.append(x_kpi)

    edge_index_kpi[0].append(i)
    edge_index_kpi[1].append(i)

    uplink = float(doc.get("uplink_traffic_MB", 0))
    latency = float(doc.get("latency_ms", 0))
    resource_util = float(doc.get("resource_utilization", 0))
    pkt_loss = float(doc.get("packet_loss_rate", 0))
    jitter = float(doc.get("jitter_ms", 0))
    call_drop = float(doc.get("call_drop_rate", 0))

    score = 0
    score += uplink > 40
    score += latency > 80
    score += resource_util > 75
    score += pkt_loss > 1.5
    score += jitter > 20
    score += call_drop > 2.0

    is_congested = score >= 2  # You can adjust this threshold to tune sensitivity
    labels.append(1 if is_congested else 0)


# --- Load device-to-device edges ---
for edge in edge_device_col.all():
    src = edge["_from"]
    dst = edge["_to"]
    if src in cell_id_to_index and dst in cell_id_to_index:
        edge_index_device[0].append(cell_id_to_index[src])
        edge_index_device[1].append(cell_id_to_index[dst])

# --- PyG HeteroData setup ---
data = HeteroData()
data["device"].x = torch.stack(device_features)
data["devicekpi"].x = torch.stack(devicekpi_features)
data["device", "has_kpi", "devicekpi"].edge_index = torch.tensor(edge_index_kpi, dtype=torch.long)
data["device", "connected_to", "device"].edge_index = torch.tensor(edge_index_device, dtype=torch.long)

# --- Message passing ---
conv = HeteroConv({
    ("device", "has_kpi", "devicekpi"): SAGEConv(11, 11),
    ("devicekpi", "rev_has_kpi", "device"): SAGEConv(11, 11),
    ("device", "connected_to", "device"): SAGEConv(11, 11),
    ("device", "rev_connected_to", "device"): SAGEConv(11, 11),
}, aggr="sum")

out = conv(data.x_dict, {
    ("device", "has_kpi", "devicekpi"): data["device", "has_kpi", "devicekpi"].edge_index,
    ("devicekpi", "rev_has_kpi", "device"): data["device", "has_kpi", "devicekpi"].edge_index.flip(0),
    ("device", "connected_to", "device"): data["device", "connected_to", "device"].edge_index,
    ("device", "rev_connected_to", "device"): data["device", "connected_to", "device"].edge_index.flip(0),
})

device_embeddings = out["device"].detach()
labels = torch.tensor(labels, dtype=torch.long)  # ✅ Convert before indexing

# --- Split data into train and test sets ---
train_idx, test_idx = train_test_split(
    list(range(len(labels))), test_size=0.4, random_state=42, stratify=labels
)

train_embeddings = device_embeddings[train_idx]
train_labels = labels[train_idx]
test_embeddings = device_embeddings[test_idx]
test_labels = labels[test_idx]

print(labels)
print("Label distribution:", torch.bincount(labels))


# --- Classification ---
classifier = torch.nn.Linear(device_embeddings.shape[1], 2)
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)
class_counts = torch.bincount(train_labels)
weights = 1.0 / class_counts.float()
loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
print("Class weights:", weights)


for epoch in range(100):
    logits = classifier(train_embeddings)
    loss = loss_fn(logits, train_labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# --- Prediction on all data ---
with torch.no_grad():
    all_logits = classifier(device_embeddings)
    all_predicted = torch.argmax(F.softmax(all_logits, dim=1), dim=1)

# --- Evaluation Metrics (only on test set) ---
test_logits = all_logits[test_idx]
test_predicted = all_predicted[test_idx]

true_labels = test_labels.cpu().numpy()
predicted_labels = test_predicted.cpu().numpy()

accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, zero_division=0)
recall = recall_score(true_labels, predicted_labels, zero_division=0)

print(f"\nEvaluation Metrics (Test Set):")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")

# --- Update predictions back to DB (for all devices) ---
for i, prediction in enumerate(all_predicted):
    color = "red" if prediction.item() == 1 else "green"
    cell_key = device_keys[i]
    try:
        traffic_col.update({
            "_key": cell_key,
            "color": color,
            "last_congestion_update": datetime.now().strftime("%m/%d/%Y %H:%M:%S")
        })
        print(f"Updated device {cell_key}: prediction → {color}")
    except DocumentUpdateError as e:
        print(f"Failed to update prediction for {cell_key}: {e}")
