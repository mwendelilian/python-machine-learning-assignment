# Import the required libraries
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import Counter

# Prepare the customer data
customers = [
    [1000, 2],
    [1500, 3],
    [2000, 2],
    [5000, 6],
    [6000, 7],
    [5500, 5],
    [12000, 12],
    [15000, 14],
    [13000, 11],
    [3000, 4]
]

# Scale the customer data
scaler = StandardScaler()
scaled_customers = scaler.fit_transform(customers)

# Create the K-Means model
model = KMeans(n_clusters=3, random_state=42, n_init=10)

# Train the model and assign clusters
clusters = model.fit_predict(scaled_customers)

# Display each customer's cluster
for i, cluster in enumerate(clusters, start=1):
    print(f"Customer {i} belongs to Cluster {cluster}")

# Count customers in each cluster
cluster_counts = Counter(clusters)

print("\nTotal customers in each cluster:")
for cluster, count in sorted(cluster_counts.items()):
    print(f"Cluster {cluster}: {count} customers")
# Create a scatter plot of customer groups
import matplotlib.pyplot as plt

# Plot customers using monthly spending and monthly visits
plt.scatter(
    [customer[0] for customer in customers],
    [customer[1] for customer in customers],
    c=clusters,
    cmap="viridis",
    s=100
)

# Add labels and title
plt.xlabel("Monthly Spending (KES)")
plt.ylabel("Monthly Visits")
plt.title("Customer Segmentation Using K-Means")

# Display the cluster colour bar
plt.colorbar(label="Cluster")

# Display the graph
plt.grid(True)
plt.show()
