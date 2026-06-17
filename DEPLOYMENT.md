# Deployment & Hosting Guide

This guide describes how to build, run, and host the **AgriPrice Prediction API** in a containerized production environment (e.g., Render, Railway, AWS ECS, GCP Cloud Run).

---

## 1. Local Containerized Build

To build and run the API locally using Docker:

```bash
# Build the Docker image
docker build -t agriprice-api .

# Run the container (mapping host port 8000 to container port 8000)
docker run -p 8000:8000 agriprice-api
```

---

## 2. Persistent Storage (Critical for SQLite)

The prediction API pulls live data and stores historical records in a local SQLite database (`data/feature_store.db`).
When deploying the container, you **must** configure a **persistent disk volume** mounted to `/app/data` inside the container. 

* If no volume is mounted, SQLite updates (like nightly API data ingestion updates) will be lost when the container restarts.
* The API container includes a custom `entrypoint.sh` that checks if the mounted volume is empty. If empty on first boot, it automatically initializes and seeds `data/feature_store.db` from local templates.

---

## 3. Deployment on Render (via Blueprint)

The repository includes a `render.yaml` configuration file for zero-config blueprint deployments.

1. Push this repository to GitHub.
2. Go to [Render Dashboard](https://dashboard.render.com).
3. Click **New** -> **Blueprint**.
4. Connect this GitHub repository.
5. Render will read `render.yaml`, set up the Docker service, provision a 1 GB persistent disk, and mount it to `/app/data` automatically.

---

## 4. Deployment on AWS ECS or GCP Cloud Run

When deploying on container platforms:

1. Build the Docker image and push it to a container registry (like AWS ECR or GCP Artifact Registry).
2. Create a service/task definition with:
   * **Container Port:** `8000`
   * **Persistent Storage Mount:** Mount a persistent directory (e.g. AWS EFS volume or GCP Filestore) to `/app/data` in the container.
