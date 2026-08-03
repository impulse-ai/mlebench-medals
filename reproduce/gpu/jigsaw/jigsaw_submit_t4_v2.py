import os, sys, time
REGION = sys.argv[1] if len(sys.argv) > 1 else "us-central1"
PROJECT = "engg-ai-experimental"
STAGING = "gs://engg-ai-experimental-gpu-artifacts"
SA = "gpu-sandbox-sa-dev@engg-ai-experimental.iam.gserviceaccount.com"
# GPU image (has CUDA torch); the cheap-cpu default lacks a CUDA build.
IMAGE = f"{REGION}-docker.pkg.dev/{PROJECT}/gpu-sandbox/gpu-sandbox-cheap-gpu:latest"

from google.cloud import aiplatform
aiplatform.init(project=PROJECT, location=REGION, staging_bucket=STAGING)

code = open(os.path.expanduser("~/jigsaw_gpu_job_t4_v2.py")).read()
session_id = "jigsaw-t4v2-" + REGION.replace("-", "")

worker_pool_spec = [{
    "machine_spec": {
        "machine_type": "n1-standard-4",
        "accelerator_type": "NVIDIA_TESLA_T4",
        "accelerator_count": 1,
    },
    "replica_count": 1,
    "container_spec": {
        "image_uri": IMAGE,
        "command": ["python", "-c", code],
        "env": [{"name": "SESSION_ID", "value": session_id}],
    },
}]

job = aiplatform.CustomJob(
    display_name="jigsaw-t4v2-" + REGION.replace("-", "")[:8],
    worker_pool_specs=worker_pool_spec,
)
t0 = time.time()
# Job self-limits via internal wall budget; an external watchdog cancels as backstop.
job.submit(service_account=SA)
print("SUBMIT_SECONDS", round(time.time() - t0, 1))
print("IMAGE", IMAGE)
print("job_id:", job.resource_name)
