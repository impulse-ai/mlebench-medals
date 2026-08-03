import os, sys, asyncio, time
REGION = sys.argv[1] if len(sys.argv) > 1 else "us-central1"
os.environ["GCP_PROJECT_ID"] = "engg-ai-experimental"
os.environ["GCP_REGION"] = REGION
os.environ["VERTEX_STAGING_BUCKET"] = "gs://engg-ai-experimental-gpu-artifacts"
sys.path.insert(0, os.path.expanduser("~/impulse/agent-api"))
import tools.vertex_compute as vc
print("REGION", vc.REGION, "STAGING", vc.STAGING_BUCKET)
code = open(os.path.expanduser("~/histo_gpu_job_t4.py")).read()
t0 = time.time()
out = asyncio.run(vc.submit_vertex_job(
    code=code,
    runtime_profile=None,
    auto_profile=False,
    machine_type="n1-standard-4",
    gpu_type="NVIDIA_TESLA_T4",
    gpu_count=1,
    timeout_hours=1.5,
    per_job_max_usd=5.0,
    session_id="histo-t4-" + REGION.replace("-", "")))
print("SUBMIT_SECONDS", round(time.time() - t0, 1))
print("JOB_OUT", out)
