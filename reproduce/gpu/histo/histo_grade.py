import json
from pathlib import Path
from google.cloud import storage
import pandas as pd

B = "engg-ai-experimental-gpu-artifacts"
sc = storage.Client(project="engg-ai-experimental")
bk = sc.bucket(B)
bk.blob("histo/out/submission.csv").download_to_filename("/tmp/histo_submission.csv")
bk.blob("histo/out/metrics.json").download_to_filename("/tmp/histo_metrics.json")
sub = pd.read_csv("/tmp/histo_submission.csv")
print("submission rows", len(sub), "cols", list(sub.columns), flush=True)
print("label stats min/mean/max",
      round(sub["label"].min(), 4), round(sub["label"].mean(), 4), round(sub["label"].max(), 4), flush=True)
print("METRICS", open("/tmp/histo_metrics.json").read(), flush=True)

from mlebench.registry import registry
from mlebench.grade import grade_csv
reg = registry.set_data_dir(Path.home() / ".cache/mle-bench/data")
comp = reg.get_competition("histopathologic-cancer-detection")
rep = grade_csv(Path("/tmp/histo_submission.csv"), comp)
rd = rep.to_dict() if hasattr(rep, "to_dict") else dict(rep.__dict__)
print("GRADE_REPORT", json.dumps(rd, default=str, indent=2), flush=True)
medal = ("GOLD" if rd.get("gold_medal") else "SILVER" if rd.get("silver_medal")
         else "BRONZE" if rd.get("bronze_medal") else
         ("ABOVE_MEDIAN" if rd.get("above_median") else "NONE"))
print("OFFICIAL_AUC", rd.get("score"), "MEDAL", medal, flush=True)
print("HISTO_GRADE_DONE", flush=True)
