from mlebench.registry import registry
from mlebench.grade import grade_csv
from pathlib import Path
registry.set_data_dir(Path.home()/".cache"/"mle-bench"/"data")
comp = registry.get_competition("jigsaw-toxic-comment-classification-challenge")
r = grade_csv(Path.home()/"jigsaw_submission.csv", comp)
try:
    d = r.to_dict()
except Exception:
    d = r.__dict__
import json
print(json.dumps({k:(str(v) if not isinstance(v,(int,float,bool,type(None),str)) else v) for k,v in d.items()}, indent=2, default=str))
