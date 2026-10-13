"""Save synthetic software verification; never report benchmark accuracy."""
from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import xml.etree.ElementTree as ET

import torch

from keaf_net.config import Config, ModelConfig
from keaf_net.data import random_batch
from keaf_net.engine import Trainer
from keaf_net.models import KEAFNet
from keaf_net.utils import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="verification/DC3_SOFTWARE_VERIFICATION.json")
    parser.add_argument("--junit", default=None)
    args = parser.parse_args()
    torch.set_num_threads(1)
    set_seed(0)
    cfg = Config(model=ModelConfig(hidden_dim=32, region_feat_dim=64,
                                  knowledge_feat_dim=16, gat_heads=4,
                                  num_answers=10, akf_loo_samples=2, dropout=0), device="cpu")
    model = KEAFNet(cfg.model)
    with torch.no_grad():
        model.akf.score.bias.fill_(1.)
    samples = random_batch(cfg.model,batch_size=2)
    out = model(samples)
    trainer = Trainer(model,cfg)
    loss,log = trainer.compute_losses(samples)
    loss.backward()
    gradient = float(model.akf.threshold_logit.grad)
    assert gradient != 0 and torch.isfinite(model.akf.threshold_logit.grad)
    before = float(model.akf.threshold.detach())
    trainer.opt.step()
    after = float(model.akf.threshold.detach())
    assert before != after
    with torch.no_grad():
        paper_width = KEAFNet(ModelConfig())
        projected = paper_width.knowledge_proj(torch.zeros(2,384))
        assert projected.shape == (2,768)
    test_result = None
    if args.junit:
        suites = ET.parse(args.junit).getroot()
        test_result = {key:sum(int(s.attrib.get(key,0)) for s in suites.findall("testsuite"))
                       for key in ("tests","failures","errors","skipped")}
    report = {
        "generated_utc":datetime.now(timezone.utc).isoformat(),
        "status":"software_verified_on_synthetic_fixtures", "benchmark_completed":False,
        "canonical_repository":"https://github.com/junnubabu-ctrl/KEAF-Net",
        "base_commit":"c8a07e47cbf9084a8e2e52d65347e34fab022104",
        "git_head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "working_tree_has_repair":True,
        "environment":{"python":platform.python_version(), "torch":torch.__version__,
                       "cuda_available":torch.cuda.is_available(), "verification_device":"cpu"},
        "tests":test_result, "config":asdict(cfg),
        "synthetic_probe":{"samples":2,"answer_logits_shape":list(out["logits"].shape),
                           "threshold_gradient":gradient,"threshold_before":before,"threshold_after":after,
                           "loss":log.total,"akf_loss":log.akf},
        "interfaces":{"MiniLM_projection":[384,768],
                      "HGAF_output":"fused node collection [N,hidden_dim]",
                      "MHSR_input":"same node collection, batch IDs, question CLS and keep mask",
                      "historical_interfaces_authenticated":False},
        "historical_benchmark_results":"not yet independently reproduced",
        "real_data_execution":False,
        "required_next":["Rebuild version-2 grid/question/knowledge caches with detector coordinate provenance",
                         "Retrain corrected revision with official dataset splits and pinned encoders/knowledge",
                         "Retain complete checkpoints, predictions and official evaluator outputs",
                         "Run matched ablations, multi-seed uncertainty and robustness/cost evaluation"],
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"out":str(path),"status":report["status"],"tests":test_result,
                      "threshold_gradient":gradient,"benchmark_completed":False}))


if __name__ == "__main__":
    main()
