"""Regression cases for source-confirmed DC3 audit defects.

All fixtures are synthetic. Passing these checks is software evidence, not a
real-data benchmark or validation of historical manuscript accuracies.
"""
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from keaf_net.data import AnswerVocab, random_batch
from keaf_net.data.grounding import (boxes_in_grid_frame, phrase_pairs_to_tokens,
                                     question_sha256, region_pairs_to_visual,
                                     validate_cache_alignment)
from keaf_net.engine import Evaluator
from keaf_net.engine.trainer import Trainer, deterministic_reasoning
from keaf_net.models import AdaptiveKnowledgeFilter, KEAFNet
from keaf_net.models.encoders import patch_boxes
from keaf_net.models.graph import build_sample_graph
from keaf_net.structures import EDGE_TK, EDGE_VK, KNOWLEDGE, TEXTUAL, VISUAL
from keaf_net.utils.metrics import (aokvqa_direct_accuracy, vqa_official_accuracy,
                                    vqa_soft_accuracy, wilcoxon_pvalue)
from keaf_net.utils.text import normalize_answer


@pytest.mark.parametrize("matches,expected", [(0,0), (1,.3), (2,.6), (3,.9), (4,1), (10,1)])
def test_vqa_leave_one_annotator_out(matches, expected):
    references = ["dog"] * matches + ["cat"] * (10-matches)
    assert vqa_soft_accuracy("dog", references) == pytest.approx(expected)


@pytest.mark.parametrize("matches,expected", [(0,0), (1,1/3), (2,2/3), (3,1), (10,1)])
def test_aokvqa_has_its_own_exact_agreement(matches, expected):
    assert aokvqa_direct_accuracy("dog", ["dog"]*matches + ["cat"]*(10-matches)) == pytest.approx(expected)
    assert aokvqa_direct_accuracy("The cat", ["cat"]*10) == 0


@pytest.mark.parametrize("raw,expected", [("red-blue","red blue"), ("1,000","1000"),
                                           ("couldve","could've"), ("two cats","2 cats"),
                                           ("3.14","3.14"), ("red - blue","red blue")])
def test_vqa_normalizer_primary_source_cases(raw, expected):
    assert normalize_answer(raw) == expected


def test_missing_references_and_misaligned_statistics_fail():
    with pytest.raises(ValueError, match="reference"):
        vqa_soft_accuracy("dog", [])
    with pytest.raises(ValueError, match="aligned"):
        wilcoxon_pvalue([1], [1,0])


def test_vqa_upstream_conditional_normalization_is_preserved():
    assert vqa_official_accuracy("cat", ["The cat"]*10) == 0
    assert vqa_official_accuracy("cat", ["The cat"]*3+["dog"]*7) == pytest.approx(.9)
    assert vqa_official_accuracy("red blue", ["red-blue"]*3+["dog"]*7) == pytest.approx(.9)


class FixedAnswers(torch.nn.Module):
    def forward(self, samples):
        return {"logits": torch.tensor([[10., 0.] for _ in samples])}


def _record(qid, answers, difficult=False):
    return {"meta": {"qid": qid, "answers": answers, "difficult_direct_answer": difficult}}


def test_aokvqa_population_and_exact_strings():
    samples = [_record("easy", ["cat"]*3+["dog"]*7),
               _record("hard", ["dog"]*10, difficult=True)]
    evaluator = Evaluator(FixedAnswers(), AnswerVocab(["cat", "dog"]), dataset="aokvqa")
    result = evaluator.evaluate([samples])
    assert result["accuracy"] == 1.0
    assert result["scored_questions"] == 1
    assert len(result["predictions"]) == 2
    assert evaluator.evaluate([[_record("capital", ["The cat"]*10)]])["accuracy"] == 0


def test_hidden_test_cannot_be_given_accuracy():
    evaluator = Evaluator(FixedAnswers(), AnswerVocab(["cat", "dog"]), dataset="aokvqa")
    result = evaluator.evaluate([[_record("test", [])]])
    assert result["accuracy"] is None
    assert result["missing_references"] is True


def test_aokvqa_missing_population_metadata_and_duplicate_ids_fail():
    evaluator = Evaluator(FixedAnswers(), AnswerVocab(["cat", "dog"]), dataset="aokvqa")
    with pytest.raises(ValueError, match="metadata"):
        evaluator.evaluate([[{"meta": {"qid":"q", "answers":["cat"]*10}}]])
    with pytest.raises(ValueError, match="duplicate"):
        evaluator.evaluate([[_record("q", ["cat"]*10), _record("q", ["cat"]*10)]])


def test_threshold_hard_gate_has_both_score_and_threshold_gradients():
    akf = AdaptiveKnowledgeFilter(8)
    with torch.no_grad():
        akf.score.weight.zero_()
        akf.score.bias.fill_(.2)
    nf = torch.randn(3,8)
    nt = torch.tensor([VISUAL,TEXTUAL,KNOWLEDGE])
    nb = torch.zeros(3,dtype=torch.long)
    _, gate, _, _ = akf(nf,nt,nb,torch.randn(1,8))
    assert gate.tolist() == [1.0]
    optimizer = torch.optim.SGD(akf.parameters(), lr=.01)
    before = akf.threshold.detach().clone()
    gate.sum().backward()
    assert torch.isfinite(akf.threshold_logit.grad)
    assert akf.threshold_logit.grad.abs() > 0
    assert akf.score.bias.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(before, akf.threshold.detach())


@pytest.mark.parametrize("threshold,temp", [(0,.1), (1,.1), (.5,0), (.5,-1)])
def test_invalid_filter_parameters_fail(threshold,temp):
    with pytest.raises(ValueError):
        AdaptiveKnowledgeFilter(8, threshold_init=threshold, gate_temperature=temp)


def test_semantic_links_reach_detector_regions_and_bert_wordpieces():
    # Four grid nodes precede detector rows cat=0, truck=1. Truck must be node5.
    vk = region_pairs_to_visual([(1,0)], num_grid=4, num_regions=2)
    offsets = torch.tensor([[0,0],[0,4],[5,7],[8,11],[12,16],[17,22],[23,28],[28,29],[0,0]])
    tk = phrase_pairs_to_tokens([(0,0)], [(12,22)], offsets)
    assert vk.tolist() == [[5,0]]
    assert tk.tolist() == [[4,0],[5,0]]
    v,t,k = torch.randn(6,8),torch.randn(9,8),torch.randn(1,8)
    _, _, edges, types = build_sample_graph(v, torch.tensor([[0.,0.,10.,10.]]*6), t,k,vk,tk)
    assert set(map(tuple,edges[:,types==EDGE_VK].t().tolist())) == {(5,15),(15,5)}
    assert set(map(tuple,edges[:,types==EDGE_TK].t().tolist())) == {(10,15),(15,10),(11,15),(15,11)}


def test_truncated_and_special_tokens_do_not_get_phrase_edges():
    offsets = torch.tensor([[0,0],[0,3],[0,0]])
    assert phrase_pairs_to_tokens([(0,0)],[(10,20)],offsets).shape == (0,2)
    with pytest.raises(ValueError):
        region_pairs_to_visual([(-1,0)],4,2)
    with pytest.raises(ValueError):
        phrase_pairs_to_tokens([(2,0)],[(0,3)],offsets)


def test_grid_and_detector_boxes_use_one_explicit_coordinate_frame():
    grid = {"box_coordinate_frame":"processed_pixels", "image_size":[224,224],
            "source_image_size":[640,480], "spatial_transform":"resize_only"}
    region = {"boxes":torch.tensor([[320.,240.,640.,480.]]),
              "box_coordinate_frame":"original_pixels", "image_size":[640,480]}
    assert torch.equal(boxes_in_grid_frame(region,grid), torch.tensor([[112.,112.,224.,224.]]))
    region["box_coordinate_frame"] = "normalized_xyxy"
    region["spatial_transform"] = "resize_only"
    region["boxes"] = torch.tensor([[.5,.5,1.,1.]])
    assert torch.equal(boxes_in_grid_frame(region,grid), torch.tensor([[112.,112.,224.,224.]]))
    region["spatial_transform"] = "letterbox"
    with pytest.raises(ValueError, match="geometry"):
        boxes_in_grid_frame(region,grid)
    del region["box_coordinate_frame"]
    with pytest.raises(ValueError, match="frame"):
        boxes_in_grid_frame(region,grid)


def test_legacy_or_stale_grounding_cache_fails_before_graph_construction():
    text = "What is the truck doing?"
    grid,region,q = {"feat":torch.zeros(4,8)}, {"feat":torch.zeros(2,16)}, {"tokens":torch.zeros(9,8)}
    with pytest.raises(ValueError, match="regenerate"):
        validate_cache_alignment({},grid,region,q,text)
    q["question_sha256"] = question_sha256(text)
    q["tokenizer_name"] = "bert-base-uncased"
    kg = {"alignment":{"version":2,"num_grid":4,"num_regions":2,"num_tokens":9,
                       "question_sha256":question_sha256(text), "tokenizer_name":"bert-base-uncased"},
          "vk_index_space":"visual_grid_then_region", "tk_index_space":"bert_token"}
    validate_cache_alignment(kg,grid,region,q,text)
    kg["alignment"]["tokenizer_name"] = "different-tokenizer"
    with pytest.raises(ValueError, match="tokenizer"):
        validate_cache_alignment(kg,grid,region,q,text)
    kg["alignment"]["tokenizer_name"] = "bert-base-uncased"
    with pytest.raises(ValueError):
        validate_cache_alignment(kg,grid,region,q,"different question")


def test_invalid_graph_pairs_do_not_silently_target_another_node_type():
    v,t,k = torch.randn(2,8),torch.randn(2,8),torch.randn(1,8)
    boxes = torch.tensor([[0.,0.,10.,10.]]*2)
    with pytest.raises(ValueError, match="index space"):
        build_sample_graph(v,boxes,t,k,torch.tensor([[2,0]]),torch.zeros(0,2,dtype=torch.long))


def test_patch_counts_are_exact_and_invalid_geometry_is_rejected():
    assert patch_boxes(224,16).shape == (196,4)
    assert patch_boxes(448,16).shape == (784,4)
    with pytest.raises(ValueError):
        patch_boxes(224,0)
    with pytest.raises(ValueError):
        patch_boxes(225,16)


def test_loo_same_evidence_control_is_deterministic_and_restores_modes(cfg):
    cfg.model.dropout = .5
    model = KEAFNet(cfg.model).train()
    out = model(random_batch(cfg.model,batch_size=2))
    with deterministic_reasoning(model):
        a,_ = model.reason(out["batch"],out["node_keep"].detach())
        b,_ = model.reason(out["batch"],out["node_keep"].detach())
        assert torch.equal(a,b)
        assert not a.requires_grad
    assert model.training and model.hgaf.training and model.mhsr.training


def test_loo_scores_remain_trainable_after_detached_targets(cfg):
    model = KEAFNet(cfg.model)
    # Ensure there are retained facts rather than a random all-discarded batch.
    with torch.no_grad():
        model.akf.score.bias.fill_(1.)
    trainer = Trainer(model,cfg)
    loss,log = trainer.compute_losses(random_batch(cfg.model,batch_size=2))
    loss.backward()
    assert log.akf >= 0
    assert model.akf.score.weight.grad is not None
    assert model.akf.threshold_logit.grad is not None
    assert torch.isfinite(model.akf.threshold_logit.grad)


def test_aokvqa_converter_preserves_population_fields(tmp_path):
    path = Path(__file__).parents[1]/"scripts"/"prepare_annotations.py"
    import sys
    sys.path.insert(0,str(path.parent))
    spec = importlib.util.spec_from_file_location("prepare_annotations_regression",path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    inp,out = tmp_path/"official.json",tmp_path/"converted.json"
    inp.write_text(json.dumps([{"question_id":"q", "image_id":1, "question":"Why?",
                               "direct_answers":["yes"], "difficult_direct_answer":True,
                               "choices":["yes","no"], "correct_choice_idx":0}]))
    module.convert_aokvqa(str(inp),"val2017",str(out))
    record = json.loads(out.read_text())[0]
    assert record["difficult_direct_answer"] is True
    assert record["choices"] == ["yes","no"]
