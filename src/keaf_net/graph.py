"""Heterogeneous graph utilities for Paper 3 KEAF-Net."""
from __future__ import annotations
import torch
import torch.nn.functional as F

EDGE_VISUAL_SPATIAL = 0
EDGE_TEXT_SEQUENTIAL = 1
EDGE_VISUAL_KNOWLEDGE = 2
EDGE_TEXT_KNOWLEDGE = 3
EDGE_VISUAL_TEXT_SEMANTIC = 4


def build_paper3_graph(visual, text, knowledge, visual_boxes=None, semantic_threshold=0.35):
    """Build the five edge families described in Paper 3.

    Exact entity co-reference requires dataset/retrieval metadata. In this
    feature-only builder, cross-modal edges use cosine similarity as a
    deterministic fallback; production dataset adapters should pass explicit
    entity links when available.
    """
    b,nv,d = visual.shape; nt=text.size(1); nk=knowledge.size(1); n=nv+nt+nk
    adj=torch.zeros(b,n,n,dtype=torch.bool,device=visual.device)
    et=torch.zeros(b,n,n,dtype=torch.long,device=visual.device)
    # Visual spatial adjacency: nearest/overlapping regions if boxes exist; otherwise dense visual context.
    if visual_boxes is None:
        adj[:,:nv,:nv]=True; et[:,:nv,:nv]=EDGE_VISUAL_SPATIAL
    else:
        x1=torch.maximum(visual_boxes[:,:,None,0],visual_boxes[:,None,:,0]); y1=torch.maximum(visual_boxes[:,:,None,1],visual_boxes[:,None,:,1])
        x2=torch.minimum(visual_boxes[:,:,None,2],visual_boxes[:,None,:,2]); y2=torch.minimum(visual_boxes[:,:,None,3],visual_boxes[:,None,:,3])
        overlap=((x2-x1).clamp_min(0)*(y2-y1).clamp_min(0))>0
        adj[:,:nv,:nv]=overlap; et[:,:nv,:nv]=EDGE_VISUAL_SPATIAL
    # Sequential text edges.
    for i in range(max(0,nt-1)):
        adj[:,nv+i,nv+i+1]=True; adj[:,nv+i+1,nv+i]=True
        et[:,nv+i,nv+i+1]=EDGE_TEXT_SEQUENTIAL; et[:,nv+i+1,nv+i]=EDGE_TEXT_SEQUENTIAL
    def connect(a, a0, z, z0, edge_type):
        sim=torch.einsum('bid,bjd->bij',F.normalize(a,dim=-1),F.normalize(z,dim=-1))
        mask=sim>=semantic_threshold
        adj[:,a0:a0+a.size(1),z0:z0+z.size(1)] |= mask
        adj[:,z0:z0+z.size(1),a0:a0+a.size(1)] |= mask.transpose(1,2)
        et[:,a0:a0+a.size(1),z0:z0+z.size(1)] = edge_type
        et[:,z0:z0+z.size(1),a0:a0+a.size(1)] = edge_type
    connect(visual,0,knowledge,nv+nt,EDGE_VISUAL_KNOWLEDGE)
    connect(text,nv,knowledge,nv+nt,EDGE_TEXT_KNOWLEDGE)
    connect(visual,0,text,nv,EDGE_VISUAL_TEXT_SEMANTIC)
    return adj,et
