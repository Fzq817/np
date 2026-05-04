import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import check_tensor_valid

def masked_categorical(logits, valid_indices, total_size=None):
    mask = torch.full_like(logits, -1e8)
    for idx in valid_indices:
        if idx < logits.size(-1):
            mask[..., idx] = logits[..., idx]
    return torch.distributions.Categorical(logits=mask)

class PreferenceConditionedGATLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads=4, pref_dim=3, concat_heads=True):
        super().__init__()
        self.num_heads = num_heads
        self.out_per_head = out_features // num_heads if concat_heads else out_features
        self.concat_heads = concat_heads

        self.W = nn.Linear(in_features, self.out_per_head * num_heads, bias=False)
        self.a_src = nn.Parameter(torch.randn(num_heads, self.out_per_head))
        self.a_dst = nn.Parameter(torch.randn(num_heads, self.out_per_head))
        self.pref_attn = nn.Linear(pref_dim, num_heads, bias=False)

        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_normal_(self.a_src)
        nn.init.xavier_normal_(self.a_dst)

    def forward(self, x, adj, pref):
        B, N, _ = x.shape
        h = self.W(x)
        h = h.view(B, N, self.num_heads, self.out_per_head)

        src_score = (h * self.a_src).sum(dim=-1)
        dst_score = (h * self.a_dst).sum(dim=-1)

        pref_bias = self.pref_attn(pref)

        e = src_score.unsqueeze(2) + dst_score.unsqueeze(1) + pref_bias.unsqueeze(1).unsqueeze(1)
        e = F.leaky_relu(e, 0.2)

        adj_expanded = adj.unsqueeze(-1)
        e = e.masked_fill(adj_expanded == 0, -1e9)
        alpha = F.softmax(e, dim=2)

        h_perm = h.permute(0, 2, 1, 3)
        alpha_perm = alpha.permute(0, 3, 1, 2)
        out = torch.matmul(alpha_perm, h_perm)
        out = out.permute(0, 2, 1, 3)

        if self.concat_heads:
            out = out.reshape(B, N, self.num_heads * self.out_per_head)
        else:
            out = out.mean(dim=2)

        return out

class HeteroPreferenceGATEncoder(nn.Module):
    def __init__(self, feat_dim=10, hidden=128, num_heads=4, pref_dim=3):
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden = hidden
        self.pref_dim = pref_dim

        self.proj_job = nn.Linear(feat_dim, hidden)
        self.proj_resource = nn.Linear(feat_dim, hidden)

        self.gat_layer1 = PreferenceConditionedGATLayer(
            hidden, hidden, num_heads=num_heads, pref_dim=pref_dim, concat_heads=True
        )
        self.norm1 = nn.LayerNorm(hidden)

        self.gat_layer2 = PreferenceConditionedGATLayer(
            hidden, hidden, num_heads=num_heads, pref_dim=pref_dim, concat_heads=False
        )
        self.norm2 = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, hidden)

        self.graph_proj = nn.Linear(hidden * 2 + pref_dim, hidden)

    def forward(self, features, adj, types, pref=None):
        B, N, _ = features.shape

        if pref is None:
            pref = torch.ones(B, self.pref_dim, device=features.device) / self.pref_dim
        if pref.dim() == 1:
            pref = pref.unsqueeze(0).expand(B, -1)

        job_mask = (types == 0).unsqueeze(-1).float()
        resource_mask = (types == 1).unsqueeze(-1).float()

        h_job = self.proj_job(features)
        h_resource = self.proj_resource(features)
        h = h_job * job_mask + h_resource * resource_mask

        h1 = self.gat_layer1(h, adj, pref)
        h1 = self.norm1(h1)
        h1 = F.elu(h1) + h

        h2 = self.gat_layer2(h1, adj, pref)
        h2 = self.norm2(h2)
        h2 = F.elu(h2)
        h2 = self.out_proj(h2) + h1

        node_embeddings = h2

        job_mask_2d = (types == 0).float()
        resource_mask_2d = (types == 1).float()

        job_count = job_mask_2d.sum(dim=1, keepdim=True).clamp(min=1)
        resource_count = resource_mask_2d.sum(dim=1, keepdim=True).clamp(min=1)

        job_pool = (node_embeddings * job_mask_2d.unsqueeze(-1)).sum(dim=1) / job_count
        resource_pool = (node_embeddings * resource_mask_2d.unsqueeze(-1)).sum(dim=1) / resource_count

        graph_input = torch.cat([job_pool, resource_pool, pref], dim=-1)
        graph_embedding = F.relu(self.graph_proj(graph_input))

        return node_embeddings, graph_embedding

class SimpleActorCritic(nn.Module):
    def __init__(self, state_dim=128, num_factories=2, num_speeds=5,
                 max_candidate_jobs=5):
        super().__init__()
        self.num_factories = num_factories
        self.num_speeds = num_speeds
        self.max_candidate_jobs = max_candidate_jobs

        self.shared_mlp = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
        )

        self.job_embed = nn.Embedding(max_candidate_jobs, 16)
        self.factory_embed = nn.Embedding(num_factories, 8)

        self.job_head = nn.Linear(128, max_candidate_jobs)
        self.factory_head = nn.Linear(128 + 16, num_factories)
        self.speed_head = nn.Linear(128 + 16 + 8, num_speeds)

        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward_base(self, state, pref=None):
        fused = self.shared_mlp(state)
        value = self.critic(fused)
        return fused, value

    def forward(self, state, pref=None):
        fused, value = self.forward_base(state, pref)
        return fused, value

    def get_action(self, state, pref, valid_actions=None, deterministic=False):
        if state.dim() == 1:
            state = state.unsqueeze(0)

        shared_state, value = self.forward_base(state, pref)

        job_logits = self.job_head(shared_state)
        check_tensor_valid('job_logits', job_logits)

        if valid_actions:
            va = valid_actions[0] if isinstance(valid_actions, list) else valid_actions
            candidate_jobs = va['jobs']
            num_candidates = len(candidate_jobs)
            valid_job_indices = list(range(num_candidates))
            j_dist = masked_categorical(job_logits, valid_job_indices)
            job_idx = j_dist.probs.argmax(dim=-1) if deterministic else j_dist.sample()
            job_log_prob = j_dist.log_prob(job_idx)
            job_id = candidate_jobs[job_idx.item()]
        else:
            j_dist = torch.distributions.Categorical(logits=job_logits)
            job_idx = j_dist.probs.argmax(dim=-1) if deterministic else j_dist.sample()
            job_log_prob = j_dist.log_prob(job_idx)
            job_id = job_idx.item()

        job_emb = self.job_embed(job_idx)
        factory_logits = self.factory_head(torch.cat([shared_state, job_emb], dim=-1))
        check_tensor_valid('factory_logits', factory_logits)

        if valid_actions:
            per_job_facs = va.get('per_job_factories', {})
            if job_id in per_job_facs:
                valid_facs = per_job_facs[job_id]
            else:
                valid_facs = va['factories']
            f_dist = masked_categorical(factory_logits, valid_facs)
            factory = f_dist.probs.argmax(dim=-1) if deterministic else f_dist.sample()
            factory_log_prob = f_dist.log_prob(factory)
        else:
            f_dist = torch.distributions.Categorical(logits=factory_logits)
            factory = f_dist.probs.argmax(dim=-1) if deterministic else f_dist.sample()
            factory_log_prob = f_dist.log_prob(factory)

        fac_emb = self.factory_embed(factory)
        speed_logits = self.speed_head(torch.cat([shared_state, job_emb, fac_emb], dim=-1))
        check_tensor_valid('speed_logits', speed_logits)

        s_dist = torch.distributions.Categorical(logits=speed_logits)
        speed = s_dist.probs.argmax(dim=-1) if deterministic else s_dist.sample()
        speed_log_prob = s_dist.log_prob(speed)

        total_log_prob = job_log_prob + factory_log_prob + speed_log_prob
        check_tensor_valid('total_log_prob (sampling)', total_log_prob)

        scalarized = value

        action = {
            'job': job_id,
            'job_idx': job_idx.item(),
            'factory': factory.item(),
            'speed': speed.item()
        }

        return action, total_log_prob, scalarized, value

    def get_action_batch(self, state_batch, pref_batch, valid_actions_list, deterministic=False):
        B = state_batch.size(0)
        shared_state, value = self.forward_base(state_batch, pref_batch)
        job_logits = self.job_head(shared_state)

        actions = []
        for b in range(B):
            h_b = shared_state[b:b+1]
            jl_b = job_logits[b:b+1]

            va = valid_actions_list[b]
            candidate_jobs = va['jobs']
            num_candidates = len(candidate_jobs)
            valid_job_indices = list(range(num_candidates))
            j_dist = masked_categorical(jl_b, valid_job_indices)
            job_idx = j_dist.probs.argmax(dim=-1) if deterministic else j_dist.sample()
            job_id = candidate_jobs[job_idx.item()]

            job_emb = self.job_embed(job_idx)
            factory_logits = self.factory_head(torch.cat([h_b, job_emb], dim=-1))
            per_job_facs = va.get('per_job_factories', {})
            valid_facs = per_job_facs.get(job_id, va['factories'])
            f_dist = masked_categorical(factory_logits, valid_facs)
            factory = f_dist.probs.argmax(dim=-1) if deterministic else f_dist.sample()

            fac_emb = self.factory_embed(factory)
            speed_logits = self.speed_head(torch.cat([h_b, job_emb, fac_emb], dim=-1))
            s_dist = torch.distributions.Categorical(logits=speed_logits)
            speed = s_dist.probs.argmax(dim=-1) if deterministic else s_dist.sample()

            actions.append({
                'job': job_id,
                'job_idx': job_idx.item(),
                'factory': factory.item(),
                'speed': speed.item()
            })

        return actions

def build_models(config: dict, feat_dim: int, num_factories: int = 2,
                 device: torch.device = None):
    if device is None:
        device = torch.device('cpu')

    encoder = HeteroPreferenceGATEncoder(feat_dim=feat_dim)
    actor_critic = SimpleActorCritic(num_factories=num_factories)

    encoder = encoder.to(device)
    actor_critic = actor_critic.to(device)

    return encoder, actor_critic
