# -*- coding: utf-8 -*-
"""§I.5 research-suite trainers (embedding-only, GPU-light) — lifted from
smartwoodid_experiments_full.py:5526-6039.

Operates on the cached frozen-DINOv2 meta embeddings (see meta_embeddings.py), so
every function here is fast (seconds–minutes on cached 768-d vectors). Produces:
  - urd_v2_checkpoint_{suffix}_v2.pt          (URDProjectionHead)        L5696
  - sc_urd_checkpoint_{suffix}_v2.pt          (SC-URD residual/anchored)  L5771
  - sc_urd_checkpoint_{phase2}_v2.pt          (Phase-2 fine-tune)         L5891
  - b1_density_gate_v2.pt + _meta.json        (learned density gate)      L5526

Skip-if-exists with metadata validation (matches the monolith); pass force=True to
retrain. The SC-URD seed checkpoints (scurd_r01_e20_seed{42,43,44}) are produced by
the validated engine (_engines/variance_retrieval_evidence_colab.train_one_scurd_seed),
NOT here — see train_all.py.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..scurd import SCURDResidualHead
from .heads import URDProjectionHead, B1DensityGate, make_scurd_model
from .samplers import urd_episode_indices, scurd_episode_indices, urd_logits
from . import rs_eval
from . import config as TC
from .. import config as C

RESEARCH_DIR = C.RESEARCH_DIR
RS_TAU = TC.RS_TAU


def _json_dump(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
    print(f"  ✅ Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════
# URD-v2 — episodic projection head on frozen DINOv2  (L5696-5760)
# ══════════════════════════════════════════════════════════════════════════
def train_urd_v2(meta, device, lambda_cons=0.5, suffix="full_e10",
                 epochs=10, episodes=500, force=False):
    ckpt_path = RESEARCH_DIR / f"urd_v2_checkpoint_{suffix}_{TC.RESEARCH_CACHE_VERSION}.pt"
    log_path = RESEARCH_DIR / f"urd_v2_train_log_{suffix}_{TC.RESEARCH_CACHE_VERSION}.json"
    in_dim = int(meta["train_weak"].shape[1])
    model = URDProjectionHead(in_dim, out_dim=512).to(device)
    expected_meta = {"method": "urd_v2", "epochs": int(epochs), "episodes_per_epoch": int(episodes),
                     "lambda_cons": float(lambda_cons), "cache_version": TC.RESEARCH_CACHE_VERSION}
    if ckpt_path.exists() and not force:
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
            if not all(ckpt.get(k) == v for k, v in expected_meta.items()):
                raise RuntimeError(f"metadata mismatch (found epochs={ckpt.get('epochs')})")
            model.load_state_dict(ckpt["model_state_dict"]); model.eval()
            print(f"  ✅ Loaded URD-v2 checkpoint: {ckpt_path.name}")
            return ckpt_path
        except Exception as e:
            print(f"  ⚠️ URD-v2 checkpoint stale ({e}); retraining.")

    train_w = torch.tensor(meta["train_weak"], dtype=torch.float32)
    train_s = torch.tensor(meta["train_strong"], dtype=torch.float32)
    labels = meta["train_labels"]
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    n_way, k_support, q_query = TC.N_WAY, TC.K_SUPPORT, TC.Q_QUERY
    rng = np.random.RandomState(2026)
    train_log, best_loss, best_state = [], float("inf"), None
    for ep in range(1, epochs + 1):
        model.train(); losses, cls_l, cons_l = [], [], []
        for _ in range(episodes):
            sup_idx, qry_idx, yq, ys = urd_episode_indices(labels, n_way, k_support, q_query, rng)
            yq = yq.to(device); ys = ys.to(device)
            support_z = model(train_w[sup_idx].to(device))
            qw = model(train_w[qry_idx].to(device))
            qs = model(train_s[qry_idx].to(device))
            n_way_eff = int(yq.max().item() + 1)
            logits_w = urd_logits(qw, support_z, ys, n_way_eff, RS_TAU)
            logits_s = urd_logits(qs, support_z, ys, n_way_eff, RS_TAU)
            loss_cls = F.cross_entropy(logits_w, yq)
            pw = F.log_softmax(logits_w, dim=1); ps = F.log_softmax(logits_s, dim=1)
            loss_cons = 0.5 * (F.kl_div(pw, ps.exp(), reduction="batchmean")
                               + F.kl_div(ps, pw.exp(), reduction="batchmean"))
            loss = loss_cls + float(lambda_cons) * loss_cons
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss.item())); cls_l.append(float(loss_cls.item())); cons_l.append(float(loss_cons.item()))
        rec = {"epoch": ep, "loss": float(np.mean(losses)), "cls_loss": float(np.mean(cls_l)),
               "cons_loss": float(np.mean(cons_l)), "lambda_cons": float(lambda_cons)}
        train_log.append(rec)
        print(f"  URD {suffix} ep {ep}/{epochs}: loss={rec['loss']:.4f} cls={rec['cls_loss']:.4f} cons={rec['cons_loss']:.4f}")
        if rec["loss"] < best_loss:
            best_loss = rec["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval()
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "lambda_cons": lambda_cons,
                "in_dim": in_dim, "out_dim": 512, "training_complete": True,
                **expected_meta}, ckpt_path)
    _json_dump({"config": {"n_way": n_way, "k_support": k_support, "q_query": q_query,
                           "epochs": epochs, "episodes_per_epoch": episodes,
                           "lambda_cons": float(lambda_cons), "tau": RS_TAU},
                "train_log": train_log, "checkpoint": str(ckpt_path)}, log_path)
    return ckpt_path


# ══════════════════════════════════════════════════════════════════════════
# SC-URD — gallery-conditioned residual / anchored head  (L5771-5888)
# ══════════════════════════════════════════════════════════════════════════
def train_sc_urd(meta, device, beta=0.1, learnable_beta=False, use_ood_cons=False,
                 lambda_cons=0.5, suffix="r01", epochs=10, episodes=500,
                 head_type="residual", teacher_cons_eta=0.0, force=False):
    ckpt_path = RESEARCH_DIR / f"sc_urd_checkpoint_{suffix}_{TC.SC_URD_CACHE_VERSION}.pt"
    log_path = RESEARCH_DIR / f"sc_urd_train_log_{suffix}_{TC.SC_URD_CACHE_VERSION}.json"
    in_dim = int(meta["train_weak"].shape[1])
    beta_meta = "learnable" if learnable_beta else float(beta)
    model = make_scurd_model(in_dim, head_type=head_type,
                             beta=(0.1 if learnable_beta else beta),
                             learnable_beta=learnable_beta, device=device)
    expected_meta = {
        "method": "sc_urd", "head_type": str(head_type), "epochs": int(epochs),
        "episodes_per_epoch": int(episodes), "lambda_cons": float(lambda_cons),
        "cache_version": TC.SC_URD_CACHE_VERSION, "research_cache_version": TC.RESEARCH_CACHE_VERSION,
        "teacher_cons_eta": float(teacher_cons_eta), "beta": beta_meta,
        "learnable_beta": bool(learnable_beta), "use_ood_cons": bool(use_ood_cons),
    }
    if ckpt_path.exists() and not force:
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
            if not all(ckpt.get(k) == v for k, v in expected_meta.items()):
                raise RuntimeError("metadata mismatch")
            model.load_state_dict(ckpt["model_state_dict"]); model.eval()
            print(f"  ✅ Loaded SC-URD checkpoint: {ckpt_path.name}")
            return ckpt_path
        except Exception as e:
            print(f"  ⚠️ SC-URD checkpoint stale ({e}); retraining.")

    train_w = torch.tensor(meta["train_weak"], dtype=torch.float32)
    train_s = torch.tensor(meta["train_strong"], dtype=torch.float32)
    labels = meta["train_labels"]
    opt = torch.optim.AdamW(model.parameters(), lr=TC.SCURD_TRAIN_LR, weight_decay=TC.SCURD_WEIGHT_DECAY)
    n_way, k_support, q_query, q_ood = TC.N_WAY, TC.K_SUPPORT, TC.Q_QUERY, TC.Q_OOD
    rng = np.random.RandomState(2031)
    train_log, best_loss, best_state = [], float("inf"), None
    for ep in range(1, epochs + 1):
        model.train(); losses, cls_l, con_l, idc_l, oodc_l, tea_l, accs = [], [], [], [], [], [], []
        for _ in range(episodes):
            sup_idx, qry_idx, ood_idx, yq, ys = scurd_episode_indices(
                labels, n_way, k_support, q_query, q_ood, rng)
            yq = yq.to(device); ys = ys.to(device)
            x_sup = train_w[sup_idx].to(device)
            x_qw = train_w[qry_idx].to(device)
            x_qs = train_s[qry_idx].to(device)
            support_z = model(x_sup); qw = model(x_qw); qs = model(x_qs)
            n_way_eff = int(yq.max().item() + 1)
            logits_w = urd_logits(qw, support_z, ys, n_way_eff, RS_TAU)
            logits_s = urd_logits(qs, support_z, ys, n_way_eff, RS_TAU)
            loss_cls = F.cross_entropy(logits_w, yq)
            pw = F.log_softmax(logits_w, dim=1); ps = F.log_softmax(logits_s, dim=1)
            loss_id_cons = 0.5 * (F.kl_div(pw, ps.exp(), reduction="batchmean")
                                  + F.kl_div(ps, pw.exp(), reduction="batchmean"))
            loss_teacher = torch.tensor(0.0, device=device)
            if float(teacher_cons_eta) > 0:
                with torch.no_grad():
                    t_sup = F.normalize(x_sup, dim=1)
                    t_qw = F.normalize(x_qw, dim=1)
                    t_qs = F.normalize(x_qs, dim=1)
                    pt_w = F.softmax(urd_logits(t_qw, t_sup, ys, n_way_eff, RS_TAU), dim=1)
                    pt_s = F.softmax(urd_logits(t_qs, t_sup, ys, n_way_eff, RS_TAU), dim=1)
                loss_teacher = 0.5 * (F.kl_div(pw, pt_w, reduction="batchmean")
                                      + F.kl_div(ps, pt_s, reduction="batchmean"))
            loss_ood = torch.tensor(0.0, device=device)
            if use_ood_cons and len(ood_idx) > 0:
                qo_w = model(train_w[ood_idx].to(device))
                logits_ow = urd_logits(qo_w, support_z, ys, n_way_eff, RS_TAU)
                prob_o = F.softmax(logits_ow, dim=1)
                entropy_o = -(prob_o * torch.log(torch.clamp(prob_o, min=1e-8))).sum(dim=1)
                entropy_margin = TC.SC_URD_ENTROPY_MARGIN_FRAC * np.log(max(n_way_eff, 2))
                loss_ood_reject = F.relu(
                    torch.tensor(entropy_margin, dtype=torch.float32, device=device) - entropy_o).mean()
                loss_ood = TC.SC_URD_GAMMA_OOD * loss_ood_reject
            loss_con = loss_id_cons + float(teacher_cons_eta) * loss_teacher + loss_ood
            loss = loss_cls + float(lambda_cons) * loss_con
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                acc = float((logits_w.argmax(1) == yq).float().mean().item())
            losses.append(float(loss.item())); cls_l.append(float(loss_cls.item()))
            con_l.append(float(loss_con.item())); idc_l.append(float(loss_id_cons.item()))
            oodc_l.append(float(loss_ood.item())); tea_l.append(float(loss_teacher.item())); accs.append(acc)
        rec = {"epoch": ep, "loss": float(np.mean(losses)), "cls_loss": float(np.mean(cls_l)),
               "con_loss": float(np.mean(con_l)), "id_cons_loss": float(np.mean(idc_l)),
               "ood_cons_loss": float(np.mean(oodc_l)), "teacher_cons_loss": float(np.mean(tea_l)),
               "teacher_cons_eta": float(teacher_cons_eta), "episode_acc": float(np.mean(accs)),
               "lambda_cons": float(lambda_cons), "use_ood_cons": bool(use_ood_cons),
               "head_type": str(head_type), "beta": beta_meta,
               "beta_value": float(model.beta_value().detach().cpu().item())}
        train_log.append(rec)
        print(f"  SC-URD {suffix} ep {ep}/{epochs}: loss={rec['loss']:.4f} "
              f"cls={rec['cls_loss']:.4f} con={rec['con_loss']:.4f} acc={rec['episode_acc']:.3f}")
        if rec["loss"] < best_loss:
            best_loss = rec["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval()
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "in_dim": in_dim,
                "training_complete": True, **expected_meta}, ckpt_path)
    _json_dump({"config": {"n_way": n_way, "k_support": k_support, "q_query": q_query, "q_ood": q_ood,
                           "epochs": epochs, "episodes_per_epoch": episodes, "lambda_cons": float(lambda_cons),
                           "tau": RS_TAU, "use_ood_cons": bool(use_ood_cons), "head_type": str(head_type),
                           "teacher_cons_eta": float(teacher_cons_eta), "beta": beta_meta},
                "train_log": train_log, "checkpoint": str(ckpt_path)}, log_path)
    return ckpt_path


# ══════════════════════════════════════════════════════════════════════════
# SC-URD Phase 2 — cross-domain episodic fine-tune (SWI + non-top50 OOD)  (L5891-6039)
# ══════════════════════════════════════════════════════════════════════════
def train_sc_urd_phase2(meta, ood_weak, ood_labels, top50, device,
                        phase1_suffix="scurd_r01_e10", phase2_suffix="scurd_p2_8s8o_e10",
                        n_way_swi=8, n_way_ood=8, k_support=5, q_query=4,
                        epochs=10, episodes=500, lambda_cons=0.3, lr=3e-4,
                        noise_scale=0.05, force=False):
    p1_path = RESEARCH_DIR / f"sc_urd_checkpoint_{phase1_suffix}_{TC.SC_URD_CACHE_VERSION}.pt"
    p2_path = RESEARCH_DIR / f"sc_urd_checkpoint_{phase2_suffix}_{TC.SC_URD_CACHE_VERSION}.pt"
    p2_log = RESEARCH_DIR / f"sc_urd_train_log_{phase2_suffix}_{TC.SC_URD_CACHE_VERSION}.json"
    in_dim = int(meta["train_weak"].shape[1])

    if p2_path.exists() and not force:
        try:
            ckpt = torch.load(p2_path, map_location=device)
            model = SCURDResidualHead(in_dim, out_dim=512, beta=0.1, learnable_beta=False).to(device)
            model.load_state_dict(ckpt["model_state_dict"]); model.eval()
            print(f"  ✅ Loaded Phase2 checkpoint: {p2_path.name}")
            return p2_path
        except Exception as e:
            print(f"  ⚠️ Phase2 checkpoint stale ({e}); retraining.")
    if not p1_path.exists():
        raise RuntimeError(f"Phase 1 checkpoint not found: {p1_path}")
    p1_ckpt = torch.load(p1_path, map_location=device)

    swi_weak = torch.tensor(meta["train_weak"], dtype=torch.float32)
    swi_strong = torch.tensor(meta["train_strong"], dtype=torch.float32)
    swi_labels = meta["train_labels"]

    top50_set = set(top50)
    ood_labels = np.asarray(ood_labels)
    _ood_mask = np.array([l not in top50_set for l in ood_labels])
    ood_labels_p2 = ood_labels[_ood_mask]
    ood_train_emb = torch.tensor(ood_weak[_ood_mask], dtype=torch.float32)

    swi_by_sp = {sp: np.where(swi_labels == sp)[0] for sp in np.unique(swi_labels)}
    ood_by_sp = {sp: np.where(ood_labels_p2 == sp)[0] for sp in np.unique(ood_labels_p2)}
    valid_swi = [sp for sp, idx in swi_by_sp.items() if len(idx) >= k_support + q_query]
    valid_ood = [sp for sp, idx in ood_by_sp.items() if len(idx) >= k_support + q_query]
    n_way_ood_eff = min(n_way_ood, len(valid_ood))
    n_way_swi_eff = min(n_way_swi, len(valid_swi))
    print(f"  Phase2: SWI={len(valid_swi)}sp, OOD non-top50={len(valid_ood)}sp | "
          f"n_way={n_way_swi_eff}swi+{n_way_ood_eff}ood epochs={epochs} λ={lambda_cons} lr={lr}")

    model = SCURDResidualHead(in_dim, out_dim=512, beta=0.1, learnable_beta=False).to(device)
    model.load_state_dict(p1_ckpt["model_state_dict"])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.RandomState(2099)
    train_log, best_loss, best_state = [], float("inf"), None

    for ep in range(1, epochs + 1):
        model.train(); losses, cls_l, cons_l, accs = [], [], [], []
        for _ in range(episodes):
            swi_ch = rng.choice(valid_swi, n_way_swi_eff, replace=False)
            ood_ch = rng.choice(valid_ood, n_way_ood_eff, replace=False)
            sup_si, qry_si, ys_swi, yq_swi = [], [], [], []
            sup_oi, qry_oi, ys_ood, yq_ood = [], [], [], []
            for ci, sp in enumerate(swi_ch):
                idx = swi_by_sp[sp].copy(); rng.shuffle(idx)
                ns = min(k_support, max(1, len(idx) // 2)); nq = min(q_query, len(idx) - ns)
                sup_si.extend(idx[:ns].tolist()); qry_si.extend(idx[ns:ns + nq].tolist())
                ys_swi.extend([ci] * ns); yq_swi.extend([ci] * nq)
            offset = len(swi_ch)
            for ci, sp in enumerate(ood_ch):
                idx = ood_by_sp[sp].copy(); rng.shuffle(idx)
                ns = min(k_support, max(1, len(idx) // 2)); nq = min(q_query, len(idx) - ns)
                sup_oi.extend(idx[:ns].tolist()); qry_oi.extend(idx[ns:ns + nq].tolist())
                ys_ood.extend([ci + offset] * ns); yq_ood.extend([ci + offset] * nq)
            ys_all = torch.tensor(ys_swi + ys_ood, dtype=torch.long, device=device)
            yq_all = torch.tensor(yq_swi + yq_ood, dtype=torch.long, device=device)
            n_way_eff = int(ys_all.max().item() + 1)
            sup_si, qry_si = np.array(sup_si), np.array(qry_si)
            sup_oi, qry_oi = np.array(sup_oi), np.array(qry_oi)
            x_sup = torch.cat([swi_weak[sup_si].to(device), ood_train_emb[sup_oi].to(device)], dim=0)
            x_qw_s = swi_weak[qry_si].to(device)
            x_qs_s = swi_strong[qry_si].to(device)
            x_qw_o = ood_train_emb[qry_oi].to(device)
            sup_z = model(x_sup); qw_swi = model(x_qw_s); qs_swi = model(x_qs_s); qw_ood = model(x_qw_o)
            all_qw = torch.cat([qw_swi, qw_ood], dim=0)
            logits_all = urd_logits(all_qw, sup_z, ys_all, n_way_eff, RS_TAU)
            L_cls = F.cross_entropy(logits_all, yq_all)
            acc = float((logits_all.argmax(1) == yq_all).float().mean().item())
            lg_w = urd_logits(qw_swi, sup_z, ys_all, n_way_eff, RS_TAU)
            lg_s = urd_logits(qs_swi, sup_z, ys_all, n_way_eff, RS_TAU)
            pw = F.log_softmax(lg_w, dim=1); ps = F.log_softmax(lg_s, dim=1)
            L_cons = 0.5 * (F.kl_div(pw, ps.exp(), reduction="batchmean")
                            + F.kl_div(ps, pw.exp(), reduction="batchmean"))
            loss = L_cls + float(lambda_cons) * L_cons
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss)); cls_l.append(float(L_cls)); cons_l.append(float(L_cons)); accs.append(acc)
        rec = {"epoch": ep, "loss": float(np.mean(losses)), "cls_loss": float(np.mean(cls_l)),
               "cons_loss": float(np.mean(cons_l)), "episode_acc": float(np.mean(accs))}
        train_log.append(rec)
        print(f"  P2 {phase2_suffix} ep {ep}/{epochs}: loss={rec['loss']:.4f} "
              f"cls={rec['cls_loss']:.4f} cons={rec['cons_loss']:.4f} acc={rec['episode_acc']:.3f}")
        if rec["loss"] < best_loss:
            best_loss = rec["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval()
    p2_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "in_dim": in_dim,
                "phase1_suffix": phase1_suffix, "phase2_suffix": phase2_suffix,
                "n_way_swi": n_way_swi_eff, "n_way_ood": n_way_ood_eff,
                "lambda_cons": lambda_cons, "lr": lr, "noise_scale": noise_scale,
                "epochs": epochs, "training_complete": True}, p2_path)
    _json_dump({"config": {"phase1": phase1_suffix, "n_way_swi": n_way_swi_eff,
                           "n_way_ood": n_way_ood_eff, "k_support": k_support, "q_query": q_query,
                           "epochs": epochs, "episodes": episodes, "lambda_cons": lambda_cons,
                           "lr": lr, "noise_scale": noise_scale, "n_ood_train_species": len(valid_ood)},
                "train_log": train_log}, p2_log)
    return p2_path


# ══════════════════════════════════════════════════════════════════════════
# B1 learned density gate  (L5499-5574)
# ══════════════════════════════════════════════════════════════════════════
B1_EXPERTS = ["ArcFace", "DINOv2", "ArcFace-ViT", "SupCon-ConvNeXt"]


def extract_b1_metaval_experts(manifest, models, transforms_out, device, cache_path,
                               experts=None, force=False):
    """Extract per-expert meta-val embeddings (ArcFace-557, DINOv2, ViT/SupCon variants).

    `models` is the dict from swid_retrieval.models.build_models (keys: arc, dinov2,
    Var_ViTB_Arc, Var_CvNxt_SupCon); transforms_out supplies the DINOv2 transform.
    """
    from ..data import ManifestDataset, get_transforms, canonical_label
    from torch.utils.data import DataLoader
    from .common import extract_embeddings
    from .meta_embeddings import extract_dino_dataset

    experts = experts or B1_EXPERTS
    cache_path = Path(cache_path)
    if cache_path.exists() and not force:
        try:
            c = np.load(cache_path, allow_pickle=False)
            embs = {e: c[f"{e.replace('-', '_')}__emb"] for e in experts}
            print(f"  ✅ Loaded B1 meta-val expert embeddings: {cache_path.name}")
            return embs, c["labels"]
        except Exception as e:
            print(f"  ⚠️ B1 meta-val cache stale ({e}); rebuilding.")

    print(f"  Extracting B1 meta-val expert embeddings → {cache_path}")
    items = [(p, canonical_label(sp)) for p, sp in manifest["meta-val"]]
    nw = C.num_workers()
    ds = ManifestDataset(items, transform=get_transforms(224, augment=False))
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=nw, pin_memory=True)
    labels = np.array([canonical_label(ds.idx_to_class[i]) for i in ds.get_labels()])
    out, embs = {"labels": labels}, {}
    expert_to_model = {"ArcFace": "arc", "ArcFace-ViT": "Var_ViTB_Arc",
                       "SupCon-ConvNeXt": "Var_CvNxt_SupCon"}
    for e in experts:
        if e == "DINOv2":
            embs[e], _ = extract_dino_dataset(models["dinov2"], items, transforms_out["dinov2"],
                                              device, "B1 DINOv2")
        else:
            embs[e], _ = extract_embeddings(models[expert_to_model[e]], loader, device)
        out[f"{e.replace('-', '_')}__emb"] = embs[e]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **out)
    return embs, labels


def train_b1_gate(embs, labels, device, experts=None, force=False):
    """Train the learned density gate on meta-val expert soft-retrieval features."""
    experts = experts or B1_EXPERTS
    ckpt_path = RESEARCH_DIR / f"b1_density_gate_{TC.RESEARCH_CACHE_VERSION}.pt"
    meta_path = RESEARCH_DIR / f"b1_density_gate_meta_{TC.RESEARCH_CACHE_VERSION}.json"
    if ckpt_path.exists() and meta_path.exists() and not force:
        meta = json.load(open(meta_path))
        if meta.get("experts") == list(experts):
            gate = B1DensityGate(meta["in_dim"], len(experts)).to(device)
            gate.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state_dict"])
            gate.eval()
            print(f"  ✅ Loaded B1 learned density gate: {ckpt_path.name}")
            return ckpt_path
        print("  ⚠️ B1 gate cache expert list differs; retraining.")

    support_idx, query_idx = rs_eval.make_species_split(labels, 5, 10, seed=777)
    class_order = np.array(sorted(np.unique(labels[support_idx])))
    feature_list, score_list = [], []
    for e in experts:
        ev = rs_eval.soft_retrieval_eval(embs[e][query_idx], labels[query_idx],
                                         embs[e][support_idx], labels[support_idx],
                                         top_m=TC.RS_TOP_M, tau=RS_TAU, class_order=class_order)
        feature_list.append(rs_eval.density_features(ev))
        score_list.append(ev["scores"])
    features = np.stack(feature_list, axis=1)
    scores = np.stack(score_list, axis=1)
    y = np.array([np.where(class_order == c)[0][0] for c in labels[query_idx]], dtype=np.int64)
    x_t = torch.tensor(features.reshape(features.shape[0], -1), dtype=torch.float32, device=device)
    scores_t = torch.tensor(scores, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.long, device=device)
    gate = B1DensityGate(x_t.shape[1], len(experts)).to(device)
    opt = torch.optim.AdamW(gate.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 30 if TC.FAST else 120
    best_acc, best_state = -1.0, None
    for _ep in range(1, epochs + 1):
        gate.train()
        w = gate(x_t)
        fused = (scores_t * w.unsqueeze(-1)).sum(dim=1)
        loss = F.nll_loss(torch.log(torch.clamp(fused, min=1e-8)), y_t)
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            acc = float((fused.argmax(1) == y_t).float().mean().item())
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in gate.state_dict().items()}
    gate.load_state_dict(best_state); gate.eval()
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": gate.state_dict(), "experts": experts}, ckpt_path)
    _json_dump({"experts": experts, "in_dim": int(x_t.shape[1]), "meta_val_acc": best_acc,
                "epochs": epochs, "cache_version": TC.RESEARCH_CACHE_VERSION}, meta_path)
    print(f"  ✅ Trained B1 learned density gate: meta-val acc={best_acc:.4f}")
    return ckpt_path
