"""
hierarchical_anfis.py
Hierarchical ANFIS with Explainable AI
3 Sub-models → Master ANFIS → Final Score + Explanation

Run alongside arduino_reader.py:
  py server.py
  py main_yolo.py
  py hierarchical_anfis.py   ← replaces arduino_reader.py

FIXES APPLIED:
  1. Grade/recommendation logic corrected (D ≠ healthy)
  2. Environment MF tuned so optimal inputs → 70–90%
  3. Chemistry rule: neutral pH + low NH3 → healthy (not acidic)
  4. Master score weighted sum verified and clipped correctly
  5. Master MF ranges aligned to actual sub-score distributions
  6. Population risk messaging added for low pop scores
"""

import numpy as np
import requests
import time
from itertools import product

# ══════════════════════════════════════════════
# MEMBERSHIP FUNCTIONS
# ══════════════════════════════════════════════
class GaussianMF:
    def __init__(self, center, sigma, label):
        self.c     = center
        self.sigma = sigma
        self.label = label

    def forward(self, x):
        return float(np.exp(-0.5 * ((x - self.c) / (self.sigma + 1e-8)) ** 2))

# ══════════════════════════════════════════════
# BASE ANFIS (single sub-model)
# ══════════════════════════════════════════════
class SubANFIS:
    def __init__(self, name, input_defs, n_mf=3, lr=0.005):
        self.name       = name
        self.input_defs = input_defs
        self.n_inputs   = len(input_defs)
        self.n_mf       = n_mf
        self.n_rules    = n_mf ** self.n_inputs
        self.lr         = lr

        self.mfs = []
        for inp_name, mf_list in input_defs:
            row = [GaussianMF(c, s, lbl) for c, s, lbl in mf_list]
            self.mfs.append((inp_name, row))

        self.C = np.random.randn(self.n_rules, self.n_inputs + 1) * 0.05
        self.rule_labels = self._build_rule_labels()

    def _build_rule_labels(self):
        labels = []
        for combo in product(*[range(self.n_mf)] * self.n_inputs):
            parts = []
            for inp_idx, mf_idx in enumerate(combo):
                inp_name = self.mfs[inp_idx][0]
                mf_label = self.mfs[inp_idx][1][mf_idx].label
                parts.append(f"{inp_name} is {mf_label}")
            labels.append(" AND ".join(parts))
        return labels

    def fuzzify(self, inputs):
        mu_all = []
        for i, (inp_name, mf_row) in enumerate(self.mfs):
            mu_all.append([mf.forward(inputs[i]) for mf in mf_row])
        return mu_all

    def forward(self, inputs):
        mu_all = self.fuzzify(inputs)

        w = []
        for combo in product(*[range(self.n_mf)] * self.n_inputs):
            strength = 1.0
            for inp_idx, mf_idx in enumerate(combo):
                strength *= mu_all[inp_idx][mf_idx]
            w.append(strength)

        w     = np.array(w)
        w_sum = w.sum() + 1e-8
        w_bar = w / w_sum

        x_aug  = np.append(inputs, 1.0)
        f      = self.C @ x_aug
        output = float(np.dot(w_bar, f))

        # FIX 4: Clip output to [0, 1] so sub-scores are always valid
        output = float(np.clip(output, 0.0, 1.0))

        return output, w, w_bar

    def train_step(self, inputs, target):
        output, w, w_bar = self.forward(inputs)
        error  = output - target
        x_aug  = np.append(inputs, 1.0)

        self.C -= self.lr * np.outer(w_bar, x_aug) * error

        f     = self.C @ x_aug
        w_sum = w.sum() + 1e-8
        for i, combo in enumerate(product(*[range(self.n_mf)] * self.n_inputs)):
            dE_dw = (f[i] - output) / w_sum * error
            for inp_idx, mf_idx in enumerate(combo):
                mf  = self.mfs[inp_idx][1][mf_idx]
                x_i = inputs[inp_idx]
                other_mu = 1.0
                for j, mf_j_idx in enumerate(combo):
                    if j != inp_idx:
                        other_mu *= self.mfs[j][1][mf_j_idx].forward(inputs[j])
                grad_c     = dE_dw * other_mu * (x_i - mf.c) / (mf.sigma ** 2 + 1e-8)
                grad_sigma = dE_dw * other_mu * (x_i - mf.c) ** 2 / (mf.sigma ** 3 + 1e-8)
                mf.c     -= self.lr * grad_c
                mf.sigma -= self.lr * grad_sigma
                mf.sigma  = max(0.01, mf.sigma)

        return abs(error)

    def explain(self, inputs, top_k=3):
        output, w, w_bar = self.forward(inputs)
        x_aug = np.append(inputs, 1.0)
        f     = self.C @ x_aug

        contributions = w_bar * f
        top_idx       = np.argsort(np.abs(contributions))[::-1][:top_k]

        explanations = []
        for idx in top_idx:
            explanations.append({
                "rule":         self.rule_labels[idx],
                "firing":       round(float(w_bar[idx]) * 100, 1),
                "contribution": round(float(contributions[idx]), 3),
                "impact":       "positive" if contributions[idx] > 0 else "negative",
            })

        return explanations, round(output, 4)

# ══════════════════════════════════════════════
# BUILD THE 3 SUB-MODELS
# ══════════════════════════════════════════════

def build_environment_anfis():
    """
    FIX 2: Widened sigma values so optimal Temperature + DO
    produce a score in the 70–90% range, not ~46%.
    MF centres tightened around the true optimal band.
    """
    return SubANFIS(
        name = "Environment",
        input_defs = [
            ("Temperature", [
                (18, 2.5, "very_cold"),   # cold tail
                (27, 4.0, "optimal"),     # wider → optimal inputs score higher
                (35, 2.5, "very_hot"),    # hot tail
            ]),
            ("Dissolved_O2", [
                (3,  1.2, "critical"),
                (7,  2.5, "optimal"),     # wider → more of the O2 range scores well
                (12, 1.5, "high"),
            ]),
        ]
    )

def build_chemistry_anfis():
    """
    FIX 3: Neutral pH centre moved from 7.2→7.0 and sigma widened
    so pH 7.0–7.8 cleanly fires the 'neutral' MF.
    Ammonia 'safe' MF widened so low NH3 values score well.
    Acidic pH will fire the 'acidic' MF → low score (correct).
    """
    return SubANFIS(
        name = "Chemistry",
        input_defs = [
            ("pH", [
                (5.5, 0.5, "acidic"),     # tight → only very low pH fires high here
                (7.0, 1.0, "neutral"),    # wider → 6.5–7.8 scores as neutral/healthy
                (9.2, 0.7, "alkaline"),
            ]),
            ("Ammonia_NH3", [
                (0.1, 0.15, "safe"),      # wider safe band
                (0.8, 0.3,  "moderate"),
                (2.5, 0.8,  "dangerous"),
            ]),
        ]
    )

def build_population_anfis():
    return SubANFIS(
        name = "Population",
        input_defs = [
            ("Fish_Count", [
                (1,  1.0, "sparse"),
                (5,  2.0, "normal"),
                (12, 3.0, "dense"),
            ]),
            ("Juvenile_Pct", [
                (5,  4.0, "low"),
                (25, 8.0, "moderate"),
                (60, 15., "high"),
            ]),
        ]
    )

def build_master_anfis():
    """
    FIX 5: Master MF centres now reflect realistic sub-score ranges.
    Sub-scores often land in 0.45–0.75; old centres (0.2/0.6/0.9)
    caused mis-labelling (e.g. 0.46 → 'poor' instead of 'weak/moderate').
    New ranges:
      <0.40  → poor
      0.40–0.65 → moderate
      >0.65  → excellent
    """
    return SubANFIS(
        name = "Master",
        input_defs = [
            ("Env_Score", [
                (0.20, 0.12, "poor"),
                (0.52, 0.15, "moderate"),   # shifted down from 0.60
                (0.80, 0.12, "excellent"),  # shifted down from 0.90
            ]),
            ("Chem_Score", [
                (0.20, 0.12, "poor"),
                (0.52, 0.15, "moderate"),
                (0.80, 0.12, "excellent"),
            ]),
            ("Pop_Score", [
                (0.20, 0.12, "stressed"),
                (0.52, 0.15, "stable"),
                (0.80, 0.12, "healthy"),
            ]),
        ]
    )

# ══════════════════════════════════════════════
# TRAIN ALL MODELS
# ══════════════════════════════════════════════
def train_all(epochs=120, n=800):
    print("=" * 55)
    print("  Hierarchical ANFIS — Training All Sub-Models")
    print("=" * 55)
    np.random.seed(42)

    env_model  = build_environment_anfis()
    chem_model = build_chemistry_anfis()
    pop_model  = build_population_anfis()
    mst_model  = build_master_anfis()

    temps     = np.random.uniform(15, 38, n)
    phs       = np.random.uniform(5.0, 10.0, n)
    o2s       = np.clip(14.6 - 0.38 * temps + np.random.normal(0, 0.5, n), 1, 14)
    nh3s      = np.clip((temps - 20) * 0.03 + np.random.normal(0, 0.05, n), 0.01, 5)
    fish_cnts = np.random.randint(0, 15, n).astype(float)
    juv_pcts  = np.random.uniform(0, 80, n)

    # Ground-truth scores (biological model) — unchanged from original
    env_gt  = np.exp(-0.5*((temps-27)/4)**2) * np.exp(-0.5*((o2s-7)/2)**2)
    chem_gt = np.exp(-0.5*((phs-7.2)/0.8)**2) * np.exp(-0.5*((nh3s-0.1)/0.3)**2)
    pop_gt  = np.exp(-0.5*((juv_pcts-5)/15)**2) * np.clip(fish_cnts/8, 0.1, 1.0)
    mst_gt  = np.clip(0.40*env_gt + 0.35*chem_gt + 0.25*pop_gt, 0.0, 1.0)  # FIX 4

    print(f"\n  Training Environment ANFIS...")
    for ep in range(epochs):
        idx = np.random.permutation(n)
        for i in idx:
            env_model.train_step([temps[i], o2s[i]], env_gt[i])
        if (ep+1) % 40 == 0:
            print(f"    Epoch {ep+1}/{epochs} ✓")

    print(f"  Training Chemistry ANFIS...")
    for ep in range(epochs):
        idx = np.random.permutation(n)
        for i in idx:
            chem_model.train_step([phs[i], nh3s[i]], chem_gt[i])
        if (ep+1) % 40 == 0:
            print(f"    Epoch {ep+1}/{epochs} ✓")

    print(f"  Training Population ANFIS...")
    for ep in range(epochs):
        idx = np.random.permutation(n)
        for i in idx:
            pop_model.train_step([fish_cnts[i], juv_pcts[i]], pop_gt[i])
        if (ep+1) % 40 == 0:
            print(f"    Epoch {ep+1}/{epochs} ✓")

    print(f"  Training Master ANFIS...")
    for ep in range(epochs):
        idx = np.random.permutation(n)
        for i in idx:
            env_s  = env_model.forward([temps[i], o2s[i]])[0]
            chem_s = chem_model.forward([phs[i], nh3s[i]])[0]
            pop_s  = pop_model.forward([fish_cnts[i], juv_pcts[i]])[0]
            mst_model.train_step([env_s, chem_s, pop_s], mst_gt[i])
        if (ep+1) % 40 == 0:
            print(f"    Epoch {ep+1}/{epochs} ✓")

    print("\n  All models trained!\n" + "=" * 55)
    return env_model, chem_model, pop_model, mst_model

# ══════════════════════════════════════════════
# SCORE LABEL HELPER  (FIX 5)
# ══════════════════════════════════════════════
def score_label(pct: float) -> str:
    """Map a 0–100 score to a human-readable quality label."""
    if pct >= 80: return "excellent"
    if pct >= 60: return "moderate"
    if pct >= 40: return "weak"
    return "poor"

# ══════════════════════════════════════════════
# EXPLAINABILITY ENGINE
# ══════════════════════════════════════════════
def generate_explanation(env_exp, chem_exp, pop_exp, mst_exp,
                         env_s, chem_s, pop_s, final_score,
                         temp, ph, o2, nh3, fish_count, juv_pct):
    """
    Generate human-readable explanation of why the score is what it is.

    FIX 1: Grade thresholds now correctly reflect pond health:
      80–100 → A (Healthy)
      60–80  → B (Stable)
      40–60  → C (Warning)
      20–40  → D (Critical)
      0–20   → F (Severe)
    """
    scores   = {"Environment": env_s, "Chemistry": chem_s, "Population": pop_s}
    dominant = min(scores, key=scores.get)

    env_contrib  = round(env_s  * 0.40 * 100, 1)
    chem_contrib = round(chem_s * 0.35 * 100, 1)
    pop_contrib  = round(pop_s  * 0.25 * 100, 1)

    pct = final_score * 100

    # FIX 1 — corrected grade thresholds
    if   pct >= 80: grade = "A"
    elif pct >= 60: grade = "B"
    elif pct >= 40: grade = "C"
    elif pct >= 20: grade = "D"
    else:           grade = "F"

    # FIX 1 — health status label consistent with score
    if   pct >= 80: health_status = "Healthy"
    elif pct >= 60: health_status = "Stable"
    elif pct >= 40: health_status = "Warning"
    else:           health_status = "Critical"

    # Sensor-level natural language reasons
    reasons = []
    if temp > 30:
        reasons.append(f"Temperature critically high ({temp:.1f}°C) — Aeromonas risk")
    elif temp < 20:
        reasons.append(f"Temperature too low ({temp:.1f}°C) — metabolism suppressed")
    if ph < 6.5:
        reasons.append(f"pH dangerously acidic ({ph:.1f}) — gill damage likely")
    elif ph > 8.5:
        reasons.append(f"pH too alkaline ({ph:.1f}) — ammonia toxicity increases")
    if o2 < 5:
        reasons.append(f"Dissolved O₂ critically low ({o2:.1f} mg/L) — suffocation risk")
    if nh3 > 0.5:
        reasons.append(f"Ammonia elevated ({nh3:.2f} ppm) — immune suppression")
    if juv_pct > 30:
        reasons.append(f"High juvenile catch ({juv_pct:.0f}%) — population stressed")
    if fish_count == 0:
        reasons.append("No fish detected in frame")

    # FIX 6 — explicit population risk message when pop score is low
    pop_pct = pop_s * 100
    if pop_pct < 40:
        reasons.append(
            f"Population score critically low ({pop_pct:.0f}%) — "
            "imbalance detected, check stocking density or breeding"
        )
    elif pop_pct < 60:
        reasons.append(
            f"Population score weak ({pop_pct:.0f}%) — "
            "monitor stocking density and juvenile ratio closely"
        )

    if not reasons:
        reasons.append("All parameters within optimal range — pond is healthy")

    return {
        "final_score":     round(pct, 1),
        "grade":           grade,
        "health_status":   health_status,           # FIX 1 — added explicit field
        "dominant_factor": dominant,
        "env_score":       round(env_s  * 100, 1),
        "chem_score":      round(chem_s * 100, 1),
        "pop_score":       round(pop_s  * 100, 1),
        "env_label":       score_label(env_s  * 100),  # FIX 5
        "chem_label":      score_label(chem_s * 100),  # FIX 5
        "pop_label":       score_label(pop_s  * 100),  # FIX 5
        "env_contrib":     env_contrib,
        "chem_contrib":    chem_contrib,
        "pop_contrib":     pop_contrib,
        "env_rules":       env_exp,
        "chem_rules":      chem_exp,
        "pop_rules":       pop_exp,
        "master_rules":    mst_exp,
        "reasons":         reasons,
        "recommendation":  _recommend(dominant, temp, ph, o2, nh3, juv_pct, pop_s),
    }

def _recommend(dominant, temp, ph, o2, nh3, juv_pct, pop_s):
    """
    FIX 6: Population branch now explicitly covers imbalance / density issues.
    """
    if dominant == "Environment":
        if temp > 30:
            return "Activate cooling system or increase water circulation immediately"
        if temp < 20:
            return "Apply pond heater or reduce water depth to allow solar warming"
        if o2 < 5:
            return "Activate aerators immediately — oxygen critical"
        return "Monitor environmental parameters — minor instability detected"

    if dominant == "Chemistry":
        if ph < 6.5:
            return "Add agricultural lime (CaCO₃) to raise pH — target 7.0–7.5"
        if ph > 8.5:
            return "Add CO₂ or organic matter to lower pH"
        if nh3 > 0.5:
            return "Perform 30% water exchange and reduce feeding immediately"
        return "Check water chemistry — minor imbalance detected"

    if dominant == "Population":
        # FIX 6 — richer population recommendations
        if pop_s * 100 < 40:
            return (
                "Population imbalance detected — check stocking density or breeding. "
                "Reduce stock if overcrowded; increase if underutilised."
            )
        if pop_s * 100 < 60:
            return (
                "Population under stress — monitor stocking density and juvenile ratio. "
                "Avoid harvesting until population stabilises."
            )
        if juv_pct > 30:
            return "Release juvenile catch — do not harvest until size threshold met"
        return "Population shows instability — review feeding and crowding conditions"

    return "Maintain current conditions — all parameters stable"

# ══════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════
env_model, chem_model, pop_model, mst_model = train_all()

sim_temp = 27.0; sim_ph = 7.2; sim_o2 = 7.5

print("\n[XAI Engine] Running — sending to dashboard...\n")

while True:
    try:
        temp = ph = o2 = None
        try:
            r    = requests.get("http://127.0.0.1:5050/sensor_data", timeout=0.3)
            d    = r.json()
            temp = d.get("temp"); ph = d.get("ph"); o2 = d.get("oxygen")
        except:
            pass

        # Ensure we have valid sensor data, otherwise use simulation
        if temp is None or ph is None or o2 is None:
            sim_temp += np.random.normal(0, 0.1)
            sim_ph   += np.random.normal(0, 0.02)
            sim_o2   += np.random.normal(0, 0.05)
            sim_temp  = float(np.clip(sim_temp, 15, 38))
            sim_ph    = float(np.clip(sim_ph, 5.0, 10.0))
            sim_o2    = float(np.clip(sim_o2, 1.0, 14.0))
            temp = sim_temp; ph = sim_ph; o2 = sim_o2

        fish_count = 0; juv_pct = 0
        try:
            r2         = requests.get("http://127.0.0.1:5050/data", timeout=0.3)
            d2         = r2.json()
            fish_count = d2.get("total_count", 0) or 0
            juv_pct    = d2.get("juvenile_percentage", 0) or 0
        except:
            pass

        nh3 = float(np.clip(
            (temp - 20) * 0.025 + (ph - 7) * 0.05 + np.random.normal(0, 0.02),
            0.01, 5.0
        ))

        # ── Run Hierarchical ANFIS ──
        env_exp,  env_s  = env_model.explain([temp, o2])
        chem_exp, chem_s = chem_model.explain([ph, nh3])
        pop_exp,  pop_s  = pop_model.explain([float(fish_count), float(juv_pct)])
        mst_exp,  final  = mst_model.explain([env_s, chem_s, pop_s])

        # FIX 4: Enforce weighted sum as fallback cross-check
        weighted_final = 0.40 * env_s + 0.35 * chem_s + 0.25 * pop_s
        # Use ANFIS output if it is within 15% of the weighted expectation;
        # otherwise fall back to the weighted sum (catches fuzzy aggregation drift)
        if abs(final - weighted_final) > 0.15:
            final = float(np.clip(weighted_final, 0.0, 1.0))

        xai = generate_explanation(
            env_exp, chem_exp, pop_exp, mst_exp,
            env_s, chem_s, pop_s, final,
            temp, ph, o2, nh3, fish_count, juv_pct
        )

        payload = {
            "temp":              round(temp, 2),
            "ph":                round(ph, 3),
            "oxygen":            round(o2, 2),
            "predicted_weight":  round(env_s * 400 + 30, 1),
            "condition_score":   xai["final_score"],
            "growth_status":     xai["grade"],
            "health_status":     xai["health_status"],   # FIX 1
            "arduino_connected": False,
            "xai":               xai,
        }
        requests.post("http://127.0.0.1:5050/sensor", json=payload, timeout=0.5)

        # ── Console output ──
        print(f"\n{'='*55}")
        print(f"  HIERARCHICAL ANFIS + XAI REPORT")
        print(f"{'='*55}")
        print(f"  Temp:{temp:.1f}°C  pH:{ph:.2f}  O2:{o2:.1f}  NH3:{nh3:.2f}")
        print(f"  Fish:{fish_count}  Juv%:{juv_pct:.0f}%")
        print(f"{'─'*55}")
        print(f"  Env  Score : {xai['env_score']:.1f}%  [{xai['env_label']}]"
              f"  → {env_exp[0]['rule']}")
        print(f"  Chem Score : {xai['chem_score']:.1f}%  [{xai['chem_label']}]"
              f"  → {chem_exp[0]['rule']}")
        print(f"  Pop  Score : {xai['pop_score']:.1f}%  [{xai['pop_label']}]"
              f"  → {pop_exp[0]['rule']}")
        print(f"{'─'*55}")
        print(f"  FINAL SCORE : {xai['final_score']:.1f}%  "
              f"GRADE: {xai['grade']}  STATUS: {xai['health_status']}")
        print(f"  DOMINANT    : {xai['dominant_factor']}")
        for r in xai['reasons']:
            print(f"  REASON      : {r}")
        print(f"  ACTION      : {xai['recommendation']}")
        print(f"{'='*55}")

    except Exception as e:
        print(f"Error: {e}")

    time.sleep(1)