"""
coaching_llm.py
---------------
Lightweight local LLM coaching report generator.
Qwen2.5-3B-Instruct (4-bit) — ~2 GB, runs on Colab T4.

Output: clean coaching prose only — no analytics preamble, no stats dump.
"""
from typing import List, Dict, Optional
from collections import Counter

import torch


MODEL_NAME = 'Qwen/Qwen2.5-3B-Instruct'


def _build_analytics_payload(
    shot_events, zone_stats, rallies, bounces,
    handedness, seat_labels, fps, n_frames,
):
    duration_sec = round(n_frames / fps, 1)
    per_player = []
    for p in range(len(seat_labels)):
        shots_for_p = [e for e in shot_events if e['player_idx'] == p]
        shot_counter = Counter(e['shot_type'] for e in shots_for_p)
        zs = next((z for z in zone_stats if z['player_idx'] == p), None)
        per_player.append({
            'seat'        : seat_labels[p],
            'handedness'  : handedness[p] or 'unknown',
            'total_shots' : len(shots_for_p),
            'forehands'   : shot_counter.get('forehand', 0),
            'backhands'   : shot_counter.get('backhand', 0),
            'smashes'     : shot_counter.get('smash', 0),
            'viboras'     : shot_counter.get('vibora', 0),
            'pct_at_net'      : zs['pct_in_net']      if zs else 0,
            'pct_at_baseline' : zs['pct_in_baseline'] if zs else 0,
            'time_at_net_sec' : zs['time_in_net_sec'] if zs else 0,
            'zone_transitions': zs['n_transitions']   if zs else 0,
        })
    return {
        'session_duration_sec': duration_sec,
        'num_rallies'         : len(rallies),
        'num_bounces'         : len(bounces),
        'players'             : per_player,
    }


def _format_payload_for_prompt(payload):
    lines = []
    lines.append(f"Session length: {payload['session_duration_sec']}s")
    lines.append(f"Rallies: {payload['num_rallies']}")
    lines.append(f"Ball bounces detected: {payload['num_bounces']}")
    lines.append("")
    lines.append("Per-player statistics:")
    for pl in payload['players']:
        lines.append(
            f"  {pl['seat']} ({pl['handedness']}): "
            f"{pl['total_shots']} shots "
            f"({pl['forehands']} FH, {pl['backhands']} BH, "
            f"{pl['smashes']} smash, {pl['viboras']} vibora) | "
            f"net {pl['pct_at_net']}% / baseline {pl['pct_at_baseline']}% | "
            f"{pl['zone_transitions']} zone transitions"
        )
    return "\n".join(lines)


def _build_prompt(payload, focus_seats=('P3', 'P4')):
    analytics_text = _format_payload_for_prompt(payload)
    focus_str = ', '.join(focus_seats)
    system_msg = (
        "You are an experienced padel coach giving concise, practical advice. "
        "You analyze data from a single match and write feedback that a club-level "
        "player can act on. Keep advice specific and actionable — avoid generic "
        "phrases. Reference actual numbers where useful, but do NOT just list the "
        "stats back. Write it as analysis, not a stats dump."
    )
    user_msg = f"""Match analytics (for your reference only — do NOT repeat these stats verbatim in your output):
{analytics_text}

Write a coaching analysis with these rules:
- Start IMMEDIATELY with the analysis. No greeting, no preamble like "Based on the data..." or "Here is the analysis...".
- Do NOT list rally counts, bounce counts, or per-player stat tables.
- Do NOT use headers like "Session Overview" or "Player Stats".
- Write it as flowing coaching paragraphs.
- Focus the coaching on {focus_str} (the subscribing players).
- 3-5 short paragraphs total.
- End with one team-level tactical observation.
- Be direct and specific. No filler. No disclaimers.

Output only the coaching analysis. Begin now:"""
    return system_msg, user_msg


class CoachingLLM:
    def __init__(self, model_name=MODEL_NAME, device='cuda'):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None

    def _ensure_loaded(self):
        if self.model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        print(f'  loading LLM: {self.model_name} (4-bit) ...')
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb,
            device_map='auto',
            torch_dtype=torch.float16,
        )
        self.model.eval()
        print(f'  LLM loaded.')

    def generate(self, system_msg, user_msg, max_new_tokens=600):
        self._ensure_loaded()
        messages = [
            {'role': 'system', 'content': system_msg},
            {'role': 'user',   'content': user_msg},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer([text], return_tensors='pt').to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                repetition_penalty=1.1,
            )
        generated = output_ids[0][inputs['input_ids'].shape[1]:]
        reply = self.tokenizer.decode(generated, skip_special_tokens=True)
        return reply.strip()


def generate_coaching_report(
    shot_events, zone_stats, rallies, bounces, handedness,
    seat_labels, fps, n_frames, focus_seats=('P3', 'P4'),
    out_path=None, llm=None,
):
    payload = _build_analytics_payload(
        shot_events, zone_stats, rallies, bounces,
        handedness, seat_labels, fps, n_frames,
    )
    system_msg, user_msg = _build_prompt(payload, focus_seats)
    if llm is None:
        llm = CoachingLLM()
    report = llm.generate(system_msg, user_msg)

    # Clean output: just the report, no preamble, no stats
    if out_path:
        with open(out_path, 'w') as f:
            f.write(report.strip() + "\n")
        print(f'  coaching report -> {out_path}')

    return report