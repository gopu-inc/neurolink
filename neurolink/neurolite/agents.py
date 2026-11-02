"""Sous-agents NeuroLink : un orchestrateur découpe une tâche, des sous-agents la traitent en
parallèle, puis un agent de synthèse assemble la réponse.

Chaque agent = (modèle NeuroLink, prompt système, options). Plusieurs agents peuvent utiliser
des modèles différents, par exemple un modèle entraîné sur du code et un modèle français.
Les appels à un même modèle sont sérialisés par un verrou ; les modèles différents tournent
en parallèle.
"""
import concurrent.futures as cf
import re
import threading

from . import load_model
from .server import build_prompt, merge_options, text_events

DEFAULT_MODEL = "neurolink-think-2.1-pro"
ROLE_PROMPTS = {
    "planner": "Tu découpes une tâche en sous-tâches courtes et indépendantes.",
    "coder": "Tu écris du code Python clair et correct, avec des exemples d'utilisation.",
    "writer": "Tu réponds en français, de façon claire et structurée.",
    "reviewer": "Tu relis une réponse, signales les erreurs et proposes une version corrigée.",
}

_models, _locks = {}, {}
_guard = threading.Lock()


def _model(name):
    with _guard:
        if name not in _models:
            _models[name] = load_model(name)
            _locks[name] = threading.Lock()
        return _models[name], _locks[name]


class Agent:
    def __init__(self, name, model=DEFAULT_MODEL, system="", options=None):
        self.name, self.model, self.system = name, model, system
        self.options = options or {}

    def run(self, prompt, max_tokens=None):
        m, lock = _model(self.model)
        man = m.manifest or {}
        system = self.system or man.get("system", "")
        opts = merge_options(man.get("parameters"), self.options,
                             {"num_predict": max_tokens} if max_tokens else None)
        text = build_prompt([{"role": "user", "content": prompt}], system, True)
        content, thinking = [], []
        with lock:
            for kind, piece in text_events(m, text, opts):
                if kind == "done":
                    break
                (content if kind == "content" else thinking).append(piece)
        return {"agent": self.name, "model": self.model,
                "answer": "".join(content).strip(), "thinking": "".join(thinking).strip()}


class Orchestrator:
    def __init__(self, model=DEFAULT_MODEL, roles=None, max_subtasks=4, max_tokens=None):
        self.model = model
        self.roles = roles or []
        self.max_subtasks = max_subtasks
        self.max_tokens = max_tokens

    def _role(self, name, default_system):
        for r in self.roles:
            if r.get("name") == name:
                return Agent(name, r.get("model", self.model), r.get("system", default_system))
        return Agent(name, self.model, default_system)

    def plan(self, task):
        planner = self._role("planner", ROLE_PROMPTS["planner"])
        out = planner.run(
            "Découpe cette tâche en au plus %d sous-tâches, une par ligne, commençant par '- ' :\n%s"
            % (self.max_subtasks, task), self.max_tokens)
        items = [re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
                 for line in out["answer"].splitlines()]
        items = [i for i in items if len(i) > 3][: self.max_subtasks]
        return items or [task]

    def run(self, task):
        subtasks = self.plan(task)
        workers = [self._role(name, ROLE_PROMPTS[name])
                   for name in ("coder", "writer") if name in ROLE_PROMPTS]
        jobs = []
        with cf.ThreadPoolExecutor(max_workers=max(1, len(subtasks))) as ex:
            for i, sub in enumerate(subtasks):
                agent = workers[i % len(workers)]
                jobs.append(ex.submit(agent.run, sub, self.max_tokens))
            results = [j.result() for j in jobs]
        synth = self._role("reviewer", ROLE_PROMPTS["reviewer"])
        joined = "\n\n".join(f"Sous-tâche : {s}\nRésultat : {r['answer']}"
                             for s, r in zip(subtasks, results))
        final = synth.run(f"Tâche initiale : {task}\n\n{joined}\n\nRédige la réponse finale.",
                          self.max_tokens)
        return {"task": task, "subtasks": subtasks, "results": results,
                "answer": final["answer"]}
