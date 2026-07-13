import json
import os
import glob
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
import torch

try:
    from baseline import (
        MyModel,
        mcts_agent,
        sample_deck,
        card_table,
        to_observation_class,
        battle_start,
        battle_finish,
        battle_select,
        visualize_data,
    )
except ImportError:
    print("Error: Could not import 'baseline.py'. Ensure it is in the same directory.")
    sys.exit(1)


def load_latest_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyModel(128, 2, 256, 1, 1).to(device)
    models = sorted(glob.glob("out/model*.pth"))
    if models:
        model.load_state_dict(torch.load(models[-1], map_location=device))
    model.eval()
    return model, device


GAME_SESSION = {
    "model": None,
    "device": None,
    "current_obs": None,
    "human_index": 0,
    "is_active": False,
    "logs": [],
    "obs_log": [],
    "action_log": [],
    "version": 0,
}

OPTION_TYPES = {
    0: "END TURN",
    1: "YES",
    2: "NO",
    3: "SPECIAL CONDITION",
    4: "NUMBER",
    5: "ATTACK",
    6: "PLAY CARD",
    7: "ATTACH ENERGY",
    8: "EVOLVE POKEMON",
    9: "USE ABILITY",
    10: "DISCARD CARD",
    11: "RETREAT",
    12: "SELECT CARD",
    13: "SELECT TOOL CARD",
    14: "SELECT ENERGY CARD",
    15: "SKILL",
}


def build_option_description(opt):
    opt_type = opt.get("type", 0)
    type_name = OPTION_TYPES.get(opt_type, f"TYPE {opt_type}")
    desc = f"[{type_name}]"
    if "cardId" in opt and opt["cardId"] in card_table:
        desc += f" - {card_table[opt['cardId']].name}"
    if "index" in opt:
        desc += f" (Index {opt['index']})"
    return desc


def sync_competition_visualizer(last_obs, last_action):
    GAME_SESSION["obs_log"].append(last_obs)
    GAME_SESSION["action_log"].append(last_action)
    try:
        vis = json.loads(visualize_data())
        for i in range(min(len(vis), len(GAME_SESSION["obs_log"]))):
            vis[i]["obs"] = GAME_SESSION["obs_log"][i]
            act = GAME_SESSION["action_log"][i]
            vis[i]["action"] = [act, act]
        with open("vis.json", "w") as file:
            json.dump(vis, file)
        GAME_SESSION["version"] += 1
    except Exception:
        pass


def process_ai_turns():
    global GAME_SESSION
    while GAME_SESSION["is_active"]:
        obs = GAME_SESSION["current_obs"]
        if obs["current"]["result"] >= 0:
            GAME_SESSION["logs"].append(
                f"Match over. Result code: {obs['current']['result']}"
            )
            GAME_SESSION["is_active"] = False
            GAME_SESSION["version"] += 1
            break

        if obs["current"]["yourIndex"] == GAME_SESSION["human_index"]:
            break
        else:
            selected, _ = mcts_agent(obs, sample_deck, GAME_SESSION["model"])
            GAME_SESSION["logs"].append(f"AI selected options: {selected}")
            next_obs = battle_select(selected)
            sync_competition_visualizer(obs, selected)
            GAME_SESSION["current_obs"] = next_obs


class UnifiedWorkspaceHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(HTML_INTERFACE.encode("utf-8"))
        elif self.path == "/visualizer":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(HTML_VISUALIZER_BRIDGE.encode("utf-8"))
        elif self.path == "/api/state":
            self.send_json_state()
        elif self.path == "/api/vis_sync":
            self.send_vis_sync_data()
        else:
            self.send_error(404)

    def do_POST(self):
        global GAME_SESSION
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length).decode("utf-8")

        if self.path == "/api/start":
            battle_finish()
            obs, _ = battle_start(sample_deck, sample_deck)

            GAME_SESSION["current_obs"] = obs
            GAME_SESSION["is_active"] = True
            GAME_SESSION["logs"] = ["New match started."]
            GAME_SESSION["obs_log"] = []
            GAME_SESSION["action_log"] = []
            GAME_SESSION["version"] += 1

            sync_competition_visualizer(obs, None)
            process_ai_turns()
            self.send_json_state()

        elif self.path == "/api/action":
            if not GAME_SESSION["is_active"]:
                self.send_error(400)
                return
            try:
                data = json.loads(post_data)
                selected = [int(x) for x in data.get("indices", [])]
                obs = GAME_SESSION["current_obs"]

                GAME_SESSION["logs"].append(f"Human selected options: {selected}")
                next_obs = battle_select(selected)

                sync_competition_visualizer(obs, selected)
                GAME_SESSION["current_obs"] = next_obs

                process_ai_turns()
                self.send_json_state()
            except Exception as e:
                self.send_error(500, str(e))

    def send_json_state(self):
        global GAME_SESSION
        if GAME_SESSION["current_obs"] is None:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "no_game"}).encode("utf-8"))
            return

        obs = GAME_SESSION["current_obs"]
        options = []
        for idx, opt in enumerate(obs.get("select", {}).get("option", [])):
            card_id = opt.get("cardId")
            card_name = (
                card_table[card_id].name
                if (card_id and card_id in card_table)
                else None
            )
            options.append(
                {
                    "index": idx,
                    "type": OPTION_TYPES.get(opt.get("type", 0), "UNKNOWN"),
                    "cardName": card_name,
                    "details": {k: v for k, v in opt.items() if k != "type"},
                }
            )

        payload = {
            "status": "active" if GAME_SESSION["is_active"] else "finished",
            "is_human_turn": obs["current"]["yourIndex"] == GAME_SESSION["human_index"],
            "max_count": obs["select"]["maxCount"],
            "options": options,
            "logs": GAME_SESSION["logs"],
        }
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def send_vis_sync_data(self):
        global GAME_SESSION
        payload = ""
        if os.path.exists("vis.json"):
            try:
                with open("vis.json", "r", encoding="utf-8") as f:
                    obj = json.load(f)
                payload = (
                    json.dumps(obj["steps"][0][0]["visualize"])
                    if "steps" in obj
                    else json.dumps(obj)
                )
            except Exception:
                pass

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(
            json.dumps({"version": GAME_SESSION["version"], "payload": payload}).encode(
                "utf-8"
            )
        )


HTML_INTERFACE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>PTCG Action Workspace</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="p-6 h-screen flex flex-col bg-slate-950 text-slate-100">
    <div class="flex justify-between items-center bg-slate-900 border border-slate-800 p-4 rounded-xl mb-6 shadow-xl">
        <div class="flex items-center space-x-4">
            <h1 class="text-lg font-bold tracking-tight text-white flex items-center gap-2">
                <span class="inline-block w-2.5 h-2.5 rounded-full bg-indigo-500 animate-pulse"></span>
                Action Construction Engine
            </h1>
            <div id="turn-badge" class="px-3 py-1 rounded text-xs font-bold uppercase">No Game Session</div>
        </div>
        <div class="flex items-center gap-4">
            <a href="/visualizer" target="_blank" class="text-sm text-indigo-400 hover:text-indigo-300 font-semibold underline">Open Visualizer Sync Page</a>
            <button onclick="startGame()" class="bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-sm py-2 px-5 rounded-lg transition">
                Initialize / Reset Battle
            </button>
        </div>
    </div>
    <div class="flex-1 grid grid-cols-1 lg:grid-cols-3 gap-6 overflow-hidden">
        <div class="lg:col-span-2 flex flex-col bg-slate-900 border border-slate-800 rounded-xl p-5 overflow-hidden">
            <h2 class="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-3">Available Game Actions</h2>
            <div id="category-tabs" class="flex gap-2 border-b border-slate-800 pb-3 mb-4 overflow-x-auto"></div>
            <div id="cards-container" class="flex-1 overflow-y-auto grid grid-cols-1 md:grid-cols-2 gap-3 pr-2">
                <div class="text-slate-500 italic text-sm col-span-2 p-4 text-center">Start or reset match to load actions.</div>
            </div>
            <div class="mt-4 pt-4 border-t border-slate-800 flex justify-between items-center">
                <div class="text-xs text-slate-400">Selection requirement: <span id="selection-requirement" class="font-bold text-indigo-400">0 / 0</span> choices</div>
                <button id="submit-btn" onclick="submitAction()" disabled class="bg-slate-800 text-slate-500 font-bold text-sm py-2.5 px-6 rounded-lg cursor-not-allowed">Commit Selections</button>
            </div>
        </div>
        <div class="bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col overflow-hidden">
            <h2 class="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-2">Match Progression Log</h2>
            <pre id="log-box" class="flex-1 overflow-y-auto bg-slate-950 p-3 rounded-lg font-mono text-xs text-slate-300 space-y-1.5 border border-slate-850 select-text"></pre>
        </div>
    </div>
    <script>
        let currentOptions = []; let maxCount = 1; let selectedIndices = []; let activeCategory = null;
        async function startGame() { const r = await fetch('/api/start', { method: 'POST' }); render(await r.json()); }
        function render(data) {
            if (!data || data.status === 'no_game') return;
            maxCount = data.max_count; currentOptions = data.options; selectedIndices = [];
            const badge = document.getElementById('turn-badge');
            if (data.status === 'finished') {
                badge.innerText = "Match Concluded"; badge.className = "px-3 py-1 rounded text-xs font-bold bg-rose-500/20 text-rose-400 border border-rose-500/30";
            } else if (data.is_human_turn) {
                badge.innerText = `Your Move (Pick ${maxCount})`; badge.className = "px-3 py-1 rounded text-xs font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30";
            } else {
                badge.innerText = "AI Executing Turn"; badge.className = "px-3 py-1 rounded text-xs font-bold bg-amber-500/20 text-amber-400 border border-amber-500/30";
            }
            const logBox = document.getElementById('log-box'); logBox.innerText = data.logs.join('\\n'); logBox.scrollTop = logBox.scrollHeight;
            if (!data.is_human_turn || data.status === 'finished') {
                document.getElementById('category-tabs').innerHTML = '';
                document.getElementById('cards-container').innerHTML = `<div class="text-slate-500 italic text-sm col-span-2 p-4 text-center">${data.status === 'finished' ? 'Game Over.' : 'Waiting for AI...'}</div>`;
                validateSelection(); return;
            }
            const categories = {};
            currentOptions.forEach(opt => { if (!categories[opt.type]) categories[opt.type] = []; categories[opt.type].push(opt); });
            const keys = Object.keys(categories);
            if (keys.length > 0 && (!activeCategory || !categories[activeCategory])) activeCategory = keys[0];
            const tabsContainer = document.getElementById('category-tabs'); tabsContainer.innerHTML = '';
            keys.forEach(cat => {
                const tabBtn = document.createElement('button');
                tabBtn.className = `px-3 py-1.5 text-xs font-semibold rounded-md transition ${cat === activeCategory ? 'bg-indigo-600 text-white' : 'bg-slate-800 text-slate-400 hover:bg-slate-700'}`;
                tabBtn.innerText = `${cat} (${categories[cat].length})`;
                tabBtn.onclick = () => {
                    activeCategory = cat; renderActionCards(categories[cat]);
                    Array.from(tabsContainer.children).forEach(b => b.classList.replace('bg-indigo-600', 'bg-slate-800'));
                    Array.from(tabsContainer.children).forEach(b => b.classList.replace('text-white', 'text-slate-400'));
                    tabBtn.classList.replace('bg-slate-800', 'bg-indigo-600'); tabBtn.classList.replace('text-slate-400', 'text-white');
                };
                tabsContainer.appendChild(tabBtn);
            });
            if (activeCategory && categories[activeCategory]) renderActionCards(categories[activeCategory]);
            validateSelection();
        }
        function renderActionCards(optionsList) {
            const container = document.getElementById('cards-container'); container.innerHTML = '';
            optionsList.forEach(opt => {
                const card = document.createElement('div'); const isSelected = selectedIndices.includes(opt.index);
                card.id = `card-opt-${opt.index}`;
                card.className = `p-4 rounded-xl border cursor-pointer flex flex-col justify-between ${isSelected ? 'bg-indigo-950/40 border-indigo-500 shadow-md' : 'bg-slate-950/60 border-slate-800 hover:border-slate-700'}`;
                let detailsHtml = '';
                Object.entries(opt.details).forEach(([key, val]) => {
                    if (val !== null && val !== undefined && val !== "") {
                        detailsHtml += `<span class="bg-slate-900 px-2 py-0.5 rounded text-[10px] text-slate-400 font-mono border border-slate-800">${key}: <strong class="text-slate-200">${JSON.stringify(val)}</strong></span>`;
                    }
                });
                card.innerHTML = `<div><div class="flex justify-between items-start mb-2"><span class="text-xs font-bold text-slate-400">INDEX #${opt.index}</span><span class="text-[10px] bg-slate-800 px-2 py-0.5 rounded font-bold text-indigo-400 uppercase">${opt.type}</span></div><div class="text-sm font-semibold text-white mb-2">${opt.cardName ? opt.cardName : 'Action Definition'}</div><div class="flex flex-wrap gap-1.5">${detailsHtml}</div></div>`;
                card.onclick = () => toggleSelectIndex(opt.index); container.appendChild(card);
            });
        }
        function toggleSelectIndex(index) {
            const lookupIdx = selectedIndices.indexOf(index);
            if (lookupIdx > -1) { selectedIndices.splice(lookupIdx, 1); } else { if (maxCount === 1) { selectedIndices = [index]; } else if (selectedIndices.length < maxCount) { selectedIndices.push(index); } }
            currentOptions.forEach(opt => {
                const el = document.getElementById(`card-opt-${opt.index}`);
                if (el) el.className = selectedIndices.includes(opt.index) ? "p-4 rounded-xl border cursor-pointer flex flex-col justify-between bg-indigo-950/40 border-indigo-500 shadow-md" : "p-4 rounded-xl border cursor-pointer flex flex-col justify-between bg-slate-950/60 border-slate-800 hover:border-slate-700";
            });
            validateSelection();
        }
        function validateSelection() {
            document.getElementById('selection-requirement').innerText = `${selectedIndices.length} / ${maxCount}`;
            const btn = document.getElementById('submit-btn');
            if (selectedIndices.length === maxCount && maxCount > 0) {
                btn.disabled = false; btn.className = "bg-indigo-600 hover:bg-indigo-500 text-white font-bold text-sm py-2.5 px-6 rounded-lg shadow-lg cursor-pointer transform transition active:scale-95";
            } else {
                btn.disabled = true; btn.className = "bg-slate-800 text-slate-500 font-bold text-sm py-2.5 px-6 rounded-lg cursor-not-allowed";
            }
        }
        async function submitAction() {
            if (selectedIndices.length !== maxCount) return;
            const r = await fetch('/api/action', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ indices: selectedIndices }) });
            render(await r.json());
        }
        window.onload = async () => { const r = await fetch('/api/state'); render(await r.json()); };
    </script>
</body>
</html>
"""

HTML_VISUALIZER_BRIDGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Visualizer Auto-Sync Controller</title>
    <style>body, html { margin: 0; padding: 0; height: 100%; overflow: hidden; background: #0f172a; }</style>
</head>
<body>
    <iframe name="ptcg_view" style="width: 100%; height: 100%; border: none;"></iframe>
    <form id="visForm" method="POST" action="https://ptcgvis.heroz.jp/Visualizer/Replay/0" target="ptcg_view" style="display:none;">
        <input type="hidden" name="json" id="jsonInput">
    </form>
    <script>
        let currentVersion = -1;
        async function checkUpdate() {
            try {
                const r = await fetch('/api/vis_sync');
                const data = await r.json();
                if (data.version !== currentVersion && data.payload) {
                    currentVersion = data.version;
                    document.getElementById('jsonInput').value = data.payload;
                    document.getElementById('visForm').submit();
                }
            } catch (e) {}
        }
        checkUpdate();
        setInterval(checkUpdate, 1000);
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    model, device = load_latest_model()
    GAME_SESSION["model"] = model
    GAME_SESSION["device"] = device

    print("Unified Arena Engine running at http://localhost:8000")
    HTTPServer(("", 8000), UnifiedWorkspaceHandler).serve_forever()
