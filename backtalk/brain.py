# backtalk: local LLM brain adapter with recursive tree inspection & guaranteed speech delivery
# Connects directly to local llama-server (OpenAI-compatible) with low-latency streaming and tool calling.
"""The warm local brain — connects to local llama-server, executing multi-step
filesystem and web tools with recursive tree awareness, guaranteeing spoken output to Kokoro TTS.
"""
import asyncio
import json
import os
import re
import subprocess
import httpx
from pathlib import Path
from datetime import datetime
from bs4 import BeautifulSoup
from ddgs import DDGS

from backtalk.config import CFG, DISCIPLINE
from backtalk.vlog import log

_SENTENCE_END = re.compile(r"(?<=[.!?])\s")
_SPECIAL_TOKENS = re.compile(r"<\|im_end\|>|<\|im_start\|>|<\|endoftext\|>|<\|[^|]+?\|>")
_THINK_BLOCK = re.compile(r"<think>[\s\S]*?</think>|\[THINK\][\s\S]*?\[/THINK\]")
_TOOL_TAGS = re.compile(r"<tool_call>[\s\S]*?</tool_call>|<function=[^>]+>[\s\S]*?</function>|<parameter=[^>]+>[\s\S]*?</parameter>")
SIGNALS_DIR = Path(CFG.get("signals_dir", "/tmp/signals"))
SESSION_FILE = SIGNALS_DIR / ".backtalk_session"

# Known Project Aliases for quick voice switching
PROJECT_ALIASES = {
    "robots": "/Workspaces/AnZym_Robot_System",
    "fleet": "/Workspaces/AnZym_Robot_System",
    "robot fleet": "/Workspaces/AnZym_Robot_System",
    "robot system": "/Workspaces/AnZym_Robot_System",
    "anzym robot system": "/Workspaces/AnZym_Robot_System",
    "gcs": "/Workspaces/AnZym_Robot_System/anzym_gcs_ws",
    "ground control": "/Workspaces/AnZym_Robot_System/anzym_gcs_ws",
    "green": "/Workspaces/AnZym_Robot_System/anzym_green",
    "anzym green": "/Workspaces/AnZym_Robot_System/anzym_green",
    "rosorin": "/Workspaces/AnZym_Robot_System/anzym_rosorin",
    "orin": "/Workspaces/AnZym_Robot_System/anzym_rosorin",
    "zumo": "/Workspaces/AnZym_Robot_System/anzym_zumo",
    "anzym zumo": "/Workspaces/AnZym_Robot_System/anzym_zumo",
    "ham radio": "/anzym/HamRadio",
    "hamradio": "/anzym/HamRadio",
    "radio": "/anzym/HamRadio",
    "aethersdr": "/Workspaces/AetherSDR",
    "aether sdr": "/Workspaces/AetherSDR",
    "aether": "/Workspaces/AetherSDR",
    "code": "/anzym/CODE",
    "hydroponics": "/anzym/CODE/AnZym_Hydroponic-master",
    "solidworks": "/anzym/Solidworks",
    "cad": "/anzym/Solidworks",
    "notes": "/anzym/my-agent/vault",
    "vault": "/anzym/my-agent/vault",
    "memory": "/anzym/my-agent/vault",
}


def _set_signal_state(state: str):
    try:
        SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
        (SIGNALS_DIR / ".voice_state").write_text(state)
    except Exception:
        pass


_CODE_BLOCK = re.compile(r"```[\s\S]*?```", re.DOTALL)
_INLINE_SCRIPT = re.compile(r"<<\s*['\"]?[A-Za-z0-9_]+['\"]?[\s\S]*?[A-Za-z0-9_]+", re.DOTALL)
_RAW_CODE_LINE = re.compile(r"^(?:import |from |def |class |with open|cat >|\s*return |\s*#|\s*if __name__).*$", re.MULTILINE)


def _clean_text(text: str) -> str:
    """Strip special tokens, think blocks, code blocks, and tool tags for speech output."""
    if "<tool_call" in text or "</tool_call>" in text:
        text = re.sub(r"<tool_call>[\s\S]*?</tool_call>", "", text)
        text = re.sub(r"<tool_call>[\s\S]*", "", text)
    if '"tool":' in text or '"cmd":' in text:
        return ""
    t = _THINK_BLOCK.sub("", text)
    t = _TOOL_TAGS.sub("", t)
    t = _CODE_BLOCK.sub("", t)
    t = _INLINE_SCRIPT.sub("", t)
    t = _SPECIAL_TOKENS.sub("", t)
    t = _RAW_CODE_LINE.sub("", t)
    t = re.sub(r"^#+\s*", "", t, flags=re.MULTILINE)
    t = t.replace("**", "").replace("*", "").replace("`", "")
    lines = [
        line.strip()
        for line in t.split("\n")
        if line.strip()
        and not line.strip().startswith(("#", "//", "/*", "*", "def ", "import ", "from ", "class ", "cat >", "self.", "pygame."))
        and not (" = " in line and ("(" in line or "{" in line))
    ]
    return " ".join(lines).strip()


def parse_tool_call(text: str) -> tuple[str, dict] | None:
    """Extract tool name and arguments from JSON or Qwen XML format."""
    # Format 1: <tool_call>JSON</tool_call>
    m1 = re.search(r"<tool_call>([\s\S]*?)(?:</tool_call>|$)", text)
    if m1:
        raw = m1.group(1).strip()
        try:
            d = json.loads(raw, strict=False)
            return d.get("tool") or d.get("name"), d.get("parameters") or d.get("arguments") or d
        except Exception:
            pass

    # Format 2: <function=NAME><parameter=KEY>VALUE</parameter></function>
    m2 = re.search(r"<function=([a-zA-Z0-9_-]+)>([\s\S]*?)(?:</function>|$)", text)
    if m2:
        tool_name = m2.group(1).strip()
        params = {}
        for pm in re.finditer(r"<parameter=([a-zA-Z0-9_-]+)>([\s\S]*?)</parameter>", m2.group(2)):
            params[pm.group(1).strip()] = pm.group(2).strip()
        return tool_name, params

    # Format 3: JSON markdown code block
    m3 = re.search(r"```(?:json)?\s*(\{\s*\"(?:tool|name)\"[\s\S]*?\})\s*```", text)
    if m3:
        try:
            d = json.loads(m3.group(1), strict=False)
            return d.get("tool") or d.get("name"), d.get("parameters") or d.get("arguments") or d
        except Exception:
            pass

    # Format 4: Raw JSON containing "tool": "..."
    m4 = re.search(r"(\{\s*\"(?:tool|name)\"\s*:\s*\"[^\"]+\"[\s\S]*?\})", text)
    if m4:
        try:
            d = json.loads(m4.group(1), strict=False)
            return d.get("tool") or d.get("name"), d.get("parameters") or d.get("arguments") or d
        except Exception:
            pass

    return None


def resolve_project_path(target: str, current_cwd: str) -> str:
    """Map friendly name or path to absolute filesystem directory."""
    cleaned = target.strip().lower().replace("_", " ").replace("-", " ")
    for alias, p in PROJECT_ALIASES.items():
        if alias in cleaned or cleaned == alias:
            return p
    expanded = os.path.expanduser(target.strip())
    if os.path.isabs(expanded) and os.path.exists(expanded):
        return expanded
    rel = os.path.join(current_cwd, target.strip())
    if os.path.exists(rel):
        return rel
    anz = os.path.join("/anzym", target.strip())
    if os.path.exists(anz):
        return anz
    return expanded


def search_agent_memories(query: str, source: str = "all", max_results: int = 5) -> str:
    """Search cross-agent persistent memories from Vault, Claude Code, and Antigravity."""
    query = query.strip()
    if not query:
        return "Error: No search query provided for memory search."
    terms = [t for t in query.lower().split() if len(t) > 2]
    if not terms:
        terms = [query.lower()]
    results = []

    # 1. Search Central Vault Notes & AI Memories
    if source in ("all", "vault"):
        vault_dir = Path("/anzym/my-agent/vault")
        if vault_dir.exists():
            for md_file in vault_dir.rglob("*.md"):
                try:
                    text = md_file.read_text(encoding="utf-8", errors="ignore")
                    if any(t in text.lower() for t in terms):
                        matched = []
                        for line in text.splitlines():
                            if any(t in line.lower() for t in terms) and len(line.strip()) > 10:
                                matched.append(line.strip())
                        if matched:
                            rel = md_file.relative_to(vault_dir)
                            results.append({
                                "source": f"Vault Note ({rel})",
                                "snippet": "\n  ".join(matched[:3])
                            })
                except Exception:
                    pass

    # 2. Search Claude Code Project Memories
    if source in ("all", "claude"):
        claude_dir = Path(os.path.expanduser("~/.claude/projects"))
        if claude_dir.exists():
            for md_file in claude_dir.rglob("*.md"):
                try:
                    text = md_file.read_text(encoding="utf-8", errors="ignore")
                    if any(t in text.lower() for t in terms):
                        matched = []
                        for line in text.splitlines():
                            if any(t in line.lower() for t in terms) and len(line.strip()) > 10:
                                matched.append(line.strip())
                        if matched:
                            results.append({
                                "source": f"Claude Memory ({md_file.name})",
                                "snippet": "\n  ".join(matched[:3])
                            })
                except Exception:
                    pass

    # 3. Search Antigravity Brain Sessions & Artifacts
    if source in ("all", "antigravity", "agy"):
        agy_dir = Path(os.path.expanduser("~/.gemini/antigravity-cli/brain"))
        if agy_dir.exists():
            for md_file in list(agy_dir.glob("*/*.md")) + list(agy_dir.glob("*/scratch/*.py")):
                try:
                    text = md_file.read_text(encoding="utf-8", errors="ignore")
                    if any(t in text.lower() for t in terms):
                        matched = []
                        for line in text.splitlines():
                            if any(t in line.lower() for t in terms) and len(line.strip()) > 10:
                                matched.append(line.strip())
                        if matched:
                            results.append({
                                "source": f"Antigravity Artifact ({md_file.name})",
                                "snippet": "\n  ".join(matched[:3])
                            })
                except Exception:
                    pass

            # Search recent Antigravity session transcripts
            transcript_files = sorted(
                agy_dir.glob("*/.system_generated/logs/transcript.jsonl"),
                key=os.path.getmtime,
                reverse=True
            )[:12]
            for tf in transcript_files:
                try:
                    with open(tf, "r", encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            if any(t in line.lower() for t in terms):
                                try:
                                    item = json.loads(line)
                                    text_to_show = item.get("content") or item.get("thinking") or ""
                                    if any(t in text_to_show.lower() for t in terms):
                                        clean = re.sub(r"[{\[\]\"\\`]", "", text_to_show)[:220].strip()
                                        if len(clean) > 20:
                                            sess_id = tf.parent.parent.parent.name[:8]
                                            results.append({
                                                "source": f"Antigravity Session ({sess_id})",
                                                "snippet": clean
                                            })
                                            break
                                except Exception:
                                    pass
                except Exception:
                    pass

    if not results:
        return f"No cross-agent memories found matching '{query}' across {source}."

    out = [f"--- Memory Recall for: '{query}' ---"]
    for r in results[:max_results]:
        src = r["source"]
        snip = r["snippet"]
        out.append(f"• [{src}]\n  {snip}")
    return "\n\n".join(out)


def execute_tool(tool_name: str, args: dict, brain_ref=None) -> str:
    """Execute a local, web, or workspace tool safely with recursive depth."""
    default_ws = "/workspaces_nvme/AnZym_Robot_System" if os.path.exists("/workspaces_nvme/AnZym_Robot_System") else "/Workspaces/AnZym_Robot_System"
    cwd = brain_ref.active_project_dir if brain_ref else default_ws
    try:
        if tool_name == "switch_workspace":
            target = args.get("path") or args.get("project") or args.get("name", "")
            resolved = resolve_project_path(target, cwd)
            print(f" [MILO] 📂 Switching workspace to: {resolved}...", flush=True)
            _set_signal_state("thinking")
            if not os.path.exists(resolved):
                return f"Error: Workspace path {resolved} does not exist."
            if brain_ref:
                brain_ref.active_project_dir = resolved

            # Get 2-level directory overview
            tree_lines = []
            for root, dirs, files in os.walk(resolved):
                dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("__pycache__", "node_modules", "build")]
                rel = os.path.relpath(root, resolved)
                depth = rel.count(os.sep)
                if depth > 2:
                    continue
                indent = "  " * depth
                if rel != ".":
                    tree_lines.append(f"{indent}📂 {os.path.basename(root)}/")
                for f in files[:8]:
                    if not f.startswith("."):
                        tree_lines.append(f"{indent}  📄 {f}")
                if len(tree_lines) > 30:
                    break

            readme_text = ""
            readme_p = os.path.join(resolved, "README.md")
            if os.path.exists(readme_p):
                try:
                    with open(readme_p, "r", encoding="utf-8", errors="ignore") as f:
                        readme_text = f.read(1500)
                except Exception:
                    pass

            return (f"--- Switched Active Workspace to: {resolved} ---\n"
                    f"Workspace File Tree:\n" + "\n".join(tree_lines[:25]) + "\n"
                    + (f"README Overview:\n{readme_text}\n" if readme_text else ""))

        elif tool_name == "get_weather":
            loc = args.get("location", "").strip()
            loc_disp = loc if loc else "local station"
            print(f" [MILO] 🌤️ Checking live weather for {loc_disp}...", flush=True)
            _set_signal_state("thinking")
            url = f"https://wttr.in/{loc}?format=j1" if loc else "https://wttr.in/?format=j1"
            try:
                resp = httpx.get(url, timeout=7.0, follow_redirects=True)
                d = resp.json()
                cur = d.get("current_condition", [{}])[0]
                area = d.get("nearest_area", [{}])[0]
                city = area.get("areaName", [{}])[0].get("value", "")
                region = area.get("region", [{}])[0].get("value", "")
                loc_name = f"{city}, {region}".strip(", ") or (loc or "Local area")
                desc = cur.get("weatherDesc", [{}])[0].get("value", "Clear")
                temp_f = cur.get("temp_F", "")
                feels_f = cur.get("FeelsLikeF", "")
                humidity = cur.get("humidity", "")
                wind_speed = cur.get("windspeedMiles", "")
                wind_dir = cur.get("winddir16Point", "")
                
                today = d.get("weather", [{}])[0]
                today_max = today.get("maxtempF", "")
                today_min = today.get("mintempF", "")
                
                tomorrow = d.get("weather", [{}, {}])[1] if len(d.get("weather", [])) > 1 else {}
                tom_max = tomorrow.get("maxtempF", "")
                tom_min = tomorrow.get("mintempF", "")
                tom_desc = tomorrow.get("hourly", [{}])[4].get("weatherDesc", [{}])[0].get("value", "") if tomorrow.get("hourly") else ""

                summary = (
                    f"Weather for {loc_name}:\n"
                    f"Current: {desc}, {temp_f}°F (feels like {feels_f}°F), humidity {humidity}%, wind {wind_speed} mph {wind_dir}.\n"
                    f"Today: High {today_max}°F, Low {today_min}°F.\n"
                )
                if tom_max and tom_min:
                    summary += f"Tomorrow: {tom_desc}, High {tom_max}°F, Low {tom_min}°F."
                return summary
            except Exception as e:
                try:
                    fallback_url = f"https://wttr.in/{loc}?format=%l:+%C,+%t+(feels+like+%f),+humidity+%h,+wind+%w" if loc else "https://wttr.in/?format=%l:+%C,+%t+(feels+like+%f),+humidity+%h,+wind+%w"
                    fb_resp = httpx.get(fallback_url, timeout=5.0)
                    return f"Weather: {fb_resp.text.strip()}"
                except Exception:
                    return f"Could not retrieve weather for {loc_disp}: {e}"

        elif tool_name == "get_fox_news":
            topic = str(args.get("topic") or args.get("category") or "latest").strip().lower()
            print(f" [MILO] 📰 Fetching Fox News ({topic})...", flush=True)
            _set_signal_state("thinking")
            cat_map = {
                "latest": "latest",
                "top": "latest",
                "news": "latest",
                "headlines": "latest",
                "politics": "politics",
                "pol": "politics",
                "tech": "tech",
                "technology": "tech",
                "science": "tech",
                "world": "world",
                "international": "world",
            }
            if topic in cat_map:
                cat = cat_map[topic]
                feed_url = f"https://moxie.foxnews.com/google-publisher/{cat}.xml"
                try:
                    r = httpx.get(feed_url, timeout=8.0, follow_redirects=True)
                    soup = BeautifulSoup(r.text, "xml")
                    items = soup.find_all("item")
                    max_count = int(args.get("max_results", 5))
                    headlines = []
                    for it in items[:max_count]:
                        title = it.title.text.strip() if it.title else ""
                        desc = it.description.text.strip() if it.description else ""
                        pub = it.pubDate.text.strip() if it.pubDate else ""
                        if title:
                            headlines.append(f"• {title} ({pub[:16]})\n  Summary: {desc[:200]}")
                    if headlines:
                        return f"--- Live Fox News ({cat.title()}) Top Stories ---\n" + "\n\n".join(headlines)
                except Exception:
                    pass

            # Topic search across Fox News via DDGS
            query = f"site:foxnews.com {topic}"
            try:
                results = []
                with DDGS() as ddgs:
                    for n in ddgs.news(query, max_results=int(args.get("max_results", 4))):
                        title = n.get("title", "")
                        body = n.get("body", "")
                        date = n.get("date", "")[:10]
                        results.append(f"• {title} ({date})\n  Summary: {body[:200]}")
                if results:
                    return f"--- Fox News Coverage for '{topic}' ---\n" + "\n\n".join(results)
                return f"No recent Fox News stories found for '{topic}'."
            except Exception as e:
                return f"Error retrieving Fox News: {e}"

        elif tool_name == "get_news":
            topic = str(args.get("topic") or args.get("query") or "top national and world news").strip()
            print(f" [MILO] 📰 Checking live news for: \"{topic}\"...", flush=True)
            _set_signal_state("thinking")
            try:
                results = []
                with DDGS() as ddgs:
                    for n in ddgs.news(topic, max_results=int(args.get("max_results", 4))):
                        title = n.get("title", "")
                        src = n.get("source", "")
                        date = n.get("date", "")[:10]
                        body = n.get("body", "")
                        results.append(f"• {title} [{src}, {date}]\n  Summary: {body[:200]}")
                if results:
                    return f"--- Live News Headlines for '{topic}' ---\n" + "\n\n".join(results)
                return f"No breaking news found for '{topic}'."
            except Exception as e:
                return f"Error fetching news: {e}"

        elif tool_name == "search_web":
            query = args.get("query", "")
            if not query:
                return "Error: No search query provided."
            print(f" [MILO] 🔍 Searching the web for: \"{query}\"...", flush=True)
            _set_signal_state("thinking")
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=int(args.get("max_results", 4))):
                    title = r.get("title", "")
                    snippet = r.get("body", "")
                    href = r.get("href", "")
                    results.append(f"Title: {title}\nSnippet: {snippet}\nURL: {href}\n")
            return f"--- Live Web Search Results for '{query}' ---\n" + ("\n".join(results) if results else "No results found.")

        elif tool_name == "read_web_page":
            url = args.get("url", "")
            if not url:
                return "Error: No URL provided."
            print(f" [MILO] 🌐 Fetching web page: {url}...", flush=True)
            _set_signal_state("thinking")
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            resp = httpx.get(url, headers=headers, timeout=12.0, follow_redirects=True)
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
                tag.decompose()
            text = " ".join(soup.stripped_strings)
            max_chars = int(args.get("max_chars", 3000))
            return f"--- Web Page Content ({url}) ---\n{text[:max_chars]}"

        elif tool_name == "list_dir":
            raw_p = args.get("path", ".")
            p = raw_p if os.path.isabs(raw_p) else os.path.join(cwd, raw_p)
            p = os.path.expanduser(p)
            print(f" [MILO] 📂 Inspecting directory: {p}...", flush=True)
            _set_signal_state("thinking")
            if not os.path.exists(p):
                return f"Error: Directory {p} does not exist."
            
            # Recursive tree view (up to 3 levels deep)
            tree_items = []
            for root, dirs, files in os.walk(p):
                dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("__pycache__", "build", "install", "node_modules", ".git")]
                rel = os.path.relpath(root, p)
                depth = rel.count(os.sep)
                if depth > 2:
                    continue
                prefix = "  " * depth
                if rel != ".":
                    tree_items.append(f"{prefix}📂 {os.path.basename(root)}/")
                for f in files:
                    if not f.startswith("."):
                        full_rel = os.path.join(rel, f) if rel != "." else f
                        tree_items.append(f"{prefix}  📄 {full_rel}")
                if len(tree_items) > 50:
                    break

            return f"Recursive File Tree of {p}:\n" + "\n".join(tree_items[:45])

        elif tool_name == "read_file":
            raw_p = args.get("path", "")
            p = raw_p if os.path.isabs(raw_p) else os.path.join(cwd, raw_p)
            p = os.path.expanduser(p)
            print(f" [MILO] 📄 Reading file: {p}...", flush=True)
            _set_signal_state("thinking")
            if not os.path.exists(p):
                # Try finding file under cwd
                found = list(Path(cwd).glob(f"**/{os.path.basename(raw_p)}"))
                if found:
                    p = str(found[0])
                else:
                    return f"Error: File {raw_p} does not exist in {cwd}."
            max_lines = int(args.get("max_lines", 120))
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                lines = [f.readline() for _ in range(max_lines)]
            return f"--- File Content: {p} ---\n" + "".join(lines)

        elif tool_name == "search_files":
            raw_p = args.get("path", ".")
            base = raw_p if os.path.isabs(raw_p) else os.path.join(cwd, raw_p)
            base = os.path.expanduser(base)
            query = args.get("query", "")
            print(f" [MILO] 🔎 Searching files matching \"{query}\" in {base}...", flush=True)
            _set_signal_state("thinking")
            matches = []
            for root, _, filenames in os.walk(base):
                for fn in filenames:
                    if query.lower() in fn.lower():
                        matches.append(os.path.join(root, fn))
                        if len(matches) >= 20:
                            break
                if len(matches) >= 20:
                    break
            return f"Found matching files for '{query}':\n" + "\n".join(matches)

        elif tool_name == "run_command":
            cmd = args.get("cmd", "")
            target_cwd = args.get("cwd", cwd)
            target_cwd = target_cwd if os.path.isabs(target_cwd) else os.path.join(cwd, target_cwd)
            target_cwd = os.path.expanduser(target_cwd)
            print(f" [MILO] ⚙️ Running command: $ {cmd} (in {target_cwd})...", flush=True)
            _set_signal_state("thinking")
            res = subprocess.run(cmd, shell=True, cwd=target_cwd, capture_output=True, text=True, timeout=15)
            out = res.stdout if res.stdout else res.stderr
            return f"Command output ($ {cmd} in {target_cwd}):\n{out[:2000]}"

        elif tool_name == "generate_image":
            prompt = str(args.get("prompt") or args.get("description") or "").strip()
            if not prompt:
                return "Error: No prompt provided for image generation."
            print(f" [MILO] 🎨 Generating image via FLUX.1: \"{prompt}\"...", flush=True)
            _set_signal_state("thinking")

            output_dir = os.path.expanduser("~/Pictures/Flux_Generations")
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(output_dir, f"flux_{timestamp}.png")

            # Determine candidate FLUX API endpoints
            import urllib.parse
            base_url = brain_ref.api_base if brain_ref and hasattr(brain_ref, "api_base") else CFG.get("api_base", "http://127.0.0.1:8080/v1")
            parsed = urllib.parse.urlparse(base_url)
            host = parsed.hostname or "127.0.0.1"

            candidate_urls = []
            if "flux_api_base" in CFG:
                candidate_urls.append(CFG["flux_api_base"])
            if host not in ("127.0.0.1", "localhost"):
                candidate_urls.append(f"http://{host}:8085/v1/images/generations")
            candidate_urls.extend([
                "http://192.168.19.197:8085/v1/images/generations",
                "http://127.0.0.1:8085/v1/images/generations",
            ])
            # Deduplicate while preserving order
            candidate_urls = list(dict.fromkeys(candidate_urls))

            used_api = False
            for s_url in candidate_urls:
                try:
                    resp = httpx.post(
                        s_url,
                        json={"prompt": prompt, "size": "1024x1024", "response_format": "b64_json"},
                        timeout=90.0
                    )
                    if resp.status_code == 200:
                        import base64
                        data = resp.json()
                        b64_data = data.get("data", [{}])[0].get("b64_json")
                        if b64_data:
                            with open(output_file, "wb") as f:
                                f.write(base64.b64decode(b64_data))
                            used_api = True
                            print(f" [MILO] 🎨 FLUX image generated via {s_url}", flush=True)
                            break
                except Exception:
                    continue

            # Fallback 1: If HTTP API failed and host is remote, run run-flux.sh on Cortex via SSH
            if not used_api and (host not in ("127.0.0.1", "localhost") or not os.path.exists("/workspaces_nvme")):
                try:
                    print(" [MILO] 🔄 Attempting remote FLUX generation on Cortex via SSH fallback...", flush=True)
                    safe_prompt = prompt.replace("'", "'\\''")
                    ssh_cmd = [
                        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "cortex",
                        f"/home/pcarff/Desktop/run-flux.sh '{safe_prompt}'"
                    ]
                    res = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=180)
                    if res.returncode == 0:
                        match = re.search(r"Image generated successfully:\s*(/home/pcarff/Pictures/Flux_Generations/\S+\.png)", res.stdout)
                        if match:
                            remote_png = match.group(1)
                            scp_cmd = ["scp", f"cortex:{remote_png}", output_file]
                            scp_res = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=30)
                            if scp_res.returncode == 0:
                                used_api = True
                                print(f" [MILO] 🎨 FLUX image retrieved from Cortex via SSH to {output_file}", flush=True)
                except Exception as ex:
                    print(f" [MILO] ⚠️ SSH FLUX fallback failed: {ex}", flush=True)

            # Fallback 2: Standalone local sd-cli if running on Cortex
            if not used_api and os.path.exists("/workspaces_nvme/stable-diffusion.cpp/build/bin/sd-cli"):
                cmd = [
                    "/workspaces_nvme/stable-diffusion.cpp/build/bin/sd-cli",
                    "--diffusion-model", "/workspaces_nvme/models/flux/flux1-schnell-Q8_0.gguf",
                    "--clip_l", "/workspaces_nvme/models/flux/clip_l.safetensors",
                    "--t5xxl", "/workspaces_nvme/models/flux/t5xxl_fp8_e4m3fn.safetensors",
                    "--vae", "/workspaces_nvme/models/flux/ae.safetensors",
                    "--params-backend", "cpu",
                    "--max-vram", "10",
                    "--stream-layers",
                    "-p", prompt,
                    "-o", output_file,
                    "-W", "1024",
                    "-H", "1024",
                    "--steps", "4",
                    "--cfg-scale", "1.0",
                    "--sampling-method", "euler",
                    "-t", "16"
                ]
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
                if res.returncode != 0:
                    err_snippet = res.stderr[-400:] if res.stderr else (res.stdout[-400:] if res.stdout else "Process failed")
                    print(f" [MILO] ❌ FLUX generation failed (code {res.returncode}): {err_snippet}", flush=True)
                    return f"Error: Image generation failed (code {res.returncode}): {err_snippet}"

            # Verify the newly created output file specifically
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                subprocess.Popen(["xdg-open", output_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return f"Success: Image generated successfully using FLUX.1 engine and displayed on screen. Saved to {output_file}."

            return f"Error: Image generation completed, but expected output file {output_file} was not created."

        elif tool_name == "recall_memory":
            query = str(args.get("query") or args.get("topic") or args.get("terms") or "").strip()
            source = str(args.get("source") or "all").strip().lower()
            max_results = int(args.get("max_results", 5))
            print(f" [MILO] 🧠 Recalling cross-agent memories for '{query}' (source: {source})...", flush=True)
            _set_signal_state("thinking")
            return search_agent_memories(query, source, max_results)

        elif tool_name == "inspect_image":
            target_path = str(args.get("path") or args.get("file") or args.get("image") or "").strip()
            user_query = str(args.get("query") or args.get("prompt") or args.get("question") or "").strip()
            if not user_query:
                user_query = (
                    "Inspect and analyze this image thoroughly. Identify all hardware components, "
                    "microcontrollers, sensors, actuators, wiring connections, pin labels, "
                    "power rails, and any visible markings or anomalies."
                )

            # Resolve image candidate
            candidate = None
            if target_path and os.path.isfile(target_path):
                candidate = target_path
            elif target_path and os.path.isdir(target_path):
                exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
                files = [os.path.join(target_path, f) for f in os.listdir(target_path) if f.lower().endswith(exts)]
                if files:
                    candidate = max(files, key=os.path.getmtime)
            elif target_path:
                for base in (cwd, "/workspaces/milo_pic", "/workspaces", os.path.expanduser("~/Pictures")):
                    p = os.path.join(base, target_path)
                    if os.path.isfile(p):
                        candidate = p
                        break

            # Default fallback: newest file in /workspaces/milo_pic or ~/Pictures/Screenshots
            if not candidate:
                search_dirs = ["/workspaces/milo_pic", os.path.expanduser("~/Pictures/Screenshots"), os.path.expanduser("~/Pictures")]
                exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
                found = []
                for sdir in search_dirs:
                    if os.path.isdir(sdir):
                        try:
                            for f in os.listdir(sdir):
                                if f.lower().endswith(exts):
                                    fp = os.path.join(sdir, f)
                                    found.append((os.path.getmtime(fp), fp))
                        except Exception:
                            pass
                if found:
                    found.sort(reverse=True)
                    candidate = found[0][1]

            if not candidate or not os.path.isfile(candidate):
                return f"Error: No image found at '{target_path}'. Please specify an image path or place images in /workspaces/milo_pic."

            print(f" [MILO] 👁️ Inspecting visual telemetry: {candidate}...", flush=True)
            _set_signal_state("thinking")

            try:
                import base64
                with open(candidate, "rb") as f:
                    b64_img = base64.b64encode(f.read()).decode("utf-8")

                mime = "image/jpeg"
                if candidate.lower().endswith(".png"):
                    mime = "image/png"
                elif candidate.lower().endswith(".webp"):
                    mime = "image/webp"

                api_url = f"{brain_ref.api_base}/chat/completions" if brain_ref else f"{CFG.get('api_base', 'http://192.168.19.197:8080/v1')}/chat/completions"
                payload = {
                    "model": brain_ref.model if brain_ref else "qwen",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"You are MILO, lead robotics flight director. Deliver a crisp, spoken visual briefing (under 150 words): {user_query}"
                                },
                                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64_img}"}}
                            ]
                        }
                    ],
                    "max_tokens": 400,
                    "temperature": 0.2
                }

                resp = httpx.post(api_url, json=payload, timeout=60.0)
                if resp.status_code == 200:
                    data = resp.json()
                    desc = data["choices"][0]["message"]["content"].strip()
                    desc = _SPECIAL_TOKENS.sub("", desc).strip()
                    return f"[Visual Analysis for {os.path.basename(candidate)}]:\n{desc}"
                else:
                    return f"Error: Vision inference failed with status {resp.status_code}: {resp.text}"
            except Exception as ex:
                return f"Error: Visual inspection failed: {ex}"

        return f"Unknown tool: {tool_name}"
    except Exception as e:
        return f"Tool execution failed: {e}"


TOOL_PROMPT = """
### OPERATIONAL DIRECTIVE: MULTI-STEP TOOLS & GUARANTEED SPOKEN ANSWERS
You are MILO (Machine Intelligence Liaison Officer), lead robotics flight director.

RULES:
1. When you need to inspect directories, read code files, inspect images/hardware photos, recall past memories, check live weather, search live news, or generate images, invoke tools using <tool_call>{"tool": "name", ...}</tool_call>.
2. `list_dir` returns the full recursive tree (all files and subfolders), so you can read target files immediately on your next step.
3. As soon as you have inspected the necessary information or generated the asset, STOP calling tools and deliver your spoken flight director answer directly.
4. Conclude your answer with a specific, direct question guiding the user on what action to take next.

Available Tools:
- inspect_image(path, query): Inspect, analyze, and diagnose any image, photo, screenshot, or circuit diagram. If path is omitted, automatically inspects the newest photo in /workspaces/milo_pic. Can answer specific visual questions about wiring, components, and hardware.
- recall_memory(query, source): Recall past memories, hardware designs, user preferences, and conversation history across Antigravity, Claude Code, and the Vault. Source can be 'all', 'claude', 'antigravity', or 'vault'.
- get_weather(location): Real-time temperature, humidity, wind, and forecast. If location is omitted, checks local station.
- get_fox_news(category_or_topic, max_results): Live Fox News wire stories and breaking headlines. Category can be 'latest', 'politics', 'tech', 'world', or a specific topic search.
- get_news(topic, max_results): Live breaking global/national news across all verified wire sources.
- search_web(query): General web search via DuckDuckGo.
- read_web_page(url): Extract article or page content.
- switch_workspace(path): Switch active workspace.
- list_dir(path): Inspect directories.
- read_file(path, max_lines): Read source code or files.
- search_files(path, query): Search for files by name.
- run_command(cmd, cwd): Execute shell commands.
- generate_image(prompt): Generate an image using the FLUX.1 diffusion engine on Cortex and display it on the user's screen.
"""


class WarmBrain:
    def __init__(self, model: str | None = None, can_use_tool=None,
                 resume_id: str | None = None):
        self.api_base = CFG.get("api_base", "http://127.0.0.1:8080/v1")
        self.model = model or CFG.get("model", "qwen3.8-27b")
        self.active_project_dir = "/Workspaces/AnZym_Robot_System"
        self._can_use_tool = can_use_tool
        self.session = {"turns": 0, "out_tokens": 0, "in_tokens": 0, "cost": 0.0}
        self.messages = []
        self._dirty = False
        self._interrupted = False
        self.system_prompt = self._load_system_prompt()
        self.permission_mode = CFG.get("permission_mode", "ask")

    def _get_workspace_snapshot(self) -> str:
        """Fetch live snapshot of current active workspace, local time, and station."""
        try:
            now_str = datetime.now().strftime("%A, %B %d, %Y, %I:%M %p")
            time_header = f"\n[Current Time: {now_str}] [Station: Moncks Corner / Charleston, SC]\n"
            cwd = self.active_project_dir
            if not os.path.exists(cwd):
                return time_header
            entries = os.listdir(cwd)
            dirs = [f"{e}/" for e in sorted(entries) if os.path.isdir(os.path.join(cwd, e)) and not e.startswith(".")]
            files = [e for e in sorted(entries) if os.path.isfile(os.path.join(cwd, e)) and not e.startswith(".")]
            return (f"{time_header}"
                    f"[Active Workspace: {cwd}]\n"
                    f"Subsystems: {', '.join(dirs[:15])}\n"
                    f"Files: {', '.join(files[:15])}")
        except Exception:
            return ""

    def _load_system_prompt(self) -> str:
        agent_dir = Path(os.path.expanduser(CFG.get("agent_dir", "/anzym/my-agent")))
        prompt_parts = [DISCIPLINE, TOOL_PROMPT]
        
        for filename in ("AGENT.md", "CLAUDE.md", "SYSTEM.md"):
            p = agent_dir / filename
            if p.exists():
                try:
                    prompt_parts.append(p.read_text(encoding="utf-8"))
                    log(f"[brain] loaded persona from {p}")
                    break
                except Exception as e:
                    log(f"[brain] error reading {p}: {e}")
                    
        for extra in CFG.get("extra_dirs", []):
            extra_path = Path(os.path.expanduser(extra))
            idx = extra_path / "VAULT-INDEX.md"
            if idx.exists():
                try:
                    prompt_parts.append(f"## Vault Context ({extra})\n{idx.read_text(encoding='utf-8')[:3000]}")
                    log(f"[brain] loaded vault index from {idx}")
                except Exception:
                    pass
            readme = extra_path / "README.md"
            if readme.exists() and "vault" not in str(extra_path).lower():
                try:
                    prompt_parts.append(f"## Workspace Overview ({extra})\n{readme.read_text(encoding='utf-8')[:4000]}")
                    log(f"[brain] loaded workspace summary from {readme}")
                except Exception:
                    pass

        return "\n\n".join(prompt_parts)

    async def start(self):
        full_prompt = self.system_prompt + self._get_workspace_snapshot()
        self.messages = [{"role": "system", "content": full_prompt}]
        self._dirty = False
        self._interrupted = False
        log(f"[brain] local brain connected to {self.api_base} (model={self.model}, cwd={self.active_project_dir})")

    async def stop(self):
        self.messages.clear()
        self._dirty = False

    async def interrupt(self):
        self._interrupted = True
        self._dirty = False

    async def reset_turn(self, timeout: float = 8.0):
        self._dirty = False
        self._interrupted = False

    async def set_permission_mode(self, mode: str):
        self.permission_mode = mode
        log(f"[brain] permission mode set to: {mode}")

    async def context_usage(self):
        return {
            "total_tokens": self.session["in_tokens"] + self.session["out_tokens"],
            "turns": self.session["turns"],
        }

    async def command(self, cmd: str) -> str:
        parts = cmd.strip().split()
        if not parts:
            return ""
        verb = parts[0].lower()
        if verb == "/clear":
            full_prompt = self.system_prompt + self._get_workspace_snapshot()
            self.messages = [{"role": "system", "content": full_prompt}]
            self.session = {"turns": 0, "out_tokens": 0, "in_tokens": 0, "cost": 0.0}
            return "Conversation history cleared."
        elif verb == "/compact":
            if len(self.messages) > 9:
                self.messages = [self.messages[0]] + self.messages[-8:]
            return "Session compacted."
        elif verb == "/model" and len(parts) > 1:
            self.model = parts[1]
            return f"Switched model to {self.model}."
        elif verb == "/effort":
            return "Effort level updated."
        elif verb in ("/cd", "/workspace") and len(parts) > 1:
            res = execute_tool("switch_workspace", {"path": parts[1]}, self)
            return res
        return f"Command acknowledged: {cmd}"

    async def _query_llm(self, messages: list) -> str:
        """Non-streaming query for tool decision."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "temperature": 0.3,
            "max_tokens": 2048,
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(f"{self.api_base}/chat/completions", json=payload)
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            return ""

    async def ask_stream(self, utterance: str):
        self._dirty = True
        self._interrupted = False
        
        # Inject live workspace snapshot
        snapshot = self._get_workspace_snapshot()
        user_msg = utterance
        if snapshot:
            user_msg = f"{utterance}\n\n[Active Workspace Telemetry]:{snapshot}"
        self.messages.append({"role": "user", "content": user_msg})

        # Step 1: Multi-turn tool execution loop (up to 5 sequential tool calls)
        max_tool_turns = 5
        for turn_idx in range(max_tool_turns):
            reply = await self._query_llm(self.messages)
            parsed = parse_tool_call(reply)
            if not parsed:
                # No more tools needed; reply contains spoken answer
                break
            
            tool_name, args = parsed
            log(f"[brain] intercepted tool call ({turn_idx+1}/{max_tool_turns}): {tool_name} with args {args}")

            tool_result = execute_tool(tool_name, args, self)
            log(f"[brain] tool output ({len(tool_result)} chars):\n{tool_result[:300]}...")

            self.messages.append({"role": "assistant", "content": reply})
            
            if turn_idx == max_tool_turns - 1:
                # Last allowed tool turn: force spoken delivery
                self.messages.append({
                    "role": "user",
                    "content": f"[Tool Result from {tool_name}]:\n{tool_result}\n\n"
                               f"You now have all necessary data. Deliver your final spoken flight director answer now (do NOT call any tools)."
                })
            else:
                self.messages.append({
                    "role": "user",
                    "content": f"[Tool Result from {tool_name}]:\n{tool_result}\n\n"
                               f"If you need to read a specific code file, call read_file. Otherwise, deliver your spoken flight director answer directly."
                })

        # Step 2: Stream final voice response to Kokoro TTS
        buf = ""
        full_reply = ""
        payload = {
            "model": self.model,
            "messages": self.messages,
            "stream": True,
            "temperature": 0.7,
            "max_tokens": 2048,
        }

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream("POST", f"{self.api_base}/chat/completions", json=payload) as response:
                    if response.status_code != 200:
                        err_text = await response.aread()
                        log(f"[brain] server error {response.status_code}: {err_text.decode('utf-8', errors='ignore')}")
                        yield "I had trouble connecting to the local inference server."
                        self._dirty = False
                        return

                    async for line in response.aiter_lines():
                        if self._interrupted:
                            log("[brain] stream interrupted")
                            break
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            delta = chunk["choices"][0]["delta"].get("content", "")
                            if delta:
                                if "<|im_end|>" in delta or "<|endoftext|>" in delta:
                                    delta = _SPECIAL_TOKENS.sub("", delta)
                                    buf += delta
                                    full_reply += delta
                                    break
                                buf += delta
                                full_reply += delta
                                while True:
                                    m_sent = _SENTENCE_END.search(buf)
                                    if not m_sent:
                                        break
                                    sentence, buf = buf[:m_sent.end()].strip(), buf[m_sent.end():]
                                    cleaned = _clean_text(sentence)
                                    if cleaned:
                                        yield cleaned
                        except Exception:
                            continue
        except Exception as e:
            log(f"[brain] error querying {self.api_base}: {e}")
            yield "Sorry, I lost connection to the local model."

        tail = _clean_text(buf)
        if tail and not self._interrupted:
            yield tail

        if full_reply:
            self.messages.append({"role": "assistant", "content": _clean_text(full_reply)})
            self.session["turns"] += 1
            self.session["out_tokens"] += len(full_reply.split()) * 2
            self.session["in_tokens"] += len(utterance.split()) * 2

        self._dirty = False
        self._interrupted = False
