"""
LabOS Web UI — Gradio chat interface with streaming and agent step display.
"""

import os
import re
import sys
import time
import json
import queue
import threading
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

import gradio as gr


class LabOSUI:
    """Gradio-based web interface for LabOS."""

    def __init__(self):
        self.conversation_history: list = []
        self.created_files: list = []
        self.output_dir = Path("./agent_outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------

    def parse_steps(self, output: str) -> List[Dict]:
        """Parse agent output into structured steps."""
        steps = []
        patterns = [
            r'━+\s*Step\s+(\d+)\s*━+',
            r'=+\s*Step\s+(\d+)\s*=+',
            r'\[Step\s+(\d+)\]',
        ]
        blocks = None
        for pat in patterns:
            if re.findall(pat, output, re.IGNORECASE):
                blocks = re.split(pat, output, flags=re.IGNORECASE)
                break

        if not blocks or len(blocks) < 2:
            if output.strip():
                return [{"step_number": 1, "tools": [], "observations": [output.strip()[:500]],
                         "duration": None, "timestamp": datetime.now().strftime("%H:%M:%S"), "status": "completed"}]
            return []

        for i in range(1, len(blocks), 2):
            if i + 1 >= len(blocks):
                break
            try:
                num = int(blocks[i])
            except (ValueError, IndexError):
                continue
            content = blocks[i + 1] if i + 1 < len(blocks) else ""
            step: Dict = {
                "step_number": num,
                "tools": [],
                "observations": [],
                "duration": None,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "status": "completed" if "Duration" in content else "in_progress",
            }
            # Tool calls
            for m in re.findall(r"Calling tool:\s*'([^']+)'\s*with arguments:\s*(\{[^}]*\})", content, re.DOTALL):
                step["tools"].append({"name": m[0], "arguments": m[1][:80]})
            # Duration
            dm = re.search(r"Duration\s+([\d.]+)\s*seconds", content)
            if dm:
                step["duration"] = float(dm.group(1))
            # Observations
            for obs in re.findall(r"Observations?:\s*(.*?)(?=\[Step|\n━|$)", content, re.DOTALL):
                if obs.strip():
                    step["observations"].append(obs.strip()[:500])
            if not step["observations"]:
                lines = [l.strip() for l in content.split("\n") if l.strip()][:3]
                if lines:
                    step["observations"].append(" ".join(lines))
            steps.append(step)
        return steps

    def extract_files(self, output: str) -> List[Dict]:
        """Extract created files from output."""
        files = []
        for pat in [r'Successfully created.*?:\s*([^\n]+)', r'Created file:\s*([^\n]+)',
                     r'Saved to:\s*([^\n]+)', r'Output file:\s*([^\n]+)']:
            for m in re.findall(pat, output, re.IGNORECASE):
                p = m.strip()
                if os.path.exists(p):
                    try:
                        st = os.stat(p)
                        files.append({"path": p, "name": os.path.basename(p), "size": st.st_size})
                    except Exception:
                        files.append({"path": p, "name": os.path.basename(p), "size": 0})
        return files

    def format_steps(self, steps: List[Dict], done: bool = False, elapsed: float = None) -> str:
        if not steps:
            return "**Execution Steps**\n\nWaiting for execution..."
        hdr = "**Execution Steps**\n\n"
        if done and elapsed:
            hdr += f"Task completed in {elapsed:.1f}s\n\n"
        for s in steps:
            icon = "OK" if s["status"] == "completed" else ">>"
            hdr += f"### [{icon}] Step {s['step_number']}"
            if s["duration"]:
                hdr += f" ({s['duration']:.1f}s)"
            hdr += f" [{s['timestamp']}]\n\n"
            if s["tools"]:
                for t in s["tools"]:
                    hdr += f"  Tool: {t['name']}  Args: {t['arguments']}\n"
                hdr += "\n"
            for obs in s["observations"]:
                hdr += f"  {obs[:300]}\n\n"
            hdr += "---\n\n"
        return hdr

    def format_files(self, files: List[Dict]) -> str:
        if not files:
            return "**Created Files**\n\nNone."
        lines = [f"**Created Files** ({len(files)})\n"]
        for f in files:
            sz = f"{f['size']} B" if f["size"] < 1024 else f"{f['size']/1024:.1f} KB"
            lines.append(f"  {f['name']} ({sz}) — `{f['path']}`")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Gradio interface
    # ------------------------------------------------------------------

    def create_interface(self):
        css = """
        .gradio-container { font-family: 'Segoe UI', sans-serif !important; }
        .main-header { background: linear-gradient(90deg, #2563eb 0%, #7c3aed 100%);
            color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; text-align: center; }
        """
        with gr.Blocks(css=css, title="LabOS — Biomedical Research Agent", theme=gr.themes.Soft()) as ui:
            gr.HTML("""
            <div class="main-header">
                <h1 style="margin:0; color:white;">LabOS</h1>
                <p style="margin:5px 0 0; opacity:0.9; color:white;">Self-Evolving Multi-Agent Framework for Biomedical Research</p>
            </div>
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Chat")
                    chatbot = gr.Chatbot(height=400, show_label=False, type="messages")
                    with gr.Row():
                        msg = gr.Textbox(placeholder="Enter your request...", show_label=False, scale=4)
                        send = gr.Button("Send", variant="primary", scale=1)
                    clear = gr.Button("Clear", variant="secondary")

                with gr.Column(scale=1):
                    with gr.Tabs():
                        with gr.Tab("Steps"):
                            steps_md = gr.Markdown("**Execution Steps**\n\nWaiting...", height=400)
                        with gr.Tab("Files"):
                            files_md = gr.Markdown("**Created Files**\n\nNone.", height=300)
                            dl = gr.File(label="Download", file_count="multiple", visible=True)
                        with gr.Tab("Status"):
                            status_md = gr.Markdown("**Status**\n\nReady.", height=400)

            def on_submit(message, history):
                if not message.strip():
                    yield history, "", "", "", []
                    return

                history = history + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": "Processing..."},
                ]
                yield history, "**Execution Steps**\n\nStarting...", "**Created Files**\n\nNone.", "**Status**\n\nRunning...", []

                import labos.core as core
                out_q: queue.Queue = queue.Queue()
                acc = ""

                class _Cap:
                    def __init__(self, orig, q):
                        self.orig, self.q = orig, q
                    def write(self, t):
                        self.orig.write(t); self.orig.flush(); self.q.put(("out", t))
                    def flush(self):
                        self.orig.flush()

                orig_out = sys.stdout
                cap = _Cap(orig_out, out_q)

                def _run():
                    try:
                        sys.stdout = cap
                        t0 = time.time()
                        resp = core.manager_agent.run(message, reset=False)
                        out_q.put(("resp", resp, time.time() - t0))
                    except Exception as e:
                        out_q.put(("err", str(e)))
                    finally:
                        sys.stdout = orig_out
                        out_q.put(("done", None))

                thr = threading.Thread(target=_run, daemon=True)
                thr.start()

                last_t = time.time()
                while True:
                    try:
                        item = out_q.get(timeout=0.5)
                        if item[0] == "out":
                            acc += item[1]
                            if time.time() - last_t >= 1.0:
                                st = self.parse_steps(acc)
                                yield history, self.format_steps(st), "Scanning...", f"Running... ({len(st)} steps)", []
                                last_t = time.time()
                        elif item[0] == "resp":
                            resp, elapsed = item[1], item[2]
                            resp_str = str(resp)
                            final_steps = self.parse_steps(acc)
                            files = self.extract_files(resp_str + "\n" + acc)
                            history[-1]["content"] = resp_str[:2000]
                            dl_paths = [f["path"] for f in files if os.path.exists(f["path"])]
                            yield (history, self.format_steps(final_steps, True, elapsed),
                                   self.format_files(files),
                                   f"**Status**\n\nDone in {elapsed:.1f}s — {len(final_steps)} steps, {len(files)} files",
                                   dl_paths)
                            break
                        elif item[0] == "err":
                            history[-1]["content"] = f"Error: {item[1]}"
                            yield history, f"Error: {item[1]}", "", f"Error: {item[1]}", []
                            break
                        elif item[0] == "done":
                            break
                    except queue.Empty:
                        if time.time() - last_t >= 1.0 and acc:
                            st = self.parse_steps(acc)
                            yield history, self.format_steps(st), "Scanning...", f"Running... ({len(st)} steps)", []
                            last_t = time.time()

                thr.join(timeout=5)

            def on_clear():
                return [], "**Execution Steps**\n\nWaiting...", "**Created Files**\n\nNone.", "**Status**\n\nReady.", []

            send.click(on_submit, [msg, chatbot], [chatbot, steps_md, files_md, status_md, dl])
            msg.submit(on_submit, [msg, chatbot], [chatbot, steps_md, files_md, status_md, dl])
            clear.click(on_clear, outputs=[chatbot, steps_md, files_md, status_md, dl])

        return ui

    def launch(self, **kwargs):
        self.create_interface().launch(**kwargs)


def main(**kwargs):
    """Launch the LabOS web interface."""
    import labos.core as core
    print("LabOS: Starting web interface...")
    agent = core.initialize(
        use_template=kwargs.pop("use_template", True),
        use_mem0=kwargs.pop("use_mem0", False),
    )
    if agent is None:
        print("Failed to initialise LabOS.")
        return
    ui = LabOSUI()
    ui.launch(
        server_name=kwargs.get("server_name", "0.0.0.0"),
        server_port=kwargs.get("port", 7860),
        share=kwargs.get("share", False),
    )
