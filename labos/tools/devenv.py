"""
LabOS DevEnv Tools — shell, conda, pip, script execution.
"""

import os
import subprocess
from pathlib import Path

from smolagents import tool


@tool
def run_shell_command(command: str, working_directory: str = ".") -> str:
    """Execute a shell command and return the output.

    Args:
        command: The shell command to execute
        working_directory: Working directory (default: ".")

    Returns:
        Command output or error message
    """
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            cwd=working_directory, timeout=1800,
        )
        out = f"Command: {command}\nReturn code: {result.returncode}\nSTDOUT:\n{result.stdout}\n"
        if result.stderr:
            out += f"STDERR:\n{result.stderr}\n"
        return out
    except subprocess.TimeoutExpired:
        return f"Command timed out (30 min): {command}"
    except Exception as exc:
        return f"Error executing '{command}': {exc}"


@tool
def create_conda_environment(env_name: str, python_version: str = "3.9") -> str:
    """Create a new conda environment.

    Args:
        env_name: Environment name
        python_version: Python version (default: 3.9)

    Returns:
        Creation result
    """
    return run_shell_command(f"conda create -n {env_name} python={python_version} -y")


@tool
def install_packages_conda(env_name: str, packages: str) -> str:
    """Install packages in a conda environment.

    Args:
        env_name: Conda environment name
        packages: Space-separated packages

    Returns:
        Installation result
    """
    return run_shell_command(f"conda activate {env_name} && conda install {packages} -y")


@tool
def install_packages_pip(env_name: str, packages: str) -> str:
    """Install pip packages in a conda environment.

    Args:
        env_name: Conda environment name
        packages: Space-separated packages

    Returns:
        Installation result
    """
    return run_shell_command(f"conda activate {env_name} && pip install {packages}")


@tool
def check_gpu_status(dummy_param: str = "") -> str:
    """Check GPU availability via nvidia-smi.

    Args:
        dummy_param: Unused

    Returns:
        GPU status information
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
             "--format=csv"],
            capture_output=True, text=True, timeout=30,
        )
        return f"GPU Status:\n{result.stdout}" if result.returncode == 0 else f"Error: {result.stderr}"
    except FileNotFoundError:
        return "nvidia-smi not found. No NVIDIA GPUs available."
    except Exception as exc:
        return f"GPU check error: {exc}"


@tool
def create_script(
    script_name: str, script_content: str, directory: str = ".", script_type: str = "python"
) -> str:
    """Create a script file.

    Args:
        script_name: Script filename (with extension)
        script_content: Script content
        directory: Target directory (default: ".")
        script_type: Script type for info (python, bash)

    Returns:
        Creation result
    """
    try:
        path = Path(directory) / script_name
        path.parent.mkdir(parents=True, exist_ok=True)
        content = script_content.strip()
        for q in ('"""', "'''"):
            if content.startswith(q) and content.endswith(q):
                content = content[3:-3].strip()
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        if script_name.endswith(".sh") or script_type.lower() in ("bash", "shell"):
            os.chmod(path, 0o755)
        return f"Created {script_type} script: {path}"
    except Exception as exc:
        return f"Error creating '{script_name}': {exc}"


@tool
def run_script(
    script_path: str, env_name: str = None, working_directory: str = None, interpreter: str = "python"
) -> str:
    """Run a script with optional conda environment.

    Args:
        script_path: Path to script
        env_name: Conda environment (optional)
        working_directory: Working directory (optional)
        interpreter: Interpreter (python, bash)

    Returns:
        Script output
    """
    name = Path(script_path).name
    if name.endswith(".sh") or interpreter.lower() in ("bash", "shell"):
        cmd = f"bash {script_path}"
    else:
        cmd = f"python {script_path}"
    if env_name:
        cmd = f"conda activate {env_name} && {cmd}"
    return run_shell_command(cmd, working_directory or ".")


@tool
def create_and_run_script(
    script_name: str, script_content: str, directory: str = ".",
    env_name: str = None, interpreter: str = "python"
) -> str:
    """Create and immediately run a script.

    Args:
        script_name: Script filename
        script_content: Script content
        directory: Target directory
        env_name: Conda environment (optional)
        interpreter: Interpreter (python, bash)

    Returns:
        Creation and execution result
    """
    cr = create_script(script_name, script_content, directory, interpreter)
    if "Error" in cr:
        return cr
    path = str(Path(directory) / script_name)
    rr = run_script(path, env_name, directory, interpreter)
    return f"{cr}\n\nExecution:\n{rr}"


@tool
def create_requirements_file(requirements: str, directory: str = ".") -> str:
    """Create a requirements.txt file.

    Args:
        requirements: Content (one package per line)
        directory: Target directory

    Returns:
        Creation result
    """
    try:
        path = Path(directory) / "requirements.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(requirements)
        return f"Created requirements.txt: {path}"
    except Exception as exc:
        return f"Error creating requirements.txt: {exc}"


@tool
def monitor_training_logs(log_file_path: str, lines: int = 50) -> str:
    """Read last N lines of a training log.

    Args:
        log_file_path: Path to log file
        lines: Number of lines (default: 50)

    Returns:
        Tail of the log file
    """
    return run_shell_command(f"tail -n {lines} {log_file_path}")
