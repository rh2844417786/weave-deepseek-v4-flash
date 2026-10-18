"""Existing-image four-rank graph gate command; no network dependency installs."""
from pathlib import Path
from scripts.run_c32_comparison import gate_command


def graph_gate_command(image,work,model,name,allocation):
    command=gate_command(image,work,model,name,allocation,'hopper')
    offset=command.index('/workspace/tests/gpu/check_hopper_serving.py')
    command[offset:]=['/workspace/tests/gpu/check_hopper_graph.py','--output-dir',f'/work/{name}/graph-gate']
    command[command.index('--output-dir')+1]=f'/work/{name}/graph-gate'
    return command
