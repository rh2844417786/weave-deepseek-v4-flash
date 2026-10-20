"""Existing-image four-rank graph gate command; no network dependency installs."""
from pathlib import Path
from scripts.run_c32_comparison import gate_command


def graph_gate_command(image,work,model,name,allocation):
    command=gate_command(image,work,model,name,allocation,'hopper')
    offset=command.index('/workspace/tests/gpu/check_hopper_serving.py')
    command[offset:]=['/workspace/tests/gpu/check_hopper_graph.py','--output-dir',f'/work/{name}/graph-gate']
    command[command.index('--output-dir')+1]=f'/work/{name}/graph-gate'
    return command


def stage_event_gate_command(image,work,model,name,allocation):
    command=graph_gate_command(image,work,model,name,allocation)
    command[command.index('--name')+1]='weave-stage-event-gate-'+name
    command[command.index('/workspace/tests/gpu/check_hopper_graph.py')]='/workspace/tests/gpu/check_graph_stage_events.py'
    for index,value in enumerate(command):
        if value=='--output-dir':command[index+1]=f'/work/{name}/stage-event-gate'
    return command


def native_p0_gate_command(image,work,model,name,allocation):
    command=stage_event_gate_command(image,work,model,name,allocation)
    command[command.index('--name')+1]='weave-native-p0-gate-'+name
    command[command.index('/workspace/tests/gpu/check_graph_stage_events.py')]='/workspace/tests/gpu/check_native_p0_events.py'
    for index,value in enumerate(command):
        if value=='--output-dir':command[index+1]=f'/work/{name}/native-p0-gate'
    return command
