import subprocess

path='config/vox-256.yaml'
subprocess.call(['python','run.py',
                     "--config",path])
