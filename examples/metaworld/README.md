# Run Aloha Sim

## With Docker

```bash
export SERVER_ARGS="--env ALOHA_SIM"
docker compose -f examples/aloha_sim/compose.yml up --build
```

## Without Docker

Terminal window 1:

```bash
# Create virtual environment
uv venv --python 3.10 examples/metaworld/.venv
source examples/metaworld/.venv/bin/activate
uv pip sync examples/metaworld/requirements.txt
uv pip install -e packages/openpi-client
uv pip install third_party/metaworld

# Run the simulation
MUJOCO_GL=egl python examples/metaworld/main.py
```

Note: If you are seeing EGL errors, you may need to install the following dependencies:

```bash
sudo apt-get install -y libegl1-mesa-dev libgles2-mesa-dev
```

Terminal window 2:

```bash
# Run the server
uv run scripts/serve_policy.py --env ALOHA_SIM
```
