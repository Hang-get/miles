# Harbor agent training through the official Tinker proxy

Install the independent client package; running it does not import or require
Miles. The Harbor version is pinned to the same fork used by the Miles Harbor
example. Request/render/stop/tool/SSE behavior comes unchanged from the pinned
official `tinker_cookbook.capture.proxy`.

```bash
uv pip install 'tinker-agent-rollouts[harbor] @ git+https://github.com/radixark/miles@<commit>#subdirectory=examples/multi_lora/tinker_agent'
```

Create `trial.json` with Harbor `TrialConfig` fields, omitting `task` (the
recipe supplies each dataset task). For example, Terminus 2 with Docker:

```json
{
  "trials_dir": "/tmp/harbor-trials",
  "environment": {"type": "docker"},
  "agent": {
    "name": "terminus-2",
    "model_name": "openai/policy",
    "kwargs": {
      "api_base": "{proxy_url}/v1",
      "llm_kwargs": {"api_key": "{proxy_api_key}"},
      "temperature": 1.0
    }
  }
}
```

Use Harbor's normal schema for agent kwargs, sandbox/provider credentials,
timeouts and verifier settings. The recipe substitutes only `{proxy_url}`
and `{proxy_api_key}` in this JSON. The URL includes a trajectory-specific
`/r/traj/<id>` prefix. For `mini-swe-agent`, put `{proxy_url}/v1` in
`agent.env.OPENAI_BASE_URL` and `{proxy_api_key}` in `agent.env.OPENAI_API_KEY`.
For `claude-code`, use `{proxy_url}` in `agent.env.ANTHROPIC_BASE_URL` and
`{proxy_api_key}` in `agent.env.ANTHROPIC_API_KEY`. Choose a model/renderer
that supports the agent's requested tools and message format.

```bash
export TINKER_API_KEY=tml-your-key
python -m tinker_agent.train \
  model_name=Qwen/Qwen3-8B renderer_name=qwen3_instruct \
  tasks_dir=/data/harbor-tasks trial_config=trial.json \
  base_url=http://miles-gateway:8000 \
  listen_host=0.0.0.0 advertised_host=host.docker.internal \
  groups_per_batch=4 group_size=4 max_tokens=8192
```

Omit `base_url` to use the official Tinker service. For remote sandboxes,
`advertised_host` must resolve to this client machine from the sandbox; use
your provider's reachable host address. Each group gets a random proxy auth
token. Rendering and trace bookkeeping run on the client machine. Agent
sampling parameters come from the agent; `max_tokens` supplies the proxy's
default only when the request omits it.

One rollout group fixes one sampling client. Every trial closes admission,
drains accepted samples, then converts immutable input/output tokens and
logprobs into cookbook transitions. All sampled calls, including compaction
and repeated prompts, are retained. Cookbook computes group advantages and
merges only exact token-prefix continuations into training Datums.

A completed trial or an agent timeout with a verifier reward is scored using
`reward_key` (default `reward`). Missing/non-finite rewards, other Harbor
exceptions, cancelled/failed sampling or incomplete records abort the group.
There is no synthetic zero reward for infrastructure errors. Task and agent
limits belong to Harbor; the proxy adds no turn/token budgets, refusal
matching, stop overrides or `reasoning_content` reconstruction. Capture's
export queue is not used for training storage. Crashed groups are discarded;
there is no partial-episode recovery.

The isolated CPU tests compare the wrapped and bare official proxy and test
Harbor configuration, trial outcomes and token-to-Datum conversion with a
scripted trial. The registered H100 gateway test exercises real sampling and
PPO training. Scripted trials do not validate live sandbox provisioning or
agent binaries; those require an actual Harbor environment.
