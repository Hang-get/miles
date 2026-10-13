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
`/r/run/<run_id>/group/<group_idx>/traj/<traj_idx>` prefix. For `mini-swe-agent`, put `{proxy_url}/v1` in
`agent.env.OPENAI_BASE_URL` and `{proxy_api_key}` in `agent.env.OPENAI_API_KEY`.
For `claude-code`, use `{proxy_url}` in `agent.env.ANTHROPIC_BASE_URL` and
`{proxy_api_key}` in `agent.env.ANTHROPIC_API_KEY`. Choose a model/renderer
that supports the agent's requested tools and message format.

```bash
export TINKER_API_KEY=tml-your-key
python -m tinker_agent.train \
  config.model_name=Qwen/Qwen3-8B config.renderer_name=qwen3_instruct \
  config.learning_rate=3e-5 config.max_tokens=8192 \
  config.log_path=/tmp/harbor-tinker config.loss_fn=ppo \
  config.dataset_builder.tasks_dir=/data/harbor-tasks \
  config.dataset_builder.groups_per_batch=4 config.dataset_builder.group_size=4 \
  config.base_url=http://miles-gateway:8000 \
  harbor.trial_config=trial.json \
  harbor.listen_host=0.0.0.0 harbor.advertised_host=host.docker.internal
```

Omit `config.base_url` to use the official Tinker service. For remote sandboxes,
`harbor.advertised_host` must resolve to this client machine from the sandbox; use
your provider's reachable host address. Each group gets a random proxy auth
token. Rendering and official capture run on the client machine. Agent
sampling parameters come from the agent; `config.max_tokens` supplies the proxy's
default only when the request omits it.

`config` is the official cookbook `rl.train.Config`, with its native names,
required fields and defaults. The CLI supplies only the recipe name and the
Harbor dataset builder type. Dataset options live under `config.dataset_builder`;
`harbor` contains only the trial config path, reward key and proxy connection /
concurrency options. For example, `config.load_checkpoint_path`,
`config.rolling_save_every` and `config.async_config.max_steps_off_policy` pass
directly to cookbook without recipe-specific copies or a field allowlist.
Use `python -m tinker_agent.train --help` for the complete configuration.

The recipe installs `HarborRolloutStrategy` in `config.rollout_error_tolerance`;
setting that field explicitly is rejected. Set sampling temperature in the
Harbor agent config; non-default `config.temperature` is rejected because the
external agent owns its sampling requests. An omitted renderer is resolved
with cookbook's model recommendation. Training defaults, including the loss,
LoRA rank, save cadence and checkpoint TTL, come from cookbook. Set
`config.ttl_seconds=None` explicitly to disable checkpoint expiry.

The recipe wraps cookbook `train.main` in one official `capture_to_store`
context for the entire run, with storage under `<log_path>/capture`. Each
rollout group fixes one sampling client and uses distinct group/trajectory
coordinates in the official proxy URL. After the trials finish, the proxy
closes and drains accepted handlers; the recipe flushes the official exporter
and reads the store's public rows API. Those rows are the sole sampled-token
record, including compaction and repeated prompts. Cookbook computes group
advantages and merges only exact token-prefix continuations into Datums.

A completed trial or an agent timeout with a verifier reward is scored using
`harbor.reward_key` (default `reward`). Missing/non-finite rewards and other Harbor
exceptions abort the group. Exporter loss or rows without training logprobs
also stop training. Failed requests remain official capture annotations;
the agent and Harbor determine whether the trial can continue. There is no
synthetic zero reward for infrastructure errors. Task and agent limits belong
to Harbor; rendering, stops, tools and streaming use official proxy behavior.
There is no separate recorder, episode state machine or partial-episode
recovery.

The isolated CPU tests compare the wrapped and bare official proxy and test
Harbor configuration, trial outcomes and token-to-Datum conversion with a
scripted trial. The registered H100 gateway test exercises real sampling and
PPO training. Scripted trials do not validate live sandbox provisioning or
agent binaries; those require an actual Harbor environment.
