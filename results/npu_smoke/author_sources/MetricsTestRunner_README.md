# MetricsTestRunner

MetricsTestRunner is a comprehensive tool for running distributed LLM (Large Language Model) inference experiments. It provides a flexible configuration system using TOML files to define and execute complex distributed inference scenarios.

## Overview

The MetricsTestRunner allows you to:
- Launch multiple distributed LLM inference components (vLLM backends, routers, clients)
- Configure complex experimental setups using TOML configuration files
- Run experiments with different policies and configurations
- Collect and analyze performance metrics
- Generate visualizations of experiment results

## Usage

### Single-Node
For colocation test 4 cards. This will start a new tmux session, one window for vLLM backends, one for router, and one for clients.
It will dump all the logs in the OUTPUT_BASE specified in `azure_ali_h20_17.sh`.

```bash
./azure_ali_h20_17.sh ../../config/ali-h20-17/launch_vllm_dp4.toml ../../config/ali-h20-17/vllm_router.toml ../../config/ali-h20-17/new_clients.toml round-robin-q
```

<!-- TODO, blitz head support PD-scheduling -->
<!-- For PD-disaggregation test, you should change the vLLM configs.

```bash
./azure_ali_h20_17.sh ../../config/ali-h20-17/launch_vllm_2p6d_1node.toml ../../config/ali-h20-17/vllm_router.toml ../../config/ali-h20-17/new_clients.toml round-robin-q

``` -->

## Directory Structure

```
MetricsTestRunner/
├── smart_runner.py          # Main execution script
├── README.md                # This file
├── config/                  # Configuration files for different experiments
│   ├── *.toml              # TOML configuration files
│   └── ali-h20-17/         # Hardware-specific configurations
└── scripts/                 # Execution scripts for specific experiments
    ├── colocation/
    ├── multi_node/
    └── pd_disaggregation/
```

## Components

### smart_runner.py

The main execution script that processes TOML configuration files and launches experiments. It supports:

- **Template inheritance**: Define base templates and inherit/override properties
- **Variable substitution**: Use `${variable}` syntax for dynamic configuration
- **Multiple runtimes**: Support for raw, SSH, and MPI execution environments
- **Background process management**: Automatically manages background processes and cleanup

### Configuration Files

Configuration files use TOML format and define:

1. **Runtime configuration**: How to execute the experiment (raw, ssh, mpi)
2. **Variables**: Global variables for consistent configuration
3. **Application templates**: Reusable templates for different components
4. **Specific instances**: Concrete instances of applications with their configurations

Example configuration structure:
```toml
[runtime.raw]
config = [
    { app = "vllm_template", background = true, port_offset = 0 },
    { app = "client_conv_1", background = false }
]

[variables]
model_path = "/path/to/model"
tokenizer_path = "/path/to/tokenizer"

[app.vllm_template]
executable = ["${vllm_executable}"]
extra_args = ["${model_path}", "--port", "${port}"]

[app.vllm_template.macro_rules]
"--port" = "port_offset"

[app.client_conv_1]
inherit = "client_template"
config.tokenizer = "${tokenizer_path}"
config.endpoint = "http://localhost:58009/generate"
```

## Usage

### Basic Execution

To run an experiment with a specific configuration:

```bash
python smart_runner.py --toml config/dense_clients.toml
```

### Advanced Execution with Variables

Pass additional variables to override configuration values:

```bash
python smart_runner.py --toml config/dense_clients.toml --model-path=/custom/path --output-dir=/results
```

### Running Complete Experiments

Use provided scripts for complete experiment workflows:

```bash
# Run Azure evaluation with specific policy
./scripts/colocation/eval_azure_ali_h20_17.sh \
  config/ali-h20-17/launch_vllm_dp6.toml \
  config/ali-h20-17/vllm_router.toml \
  config/ali-h20-17/azure_qwen3_8b_client.toml \
  round-robin-q
```

## Configuration Details

### Runtime Configuration

Define how applications should be executed:

- `runtime.raw`: Direct execution on local machine
- `runtime.ssh`: Remote execution via SSH
- `runtime.mpi`: MPI-based distributed execution

### Application Templates

Define reusable application templates with:
- `executable`: The command to execute
- `extra_args`: Additional command line arguments
- `macro_rules`: Mapping of CLI arguments to configuration keys

### Variable Substitution

Use `${variable_name}` syntax in configuration files. Variables can be defined in:
- `[variables]` section of the TOML file
- Command line arguments to smart_runner.py

### Inheritance

Applications can inherit from templates using the `inherit` property, allowing for:
- Base configurations with common settings
- Specialized instances with overridden values
- Hierarchical configuration organization

## Experiment Scripts

### colocation/
Scripts for running colocation experiments on single machines.

### multi_node/
Scripts for running distributed experiments across multiple nodes. Colocation and PD-disaggregation scripts are all included.

### pd_disaggregation/
Scripts for running prefill-decode-disaggregation experiments.

## Common Workflows

### 1. Local Development Testing

```bash
python smart_runner.py --toml config/dense_clients.toml --output-dir=/tmp/test
```

### 2. Multi-node Deployment

```bash
# On master node
python smart_runner.py --toml config/distributed_setup.toml
```

### 3. Experiment Analysis

After running experiments, use the generated logs and metrics for analysis:
- JSONL files with client request data
- Router logs with system performance metrics
- Generated figures and visualizations

## Requirements

- Python 3.8+
- TOML parsing library (tomli)
- vLLM for LLM serving
- Appropriate LLM models
- Required dependencies for specific runtimes (SSH, MPI)

## Best Practices

1. **Organize configurations**: Use descriptive names and folder structure for different experiment types
2. **Use variables**: Define common paths and values as variables for easy modification
3. **Template inheritance**: Create base templates to reduce configuration duplication
4. **Version control**: Keep configurations in version control to track experiment changes
5. **Documentation**: Comment complex configurations to explain their purpose

## Troubleshooting

### Common Issues

1. **Port conflicts**: Ensure ports defined in configurations are available
2. **Path issues**: Verify all paths in configurations exist and are accessible
3. **Permission errors**: Check that the user has appropriate permissions for all paths
4. **Background processes**: Use process management tools to monitor and clean up processes

### Process Management

The smart_runner automatically manages background processes, but you can manually clean up with:
```bash
pkill -f vllm
pkill -f router_v2
pkill -f smart_runner.py
```

## Extending MetricsTestRunner

### Adding New Runtimes

Implement new runtime functions and register them with the `@register` decorator:
```python
@register("custom_runtime")
def gen_custom_cmd(app_cmd: str, rt_config: dict, app_running_config: dict) -> str:
    # Implementation here
    return full_cmd
```

### Adding New Application Types

Define new application templates in configuration files:
```toml
[app.new_app_type]
executable = ["new_executable"]
extra_args = ["--arg1", "value1"]
```

### csv
在MetricTestRunner仓库h20_config分支，model_csv目录下面有所有算子的表
1. 前缀为attn_prefill attn_decode为prefill和decode的预测，feature较为不同
2. 其余的都以token_num(这一批里面token数，decode=1，prefill=chunk_size)为feature
3. 前几列是feature，后一列是latency，单位为毫秒
4. 除samplre和schedule之外的kernel，都是单层的时间，对于整个model_forward的贡献需要乘以num_layer (28 in qwen2.5)