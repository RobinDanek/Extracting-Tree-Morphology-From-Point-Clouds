import yaml
from Modules.Pipeline.Pipeline import run_pipeline

if __name__ == "__main__":

    with open("PipelineExecution/pipeline_config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    run_pipeline( 
        cfg=cfg
        )