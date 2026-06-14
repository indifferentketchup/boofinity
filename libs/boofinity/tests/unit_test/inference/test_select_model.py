import pytest

from boofinity.args import EngineArgs
from boofinity.inference.select_model import select_model
from boofinity.primitives import Device, InferenceEngine


@pytest.mark.parametrize("engine", [e for e in InferenceEngine if e != InferenceEngine.neuron])
def test_engine(engine):
    select_model(
        EngineArgs(
            engine=engine,
            model_name_or_path=(pytest.DEFAULT_BERT_MODEL),
            batch_size=4,
            device=Device.cpu,
            model_warmup=False,
        )
    )
