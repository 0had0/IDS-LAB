"""This module contains evaluations methods for our models"""
from typing import TypeVar, Callable

import datetime
from torch.profiler import profile, record_function, ProfilerActivity

ModelType = TypeVar("ModelType")


def evalutate_torch_model(
    model: ModelType,
    callback: Callable[[ModelType], None],
    output_file: str = f"data/final/evalutate_torch_model_{datetime.datetime.now()}.txt",
    **kwargs,
) -> float:
    """evalute pyTorch deep learning models"""
    assert model, ValueError("You need a model to evaluate!")
    assert callable, ValueError("What am I supposed to evaluate?")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        **kwargs,
    ) as prof:
        with record_function("model_inference"):
            callback(model)

            f = open(output_file, "w")

            print(
                prof.key_averages(group_by_input_shape=True).table(
                    sort_by="cpu_time_total"
                ),
                file=f,
            )

            print(prof.key_averages().table(sort_by="cuda_time_total"), file=f)

            print(
                prof.key_averages().table(sort_by="self_cpu_memory_usage"),
                file=f,
            )

            print(
                prof.key_averages().table(
                    sort_by="cpu_memory_usage", row_limit=10
                ),
                file=f,
            )

            prof.export_chrome_trace("trace.json")
