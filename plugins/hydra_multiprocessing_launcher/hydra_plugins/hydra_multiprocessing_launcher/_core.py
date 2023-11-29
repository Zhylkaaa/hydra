# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
from pathlib import Path
from typing import Any, Dict, Union, List, Sequence, Optional
import copy
from enum import Enum

import cloudpickle

from hydra.core.hydra_config import HydraConfig
from hydra.core.singleton import Singleton
from hydra.core.utils import (
    JobReturn,
    JobStatus,
    configure_log,
    filter_overrides,
    run_job,
    setup_globals,
)
from hydra.plugins.experiment_sequence import ExperimentSequence
from hydra.types import HydraContext, TaskFunction
from omegaconf import OmegaConf, DictConfig, open_dict
import multiprocessing as mp
import multiprocessing.connection

from .multiprocessing_launcher import MultiprocessingLauncher

log = logging.getLogger(__name__)


class WaitingStrategy(Enum):
    FIRST_COMPLETED = 'first_completed'
    ALL_COMPLETED = 'all_completed'


def execute_job(
    idx: int,
    overrides: Sequence[str],
    hydra_context: HydraContext,
    config: DictConfig,
    task_function: TaskFunction,
    singleton_state: Dict[Any, Any],
) -> JobReturn:
    """Calls `run_job` in parallel"""
    setup_globals()
    Singleton.set_state(singleton_state)

    sweep_config = hydra_context.config_loader.load_sweep_config(
        config, list(overrides)
    )
    with open_dict(sweep_config):
        sweep_config.hydra.job.id = "{}_{}".format(sweep_config.hydra.job.name, idx)
        sweep_config.hydra.job.num = idx
    HydraConfig.instance().set_config(sweep_config)

    ret = run_job(
        hydra_context=hydra_context,
        config=sweep_config,
        task_function=task_function,
        job_dir_key="hydra.sweep.dir",
        job_subdir_key="hydra.sweep.subdir",
    )

    return ret


def _proxy_fn_call(lock: mp.Lock(), queue: mp.Queue, fn, *args, **kwargs):
    args = [cloudpickle.loads(obj) for obj in args]
    kwargs = {k: cloudpickle.loads(v) for k, v in kwargs.items()}
    serialized_result = cloudpickle.dumps(fn(*args, **kwargs))
    with lock:
        queue.put((mp.current_process().pid, serialized_result))


def process_multiprocessing_cfg(mp_cfg: Dict[str, Any]) -> None:
    for k in ["timeout", "max_workers"]:
        if k in mp_cfg.keys():
            try:
                val = mp_cfg.get(k)
                if val:
                    mp_cfg[k] = int(val)
            except ValueError:
                pass


def wait_for_results(running_processes, job_lock, result_queue, return_when=WaitingStrategy.ALL_COMPLETED):
    if not running_processes:
        return [], [], []
    done_waiting = False
    total_processes = len(running_processes)
    finished_processes = []
    results = {}

    while not done_waiting:
        mp.connection.wait([p.sentinel for p in running_processes])
        with job_lock:
            finished_processes.extend([p for p in running_processes if not p.is_alive()])
            results.update({pid: serialized_result
                            for pid, serialized_result in iter(result_queue.get_nowait, None)})

            running_processes = [p for p in running_processes if p.is_alive()]
            if return_when is WaitingStrategy.FIRST_COMPLETED or len(finished_processes) == total_processes:
                done_waiting = True

    return [results.get(p.pid) for p in finished_processes], finished_processes, running_processes


def process_results(
        results: List[Optional[JobReturn]],
        overrides_ids: Sequence[Sequence[str]],
        runs: List[Optional[JobReturn]],
        job_overrides: Union[Sequence[Sequence[str]], ExperimentSequence],
        hydra_context: HydraContext,
        config: DictConfig):

    for result, (overrides, idx) in zip(results, overrides_ids):
        # we could only get False value if None was returned from waiting function.
        # which means that process finished without returning, and most likely signals system kill (OOM probably)
        if not result:
            result = JobReturn()
            task_cfg = hydra_context.config_loader.load_sweep_config(
                config, list(overrides)
            )
            result.cfg = task_cfg
            hydra_cfg = copy.deepcopy(HydraConfig.instance().cfg)
            assert isinstance(hydra_cfg, DictConfig)
            result.hydra_cfg = hydra_cfg
            overrides = OmegaConf.to_container(config.hydra.overrides.task)
            assert isinstance(overrides, list)
            result.overrides = overrides
            result.status = JobStatus.FAILED
            result.return_value = RuntimeError('Worker Killed: Worker process exited unexpectedly. '
                                               'May be caused by system OOM kill')

        if isinstance(job_overrides, ExperimentSequence):
            job_overrides.update_sequence((overrides, result))
        runs[idx] = result


def launch(
    launcher: MultiprocessingLauncher,
    job_overrides: Union[Sequence[Sequence[str]], ExperimentSequence],
    initial_job_idx: int,
) -> Sequence[JobReturn]:
    """
    :param job_overrides: an Iterable of List<String>, where each inner list is the arguments for one job run.
    :param initial_job_idx: Initial job idx in batch.
    :return: an array of return values from run_job with indexes corresponding to the input list indexes.
    """
    setup_globals()
    assert launcher.config is not None
    assert launcher.task_function is not None
    assert launcher.hydra_context is not None

    configure_log(launcher.config.hydra.hydra_logging, launcher.config.hydra.verbose)
    sweep_dir = Path(str(launcher.config.hydra.sweep.dir))
    sweep_dir.mkdir(parents=True, exist_ok=True)

    singleton_state = Singleton.get_state()
    batch_size = v if (v := launcher.mp_config['n_jobs']) else mp.cpu_count()

    runs = [None for _ in range(len(job_overrides))]
    log.info(
        "MultiprocessingLauncher({}) is launching {} jobs".format(
            ",".join([f"{k}={v}" for k, v in launcher.mp_config.items()]),
            'generator of' if isinstance(job_overrides, ExperimentSequence) else len(job_overrides),
        )
    )

    job_lock = mp.Lock()
    result_queue = mp.Queue()

    running_tasks = {}

    for idx, override in enumerate(job_overrides):
        log.info("\t#{} : {}".format(idx, " ".join(filter_overrides(override))))

        job_process = mp.Process(
            target=_proxy_fn_call,
            args=(
                job_lock,
                result_queue,
                execute_job,
                *[cloudpickle.dumps(obj) for obj in (
                    initial_job_idx + idx,
                    override,
                    launcher.hydra_context,
                    launcher.config,
                    launcher.task_function,
                    singleton_state
                )]
            )
        )
        job_process.start()
        running_tasks[job_process] = (override, idx)

        if len(running_tasks) == batch_size:
            results, finished, running = wait_for_results(running_tasks,
                                                          job_lock,
                                                          result_queue,
                                                          return_when=WaitingStrategy.FIRST_COMPLETED)
            finished_overrides_ids = [running_tasks[p] for p in finished]
            running_tasks = {p: running_tasks[p] for p in running}

            process_results(results,
                            finished_overrides_ids,
                            runs,
                            job_overrides,
                            launcher.hydra_context,
                            launcher.config)

    results, finished, _ = wait_for_results(running_tasks,
                                            job_lock,
                                            result_queue,
                                            return_when=WaitingStrategy.ALL_COMPLETED)
    finished_overrides_ids = [running_tasks[p] for p in finished]
    process_results(results, finished_overrides_ids, runs, job_overrides, launcher.hydra_context, launcher.config)

    assert isinstance(runs, List)
    for run in runs:
        assert isinstance(run, JobReturn)
    return runs
