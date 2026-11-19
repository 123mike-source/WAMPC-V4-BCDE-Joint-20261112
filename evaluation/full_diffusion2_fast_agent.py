"""Timing and optional metric buffering around the configured planner."""
from contextlib import nullcontext
import json
import os
from pathlib import Path
import resource
import time
from unittest.mock import patch

from method.full_diffusion2.inference.carla_agent import FullDiffusion2GeoAgent
from method.full_diffusion.diffusion import VPFullDiffusion


def get_entry_point():
    return 'FullDiffusion2FastAgent'


class FullDiffusionTiming:
    method = 'full_diffusion2'
    sampler_type = VPFullDiffusion
    log_prefix = 'FULL_DIFFUSION2'
    telemetry_interval_steps = 20

    def telemetry_due(self):
        return self.step < 3 or self.step % self.telemetry_interval_steps == 0

    def event(self, event, **values):
        with (self.directory/'route_events.jsonl').open('a') as handle:
            handle.write(json.dumps({'event': event, 'route': self.route_name,
                                     'unix': time.time(), **values})+'\n')

    def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
        parts = path_to_conf_file.split('+')
        self.route_name = route_index or (parts[-1] if len(parts) > 1 else None)
        self.directory = Path(os.environ['WMMPC_CARLA_LOG_DIR'])
        self._network_seconds = None
        self._last_path = None
        self._metric_journal = None
        self._buffered_run_step = None
        self.event('agent_setup_started')
        super().setup(path_to_conf_file, self.route_name, traffic_manager)
        inference_steps = self.nets[0].run_config['diffusion']['inference_steps']
        expected_steps = int(os.environ.get('WMMPC_DIFFUSION_INFERENCE_STEPS', '100'))
        if inference_steps != expected_steps or expected_steps <= 0:
            raise ValueError(f'Expected {expected_steps} diffusion steps; got {inference_steps}')
        if self.nets[0].run_config.get('method') != self.method:
            raise ValueError('The checkpoint method differs from the evaluation agent')
        if type(self.nets[0].diffusion) is not self.sampler_type:
            raise TypeError(f'The original {self.sampler_type.__name__} sampler is required')
        if os.environ.get('WMMPC_BUFFER_METRICS') == '1':
            import sensor_agent
            from evaluation.fast_carla_io import compile_buffered_run_step
            self._buffered_run_step = compile_buffered_run_step(sensor_agent)
        forward = self.nets[0].forward

        def timed_forward(*args, **kwargs):
            start = time.perf_counter()
            result = forward(*args, **kwargs)
            self._network_seconds = time.perf_counter()-start
            if self.telemetry_due():
                self._last_path = result[2][0].detach().float().cpu().tolist()
            return result

        self.nets[0].forward = timed_forward
        self.event('agent_setup_ready', inference_steps=inference_steps, sampler=self.sampler_type.__name__,
                   sampler_replaced=False, method=self.method,
                   checkpoint_inference_steps=getattr(self.nets[0], 'checkpoint_inference_steps', inference_steps),
                   buffered_metrics=self._buffered_run_step is not None)
        print(f'{self.log_prefix}_READY route={self.route_name}', flush=True)

    def extra_telemetry(self):
        return {}

    def _store_metric(self, metric):
        if self._metric_journal is None:
            import ujson
            from evaluation.fast_carla_io import MetricJournal
            self._metric_journal = MetricJournal(self.save_path, ujson)
        self._metric_journal.append(self.step, metric)

    def run_step(self, input_data, timestamp, sensors=None):
        started = time.perf_counter()
        cpu_start = resource.getrusage(resource.RUSAGE_SELF)
        context = nullcontext()
        if self._buffered_run_step is not None:
            import sensor_agent
            context = patch.object(sensor_agent.SensorAgent, 'run_step', self._buffered_run_step)
        with context:
            control = super().run_step(input_data, timestamp, sensors)
        if self.telemetry_due():
            cpu_end = resource.getrusage(resource.RUSAGE_SELF)
            history = self.nets[0].history
            record = {'unix': time.time(), 'route': self.route_name, 'step': self.step,
                      'sim_seconds': float(timestamp), 'run_step_seconds': time.perf_counter()-started,
                      'network_seconds': self._network_seconds,
                      'process_cpu_seconds': cpu_end.ru_utime+cpu_end.ru_stime-cpu_start.ru_utime-cpu_start.ru_stime,
                      'observed_speed_mps': float(input_data['speed'][1]['speed']),
                      'model_target_speed_mps': self.nets[0].last_target_speed,
                      'control': {'steer': float(control.steer), 'throttle': float(control.throttle),
                                  'brake': float(control.brake)},
                      'history_entries': len(history.entries), 'path_xy': self._last_path,
                      'filtered_state': self.state_log[-1].tolist() if self.state_log else None}
            record.update(self.extra_telemetry())
            with (self.directory/'telemetry.jsonl').open('a') as handle:
                handle.write(json.dumps(record)+'\n')
            print(self.log_prefix+'_TICK '+json.dumps({k:v for k,v in record.items()
                  if k not in ('path_xy', 'filtered_state')}), flush=True)
        return control

    def destroy(self, results=None):
        if self._metric_journal is not None:
            self._metric_journal.finalize(self.metric_info)
        super().destroy(results)
        self.event('agent_destroy_finished')


class FullDiffusion2FastAgent(FullDiffusionTiming, FullDiffusion2GeoAgent):
    pass
