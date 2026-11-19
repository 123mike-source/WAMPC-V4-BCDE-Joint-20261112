"""Passive timing/provenance telemetry for the V7 online contract."""
from evaluation.full_diffusion2_fast_agent import FullDiffusionTiming
from method.full_diffusionv3.diffusion import SpatialTemporalVPFullDiffusion
from method.full_diffusionv7.inference.carla_agent import FullDiffusionV7Agent


def get_entry_point():
    return 'FullDiffusionV7FastAgent'


class FullDiffusionV7FastAgent(FullDiffusionTiming, FullDiffusionV7Agent):
    method = 'full_diffusionv7'
    sampler_type = SpatialTemporalVPFullDiffusion
    log_prefix = 'FULL_DIFFUSIONV7'
    telemetry_interval_steps = 1

    def extra_telemetry(self):
        net = self.nets[0]
        return {'architecture': net.run_config['architecture'], 'controller': net.run_config['controller'],
                'spatial_points': 10, 'temporal_points': 6, 'planning_hz': 20, 'control_hz': 20,
                'online_contract': net.load_report['online_contract']['version'],
                'simulation_tick': net.last_tick, 'phase_mod5': net.last_tick % 5 if net.last_tick is not None else None,
                'observation_ticks': net.last_history_ticks,
                'prior_source_tick': net.last_prior_tick,
                'future_generation_tick': net.last_future_tick,
                'future_disabled_reason': net.last_future_disabled_reason,
                'missing_simulation_ticks': net.last_tick_gap,
                'planning_update': net.last_planning_update, 'planning_updates': net.planning_updates,
                'control_updates': net.control_updates,
                'control_plan_age_seconds': net.clock.age(net._timestamp) if net.clock.anchor_tick is not None else None,
                'planning_timestamp': net.clock.anchor_tick * .05 if net.clock.anchor_tick is not None else None,
                'world_active': net.last_world_active, 'prior_age_seconds': net.last_prior_age,
                'prior_valid': net.last_prior_valid, 'world_cache_age_at_planning_seconds': 0.,
                'last_planning_timing': net.last_timings,
                'temporal_delta': net.last_temporal_delta[0].tolist() if net.last_temporal_delta is not None else None,
                'world_checkpoint_sha256': net.world.provenance['sha256'], 'world_frozen': True,
                'selected_epoch': net.load_report['progress']['epoch'], 'speed_limit_kmh': 35,
                'controller_profile': 'legacy', 'checkpoint_format': net.load_report['format'],
                'checkpoint_updates': net.load_report['progress']['updates']}
