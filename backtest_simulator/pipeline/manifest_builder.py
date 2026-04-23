"""ManifestBuilder — write Nexus manifest YAML + strategy file for a filtered sensor set."""
from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

import yaml
from nexus.infrastructure.manifest import (
    Manifest,
    SensorSpec,
    StrategySpec,
    TimerSpec,
    load_manifest,
)

_log = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).parent / '_strategy_templates'


@dataclass(frozen=True)
class StrategyParamsSpec:
    """Runtime parameters the generated strategy will read from StrategyParams.raw."""

    symbol: str
    enter_threshold: float
    stop_bps: Decimal
    qty: Decimal
    side: str = 'BUY'
    prob_key: str = 'probability'


@dataclass(frozen=True)
class SensorBinding:
    """Wire one Limen experiment_dir + permutation_ids to one strategy."""

    experiment_dir: Path
    permutation_ids: tuple[int, ...]
    interval_seconds: int


@dataclass(frozen=True)
class BuiltManifest:
    """Paths emitted by ManifestBuilder — everything BacktestLauncher needs."""

    manifest_path: Path
    strategies_base_path: Path
    manifest: Manifest
    strategy_params: dict[str, object] = field(default_factory=dict)


class ManifestBuilder:
    """Produce a Nexus manifest + strategy file from an ExperimentPipeline result."""

    DEFAULT_TEMPLATE: str = 'long_on_signal.py'

    def __init__(self, output_dir: Path, template_name: str = DEFAULT_TEMPLATE) -> None:
        self._output_dir = Path(output_dir).resolve()
        template = _TEMPLATES_DIR / template_name
        if not template.is_file():
            msg = f'strategy template not found: {template}'
            raise FileNotFoundError(msg)
        self._template_path = template

    def build(  # noqa: PLR0913 - manifest fields are all required for validation
        self,
        *,
        account_id: str,
        allocated_capital: Decimal,
        capital_pool: Decimal,
        strategy_id: str,
        sensor: SensorBinding,
        strategy_params: StrategyParamsSpec,
        capital_pct: Decimal = Decimal('100'),
    ) -> BuiltManifest:
        """Write manifest.yaml + <template>.py side-by-side; return the paths.

        Nexus's `load_manifest` resolves `StrategySpec.file` relative to the
        manifest's parent directory. Keeping manifest.yaml and the strategy
        .py in the same directory means `file: long_on_signal.py` is the
        literal filename and the loader finds it without further path-joining.
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)

        strategy_file = self._output_dir / self._template_path.name
        shutil.copyfile(self._template_path, strategy_file)

        manifest = _assemble_manifest(
            account_id=account_id, allocated_capital=allocated_capital,
            capital_pool=capital_pool, strategy_id=strategy_id,
            sensor=sensor, capital_pct=capital_pct,
            strategy_file=strategy_file.name,
        )

        raw_params = {
            'symbol': strategy_params.symbol,
            'side': strategy_params.side,
            'enter_threshold': strategy_params.enter_threshold,
            'stop_bps': str(strategy_params.stop_bps),
            'qty': str(strategy_params.qty),
            'prob_key': strategy_params.prob_key,
        }

        yaml_payload = _manifest_to_yaml(manifest, strategy_params=raw_params)
        manifest_path = self._output_dir / 'manifest.yaml'
        manifest_path.write_text(yaml_payload, encoding='utf-8')

        # Round-trip through Nexus's own loader so we fail fast on any
        # schema drift. Without this, invalid YAML would only surface at
        # BacktestLauncher boot time deep inside Nexus threading code.
        validated = load_manifest(manifest_path)

        _log.info(
            'manifest built',
            extra={'path': str(manifest_path), 'strategy_id': strategy_id},
        )
        return BuiltManifest(
            manifest_path=manifest_path, strategies_base_path=self._output_dir,
            manifest=validated, strategy_params=raw_params,
        )


def _assemble_manifest(  # noqa: PLR0913 - manifest-schema arg count
    *,
    account_id: str,
    allocated_capital: Decimal,
    capital_pool: Decimal,
    strategy_id: str,
    sensor: SensorBinding,
    capital_pct: Decimal,
    strategy_file: str,
    timers: tuple[TimerSpec, ...] = (),
) -> Manifest:
    if not sensor.experiment_dir.is_dir():
        msg = f'SensorBinding.experiment_dir not found: {sensor.experiment_dir}'
        raise ValueError(msg)
    sensor_spec = SensorSpec(
        experiment_dir=sensor.experiment_dir,
        permutation_ids=sensor.permutation_ids,
        interval_seconds=sensor.interval_seconds,
    )
    strategy = StrategySpec(
        strategy_id=strategy_id, file=strategy_file,
        sensors=(sensor_spec,), capital_pct=capital_pct,
        timers=timers,
    )
    return Manifest(
        account_id=account_id,
        allocated_capital=allocated_capital,
        capital_pool=capital_pool,
        strategies=(strategy,),
    )


def _manifest_to_yaml(manifest: Manifest, strategy_params: dict[str, object]) -> str:
    data = {
        'account_id': manifest.account_id,
        'allocated_capital': str(manifest.allocated_capital),
        'capital_pool': str(manifest.capital_pool),
        'strategies': [
            {
                'id': s.strategy_id,
                'file': s.file,
                'capital_pct': str(s.capital_pct),
                'sensors': [
                    {
                        'experiment': str(p.experiment_dir),
                        'permutation_ids': list(p.permutation_ids),
                        'interval_seconds': p.interval_seconds,
                    }
                    for p in s.sensors
                ],
                'timers': [
                    {'timer_id': t.timer_id, 'interval_seconds': t.interval_seconds}
                    for t in s.timers
                ],
                'params': strategy_params,
            }
            for s in manifest.strategies
        ],
    }
    return yaml.safe_dump(data, sort_keys=False)
