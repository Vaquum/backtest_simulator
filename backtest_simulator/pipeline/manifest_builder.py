"""ManifestBuilder — write Nexus manifest YAML + strategy file for a filtered sensor set."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
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

    symbol: str
    capital: Decimal
    kelly_pct: Decimal
    estimated_price: Decimal
    stop_bps: Decimal
    force_flatten_after: datetime | None = None
    maker_preference: bool = False

@dataclass(frozen=True)
class AccountSpec:

    account_id: str
    allocated_capital: Decimal
    capital_pool: Decimal

@dataclass(frozen=True)
class SensorBinding:

    experiment_dir: Path
    permutation_ids: tuple[int, ...]
    interval_seconds: int

@dataclass(frozen=True)
class BuiltManifest:

    manifest_path: Path
    strategies_base_path: Path
    manifest: Manifest
    strategy_params: dict[str, object] = field(
        default_factory=lambda: dict[str, object](),
    )

class ManifestBuilder:

    DEFAULT_TEMPLATE: str = 'long_on_signal.py'

    def __init__(self, output_dir: Path, template_name: str = DEFAULT_TEMPLATE) -> None:
        self._output_dir = Path(output_dir).resolve()
        template = _TEMPLATES_DIR / template_name
        if not template.is_file():
            msg = f'strategy template not found: {template}'
            raise FileNotFoundError(msg)
        self._template_path = template

    def build(
        self,
        *,
        account: AccountSpec,
        strategy_id: str,
        sensor: SensorBinding,
        strategy_params: StrategyParamsSpec,
        capital_pct: Decimal = Decimal('100'),
    ) -> BuiltManifest:
        self._output_dir.mkdir(parents=True, exist_ok=True)

        ffa = strategy_params.force_flatten_after
        if ffa is not None and ffa.utcoffset() is None:
            msg = (
                f'StrategyParamsSpec.force_flatten_after must be '
                f'tz-aware, got effectively-naive datetime {ffa!r} '
                f'(tzinfo={ffa.tzinfo!r} but utcoffset()=None). The '
                f'strategy compares this against UTC-aware signal '
                f'timestamps; a naive cutoff raises TypeError.'
            )
            raise ValueError(msg)
        raw_params: dict[str, object] = {
            'symbol': strategy_params.symbol,
            'capital': str(strategy_params.capital),
            'kelly_pct': str(strategy_params.kelly_pct),
            'estimated_price': str(strategy_params.estimated_price),
            'stop_bps': str(strategy_params.stop_bps),
            'maker_preference': bool(strategy_params.maker_preference),
            'force_flatten_after': (
                None if ffa is None else ffa.isoformat()
            ),
        }
        template_source = self._template_path.read_text(encoding='utf-8')
        strategy_file = self._output_dir / self._template_path.name
        strategy_file.write_text(
            template_source.replace('__BTS_PARAMS__', json.dumps(raw_params)),
            encoding='utf-8',
        )

        manifest = _assemble_manifest(
            account=account, strategy_id=strategy_id,
            sensor=sensor, capital_pct=capital_pct,
            strategy_file=strategy_file.name,
        )

        yaml_payload = _manifest_to_yaml(manifest, strategy_params=raw_params)
        manifest_path = self._output_dir / 'manifest.yaml'
        manifest_path.write_text(yaml_payload, encoding='utf-8')

        validated = load_manifest(manifest_path)

        _log.info(
            'manifest built',
            extra={'path': str(manifest_path), 'strategy_id': strategy_id},
        )
        return BuiltManifest(
            manifest_path=manifest_path, strategies_base_path=self._output_dir,
            manifest=validated, strategy_params=raw_params,
        )

def _assemble_manifest(
    *,
    account: AccountSpec,
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
        account_id=account.account_id,
        allocated_capital=account.allocated_capital,
        capital_pool=account.capital_pool,
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
