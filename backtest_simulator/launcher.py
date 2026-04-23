"""BacktestLauncher — thin subclass hook set for M1 (no full Praxis boot yet)."""
from __future__ import annotations

from dataclasses import dataclass

from backtest_simulator.venue.simulated import SimulatedVenueAdapter


@dataclass
class BacktestLauncher:
    """M1 placeholder: holds the venue + future Praxis injection seams.

    Full Praxis `Launcher` subclassing (with `_start_poller` overridden
    to a no-op and `_run_nexus_instance` hijacked to hand the runner to
    the Driver) lands in the follow-up slice once the full ledger-parity
    test needs it. For M1 the Driver owns the run loop directly — no
    Praxis account-loop / event-spine / poller goes through this path.
    """

    venue: SimulatedVenueAdapter
    account_id: str = 'bts-default'

    def register(self) -> None:
        self.venue.register_account(self.account_id, api_key='sim', api_secret='sim')

    def shutdown(self) -> None:
        self.venue.unregister_account(self.account_id)
