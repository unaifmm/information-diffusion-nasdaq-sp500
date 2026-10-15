"""VECM price-discovery measures, on simulated prices of the same asset."""
import numpy as np

from src.data import log_returns
from src.leadlag import net_lead
from src.pricediscovery import vecm_price_discovery
from src.simulate import simulate_same_asset_pair


def test_strong_lead_is_found():
    """ILS and the net lead both see a strong lead. (The component share CS does
    not: tiny bid-ask-like noise in A already flips it - one reason we do not
    rely on the textbook measures at 1-minute bars.)"""
    logp = simulate_same_asset_pair(w=0.5, kappa=0.5, noise_a=2e-5, noise_b=2e-5)
    pd_ = vecm_price_discovery(logp, lags=5)
    assert pd_["ILS_A"] > 0.9
    assert net_lead(log_returns(np.exp(logp), max_gap="1min"), "A", "B")["net_lead_A_pp"] > 30


def test_information_share_bounds_are_valid():
    logp = simulate_same_asset_pair(w=0.9, kappa=0.5, noise_a=2e-5, noise_b=2e-5)
    pd_ = vecm_price_discovery(logp, lags=5)
    assert 0 <= pd_["IS_low_A"] <= pd_["IS_mid_A"] <= pd_["IS_high_A"] <= 1


def test_net_lead_is_not_fooled_by_noise():
    """No lead, but A is 5x noisier: the textbook ILS wrongly says A leads
    (see 02_validate_simulation.py); the net lead stays near zero."""
    logp = simulate_same_asset_pair(w=1.0, kappa=0.5, noise_a=1e-4, noise_b=2e-5)
    nl = net_lead(log_returns(np.exp(logp), max_gap="1min"), "A", "B")["net_lead_A_pp"]
    assert abs(nl) < 2.0
