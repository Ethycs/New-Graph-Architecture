"""Unit tests for ``nga.arch.energy_minimization_trainer``.

Verifies the quality signal's [0, 1] range and monotonicities, that a single
``step`` performs a Riemannian-SGD descent on a convex energy field, that the
posterior update tracks the quality signal, and that orbit sharing is
properly a no-op without a group action and Riemannian-mean-shaped with one.
"""
from __future__ import annotations

import numpy as np
import pytest

from nga.arch.energy_function import EnergyFunction, EnergyParameters
from nga.arch.energy_minimization_trainer import (
    EnergyMinimizationTrainer,
    TrainerConfig,
    TrainerStepResult,
)
from nga.arch.graph_fsm import GraphFSM
from nga.arch.group_action_on_graph import GroupAction
from nga.arch.hyperbolic_embedding import (
    exp_map_zero,
    log_map_zero,
    poincare_distance,
    project,
)
from nga.arch.posterior_mask import PosteriorMask
from nga.drivers.graph_fsm_spec import (
    Coordinates,
    Edge,
    GraphFSMSpec,
    Vertex,
)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _make_two_vertex_fsm() -> GraphFSM:
    """Bidirectional 2-vertex graph: s0 <-> s1."""
    vertices = [Vertex(id=f"s{i}", label=f"S{i}") for i in range(2)]
    edges = [
        Edge(source="s0", target="s1"),
        Edge(source="s1", target="s0"),
        Edge(source="s0", target="s0"),
        Edge(source="s1", target="s1"),
    ]
    spec = GraphFSMSpec(
        name="two",
        vertex_count=2,
        edge_count=4,
        vertices=vertices,
        edges=edges,
        coordinates=Coordinates(space="euclidean", dimension=1),
    )
    return GraphFSM(spec)


def _make_trainer(
    *,
    n_vertices: int = 2,
    dim: int = 2,
    proto_init: np.ndarray | None = None,
    config: TrainerConfig | None = None,
    group_action: GroupAction | None = None,
) -> EnergyMinimizationTrainer:
    """Construct a trainer with one type "T" and a vertex-aligned prototype table."""
    if proto_init is None:
        proto_init = np.zeros((n_vertices, dim), dtype=np.float64)
    prototypes = {"T": project(np.asarray(proto_init, dtype=np.float64))}
    posterior = PosteriorMask(n_vertices)
    energy_fn = EnergyFunction(EnergyParameters())
    return EnergyMinimizationTrainer(
        prototypes=prototypes,
        posterior_mask=posterior,
        energy_fn=energy_fn,
        group_action=group_action,
        config=config,
    )


# ---------------------------------------------------------------------------
# Quality signal tests
# ---------------------------------------------------------------------------


def test_quality_signal_in_unit_interval() -> None:
    """Q must lie in [0, 1] for any reasonable input combination."""
    trainer = _make_trainer()
    rng = np.random.default_rng(0)
    for _ in range(50):
        e = float(rng.uniform(-2.0, 5.0))
        s = float(rng.uniform(0.0, 1.0))
        c = float(rng.choice([0.0, 1.0]))
        l = float(rng.uniform(0.0, 1.0))
        q = trainer.quality_signal(e, s, c, l)
        assert 0.0 <= q <= 1.0


def test_quality_signal_monotone_in_energy() -> None:
    """Higher energy strictly lowers Q (other inputs held fixed)."""
    trainer = _make_trainer()
    sigma, contradiction, loop = 0.2, 0.0, 0.1
    energies = [-1.0, 0.0, 0.5, 1.0, 2.0, 5.0]
    qs = [trainer.quality_signal(e, sigma, contradiction, loop) for e in energies]
    for a, b in zip(qs, qs[1:]):
        assert a > b


def test_quality_signal_monotone_in_sigma() -> None:
    """Higher sigma strictly lowers Q (other inputs held fixed)."""
    trainer = _make_trainer()
    energy, contradiction, loop = 0.0, 0.0, 0.1
    sigmas = [0.0, 0.1, 0.3, 0.5, 0.8, 1.0]
    qs = [trainer.quality_signal(energy, s, contradiction, loop) for s in sigmas]
    for a, b in zip(qs, qs[1:]):
        assert a > b


# ---------------------------------------------------------------------------
# Step result type
# ---------------------------------------------------------------------------


def test_step_returns_trainer_step_result() -> None:
    """A trainer step returns a TrainerStepResult with all named fields populated."""
    trainer = _make_trainer(
        proto_init=np.array([[0.3, 0.0], [0.0, 0.3]]),
    )
    obs = np.array([0.32, 0.0])  # close to prototype 0
    result = trainer.step(
        src_state=0,
        dst_state=1,
        observation_in_poincare=obs,
        observation_type="T",
        prototype_idx=0,
        sigma=0.1,
        contradiction=0.0,
        loop_risk=0.05,
        cost=0.0,
        progress=1.0,
    )
    assert isinstance(result, TrainerStepResult)
    assert hasattr(result, "energy_observed")
    assert hasattr(result, "energy_after_update")
    assert hasattr(result, "quality")
    assert hasattr(result, "posterior_alpha_beta_delta")
    assert hasattr(result, "prototype_shift_magnitude")
    assert hasattr(result, "monodromy_loss")
    a, b = result.posterior_alpha_beta_delta
    assert pytest.approx(a + b, abs=1e-9) == 1.0


# ---------------------------------------------------------------------------
# Convex-field descent
# ---------------------------------------------------------------------------


def test_step_decreases_energy_on_convex_field() -> None:
    """50 steps from a fixed observation should drive the rolling-mean energy down.

    Setup: a single observation point at p_obs in the Poincare ball with a
    prototype starting some distance away. Each ``step`` should pull the
    prototype closer to p_obs, which strictly lowers the uncertainty channel
    and therefore the total energy.
    """
    rng = np.random.default_rng(42)
    p_obs = np.array([0.4, 0.1])
    proto_init = np.array([[-0.2, -0.3], [0.0, 0.0]])
    trainer = _make_trainer(
        n_vertices=2,
        dim=2,
        proto_init=proto_init,
        config=TrainerConfig(lr_prototype=0.5),
    )

    energies: list[float] = []
    for _ in range(50):
        obs = p_obs + rng.normal(scale=0.005, size=2)
        obs = project(obs)
        result = trainer.step(
            src_state=0,
            dst_state=1,
            observation_in_poincare=obs,
            observation_type="T",
            prototype_idx=0,
            sigma=0.0,
            contradiction=0.0,
            loop_risk=0.0,
            cost=0.0,
            progress=1.0,
        )
        energies.append(result.energy_observed)

    # The very first energy should exceed the very last (modulo small noise).
    assert energies[0] > energies[-1] + 1e-3

    # Rolling-mean descent: average over a window of 10 should be
    # weakly monotone non-increasing across windows (with a small tolerance
    # to absorb the Gaussian observation noise -- once the prototype has
    # essentially converged onto the observation cloud, fluctuations within
    # the noise envelope can swap consecutive window means).
    window = 10
    rolling = [
        float(np.mean(energies[i : i + window]))
        for i in range(0, len(energies) - window + 1, window)
    ]
    # Tolerance scales with the observation-noise std (0.005) projected
    # through the energy field's beta_uncertainty=1 weight; 5e-3 is generous.
    for prev, curr in zip(rolling, rolling[1:]):
        assert curr <= prev + 5e-3


# ---------------------------------------------------------------------------
# Posterior update tracks Q
# ---------------------------------------------------------------------------


def test_posterior_update_increments_alpha_for_high_quality() -> None:
    """High-quality steps push alpha up; low-quality steps push beta up."""
    rng = np.random.default_rng(7)
    p_obs = np.array([0.3, 0.0])

    # High-quality: prototype right on top of observation, no signals -> Q close to 1.
    trainer = _make_trainer(
        proto_init=np.array([[0.3, 0.0], [0.0, 0.3]]),
    )
    a_before = trainer.posterior_mask.alpha[0, 1]
    b_before = trainer.posterior_mask.beta[0, 1]
    result = trainer.step(
        src_state=0,
        dst_state=1,
        observation_in_poincare=p_obs,
        observation_type="T",
        prototype_idx=0,
        sigma=0.0,
        contradiction=0.0,
        loop_risk=0.0,
    )
    a_after = trainer.posterior_mask.alpha[0, 1]
    b_after = trainer.posterior_mask.beta[0, 1]
    assert result.quality > 0.5
    assert a_after - a_before == pytest.approx(result.quality)
    assert b_after - b_before == pytest.approx(1.0 - result.quality)
    # And the alpha grew more than beta.
    assert (a_after - a_before) > (b_after - b_before)

    # Low-quality: contradiction flag on, distant prototype -> Q close to 0.
    trainer2 = _make_trainer(
        proto_init=np.array([[-0.4, -0.4], [0.0, 0.3]]),
    )
    a_before = trainer2.posterior_mask.alpha[0, 1]
    b_before = trainer2.posterior_mask.beta[0, 1]
    result2 = trainer2.step(
        src_state=0,
        dst_state=1,
        observation_in_poincare=p_obs,
        observation_type="T",
        prototype_idx=0,
        sigma=0.9,
        contradiction=1.0,
        loop_risk=0.9,
    )
    a_after = trainer2.posterior_mask.alpha[0, 1]
    b_after = trainer2.posterior_mask.beta[0, 1]
    assert result2.quality < 0.5
    assert (b_after - b_before) > (a_after - a_before)


# ---------------------------------------------------------------------------
# Orbit sharing
# ---------------------------------------------------------------------------


def test_orbit_sharing_no_op_without_group() -> None:
    """No GroupAction => enforce_orbit_sharing leaves prototypes unchanged."""
    proto_init = np.array([[0.1, 0.0], [-0.2, 0.3]])
    trainer = _make_trainer(proto_init=proto_init)
    snapshot = trainer.prototypes["T"].copy()
    trainer.enforce_orbit_sharing()
    assert np.allclose(trainer.prototypes["T"], snapshot)


def test_orbit_sharing_averages_within_orbit() -> None:
    """A 2-element orbit drives both prototypes to their Riemannian midpoint."""
    fsm = _make_two_vertex_fsm()
    # Z/2 swap: 0 <-> 1.
    identity = np.array([0, 1], dtype=int)
    swap = np.array([1, 0], dtype=int)
    group = GroupAction(fsm=fsm, permutations=[identity, swap], name="Z/2")

    p0 = np.array([0.4, 0.0])
    p1 = np.array([-0.4, 0.0])
    proto_init = np.stack([p0, p1])
    trainer = _make_trainer(
        proto_init=proto_init,
        group_action=group,
        config=TrainerConfig(enforce_orbit_sharing=True),
    )
    trainer.enforce_orbit_sharing()
    after = trainer.prototypes["T"]

    # After sharing, both rows should equal the tangent-space midpoint.
    expected_tangent = 0.5 * (log_map_zero(p0) + log_map_zero(p1))
    expected = exp_map_zero(expected_tangent)
    assert np.allclose(after[0], expected, atol=1e-9)
    assert np.allclose(after[1], expected, atol=1e-9)
    # Symmetric inputs around origin -> midpoint is the origin.
    assert np.allclose(after[0], np.zeros(2), atol=1e-9)


# ---------------------------------------------------------------------------
# Closed-walk loss
# ---------------------------------------------------------------------------


def test_closed_walk_loss_zero_for_consistent_walks() -> None:
    """When energy returns to baseline, the monodromy loss is exactly zero."""
    trainer = _make_trainer()
    walks = [[0, 1, 0], [0, 1, 0, 1, 0]]
    energies = [
        np.array([0.5, 1.2, 0.5]),
        np.array([0.7, 0.7, 0.7, 0.7, 0.7]),
    ]
    loss = trainer.closed_walk_loss(walks, energies)
    assert loss == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Phase 7 -- KL-based quality signal
# ---------------------------------------------------------------------------


def test_quality_signal_kl_zero_when_distributions_match() -> None:
    """Identical empirical and predicted distributions => KL = 0 => Q = 1."""
    trainer = _make_trainer()
    p = np.array([0.2, 0.5, 0.3])
    q = trainer.quality_signal_kl(p, p)
    assert q == pytest.approx(1.0, abs=1e-12)


def test_quality_signal_kl_decreases_with_distance() -> None:
    """More divergent predicted => lower Q."""
    trainer = _make_trainer()
    emp = np.array([0.7, 0.2, 0.1])
    near = np.array([0.65, 0.25, 0.10])
    far = np.array([0.10, 0.20, 0.70])
    q_near = trainer.quality_signal_kl(emp, near)
    q_far = trainer.quality_signal_kl(emp, far)
    assert q_near > q_far
    # Both lie in (0, 1].
    assert 0.0 < q_far < q_near <= 1.0


# ---------------------------------------------------------------------------
# Phase 7 -- Blended quality signal (curriculum)
# ---------------------------------------------------------------------------


def test_quality_signal_blended_pure_heuristic_at_blend_zero() -> None:
    """blend=0 => identical to existing heuristic quality_signal."""
    trainer = _make_trainer()
    emp = np.array([0.5, 0.3, 0.2])
    pred = np.array([0.4, 0.4, 0.2])
    q_heur = trainer.quality_signal(0.4, 0.2, 0.0, 0.1)
    q_blend = trainer.quality_signal_blended(
        0.4, 0.2, 0.0, 0.1, empirical_dist=emp, predicted_dist=pred, blend=0.0
    )
    assert q_blend == pytest.approx(q_heur, abs=1e-12)


def test_quality_signal_blended_pure_kl_at_blend_one() -> None:
    """blend=1 => identical to quality_signal_kl."""
    trainer = _make_trainer()
    emp = np.array([0.5, 0.3, 0.2])
    pred = np.array([0.4, 0.4, 0.2])
    q_kl = trainer.quality_signal_kl(emp, pred)
    q_blend = trainer.quality_signal_blended(
        0.4, 0.2, 0.0, 0.1, empirical_dist=emp, predicted_dist=pred, blend=1.0
    )
    assert q_blend == pytest.approx(q_kl, abs=1e-12)


def test_quality_signal_blended_falls_back_when_no_dists() -> None:
    """Missing empirical/predicted => pure heuristic regardless of blend."""
    trainer = _make_trainer()
    q_heur = trainer.quality_signal(0.4, 0.2, 0.0, 0.1)
    # Sweep blends; result must equal heuristic regardless of blend value.
    for blend in [0.0, 0.25, 0.5, 0.75, 1.0]:
        q_no_emp = trainer.quality_signal_blended(
            0.4, 0.2, 0.0, 0.1, empirical_dist=None, predicted_dist=np.array([0.5, 0.5]), blend=blend
        )
        q_no_pred = trainer.quality_signal_blended(
            0.4, 0.2, 0.0, 0.1, empirical_dist=np.array([0.5, 0.5]), predicted_dist=None, blend=blend
        )
        q_neither = trainer.quality_signal_blended(
            0.4, 0.2, 0.0, 0.1, blend=blend
        )
        assert q_no_emp == pytest.approx(q_heur, abs=1e-12)
        assert q_no_pred == pytest.approx(q_heur, abs=1e-12)
        assert q_neither == pytest.approx(q_heur, abs=1e-12)


# ---------------------------------------------------------------------------
# Phase 7 -- Fisher-natural gradient
# ---------------------------------------------------------------------------


def test_natural_gradient_scales_inversely_with_data() -> None:
    """Edges with little data (small alpha+beta) get larger updates."""
    trainer = _make_trainer(n_vertices=2)
    # Cold edge (0, 1): alpha=beta=1 (default prior). Hot edge (1, 0):
    # bump it heavily so alpha + beta = 200.
    pm = trainer.posterior_mask
    pm.alpha[1, 0] = 100.0
    pm.beta[1, 0] = 100.0
    # Same gradient on every edge.
    g_alpha = np.ones_like(pm.alpha)
    g_beta = np.ones_like(pm.beta)
    nat_a, nat_b = trainer.natural_gradient_update(g_alpha, g_beta)
    # Cold edge update is much larger than hot edge update.
    assert nat_a[0, 1] > nat_a[1, 0] * 50.0  # cold ~1/2, hot ~1/200
    assert nat_b[0, 1] > nat_b[1, 0] * 50.0
    # Sanity: natural gradient at cold edge ~ 1/(1+1) = 0.5.
    assert nat_a[0, 1] == pytest.approx(0.5, abs=1e-3)


def test_natural_gradient_handles_zero_data_edge_via_damping() -> None:
    """Even at minimum prior alpha=beta=1, no NaN; output is finite."""
    trainer = _make_trainer(n_vertices=3)
    g_alpha = np.full_like(trainer.posterior_mask.alpha, 1.5)
    g_beta = np.full_like(trainer.posterior_mask.beta, -0.7)
    nat_a, nat_b = trainer.natural_gradient_update(g_alpha, g_beta, damping=1e-6)
    assert np.all(np.isfinite(nat_a))
    assert np.all(np.isfinite(nat_b))


# ---------------------------------------------------------------------------
# Phase 7 -- Cramer-Rao confidence
# ---------------------------------------------------------------------------


def test_crb_confidence_zero_with_uniform_prior() -> None:
    """alpha=beta=1 (no observations) => CRB confidence near zero."""
    trainer = _make_trainer(n_vertices=3)
    # Default cold-start: alpha=beta=1 everywhere => n_eff = 0.
    c = trainer.crb_confidence(0, 1, target_variance=0.01)
    assert c == pytest.approx(0.0, abs=1e-12)


def test_crb_confidence_satisfied_with_enough_data() -> None:
    """alpha+beta=1000 => CRB confidence >= 1.0 for target variance 0.01."""
    trainer = _make_trainer(n_vertices=2)
    pm = trainer.posterior_mask
    pm.alpha[0, 1] = 500.0
    pm.beta[0, 1] = 500.0
    c = trainer.crb_confidence(0, 1, target_variance=0.01)
    assert c >= 1.0


def test_crb_confidence_matrix_shape() -> None:
    """crb_confidence_matrix shape matches posterior_mask shape."""
    trainer = _make_trainer(n_vertices=4)
    m = trainer.crb_confidence_matrix(target_variance=0.01)
    assert m.shape == trainer.posterior_mask.alpha.shape == (4, 4)


# ---------------------------------------------------------------------------
# Phase 7 -- KL convergence diagnostic
# ---------------------------------------------------------------------------


def test_kl_progress_zero_for_unchanged_posterior() -> None:
    """previous == current => mean KL = 0."""
    trainer = _make_trainer(n_vertices=3)
    prev_a = trainer.posterior_mask.alpha.copy()
    prev_b = trainer.posterior_mask.beta.copy()
    kl = trainer.kl_progress(prev_a, prev_b)
    assert kl == pytest.approx(0.0, abs=1e-9)


def test_kl_progress_positive_when_changed() -> None:
    """After updates, mean KL > 0."""
    trainer = _make_trainer(n_vertices=3)
    prev_a = trainer.posterior_mask.alpha.copy()
    prev_b = trainer.posterior_mask.beta.copy()
    # Apply a few updates to diverge the posterior from the snapshot.
    trainer.posterior_mask.update(0, 1, 0.9)
    trainer.posterior_mask.update(1, 2, 0.1)
    trainer.posterior_mask.update(2, 0, 0.8)
    kl = trainer.kl_progress(prev_a, prev_b)
    assert kl > 0.0
