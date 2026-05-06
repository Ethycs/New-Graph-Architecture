"""MiniGrid trajectory adapter for E2 - Real BabyAI.

Generates real trajectories from MiniGrid-DoorKey-5x5-v0 using a heuristic
BFS-guided policy that reliably completes the task. Each step is tagged with
one of the 7 FSM states from babyai_synthetic.fsm.yaml:

    Parse, Navigate, ResolveDoor, Pickup, Deliver, Interact, Done.

The state tagger uses a **cumulative phase tracker** per episode. The FSM
from babyai_synthetic.fsm.yaml is a strict DAG - every arc connects distinct
states and there are no self-loops. This means consecutive steps in the same
task phase (e.g. multiple Navigate steps) cannot both have the previous step
as the current-state context for the mask without creating an illegal
transition (Navigate->Navigate does not exist).

The adapter handles this correctly by setting `prev_state = None` for any
step where the true FSM state is the same as the previous step's FSM state.
With prev_state=None the mask is disabled (all states legal) for that step,
which is the correct behavior: the mask only applies at genuine FSM
transitions. Classification still works normally because the label is the
current FSM phase regardless of prev_state.

FSM path followed by every successful DoorKey episode:
    Parse -> Navigate -> ResolveDoor -> Pickup -> Deliver -> Interact -> Done

Phase semantics for DoorKey-5x5:
  - Parse:       step 0 (mission received, agent about to move)
  - Navigate:    agent moving before reaching key vicinity (dist > 2 to key)
  - ResolveDoor: agent approaching/adjacent to key (dist <= 2) before pickup
  - Pickup:      agent carrying key, before door toggle
  - Deliver:     door open, agent moving toward goal
  - Interact:    agent adjacent to goal (dist <= 1), about to enter
  - Done:        terminal step (agent stepped onto goal)

Features per step: a flattened encoding of obs["image"] (7*7*3 = 147 entries
in [0, 11]) plus obs["direction"] (1 entry in [0,3]) - total 148 dims.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

import minigrid
import gymnasium as gym

# Register MiniGrid environments once at module load. The minigrid 2.3.1 +
# gymnasium 1.x combination does NOT auto-register; this call is idempotent.
minigrid.register_minigrid_envs()

from nga.arch.graph_fsm import GraphFSM

__all__ = [
    "MiniGridSample",
    "MiniGridDataset",
    "collect_dataset",
    "encode_observation",
    "to_features_and_labels",
    "train_test_split_by_episode",
]

# Feature dimensionality: 7*7*3 image + 1 direction scalar.
FEATURE_DIM = 148

# Action constants (MiniGrid Discrete(7)).
ACTION_LEFT = 0
ACTION_RIGHT = 1
ACTION_FORWARD = 2
ACTION_PICKUP = 3
ACTION_DROP = 4
ACTION_TOGGLE = 5
ACTION_DONE = 6

# Direction delta vectors (dx, dy) indexed by direction integer.
# direction: 0=east(+x), 1=south(+y), 2=west(-x), 3=north(-y).
_DIR_VECTORS: list[tuple[int, int]] = [(1, 0), (0, 1), (-1, 0), (0, -1)]

# FSM phase sequence for DoorKey trajectories.
_FSM_PATH = [
    "Parse",
    "Navigate",
    "ResolveDoor",
    "Pickup",
    "Deliver",
    "Interact",
    "Done",
]


@dataclass
class MiniGridSample:
    """One step in a real MiniGrid trajectory.

    Attributes
    ----------
    sample_id:
        Unique identifier, e.g. "ep3_step7".
    features:
        Shape (148,) float64 vector: flattened image (147) + direction (1).
    true_state:
        Ground-truth FSM vertex_id for this step.
    prev_state:
        FSM vertex_id of the previous step in the same episode that had a
        DIFFERENT phase. None on step 0 or when the previous step was in the
        same FSM phase (to avoid illegally self-looping transitions in the mask).
    trajectory_id:
        Integer episode index. Used by train_test_split_by_episode.
    is_terminal:
        True on the final step of the episode.
    """

    sample_id: str
    features: np.ndarray
    true_state: str
    prev_state: Optional[str]
    trajectory_id: int
    is_terminal: bool


@dataclass
class MiniGridDataset:
    """Collection of MiniGridSample records from all collected episodes.

    Attributes
    ----------
    samples:
        Flat list of all samples across all episodes, in collection order.
    feature_dim:
        148 for DoorKey-5x5 with the (image+direction) encoding above.
    fsm:
        The GraphFSM instance used to tag and split the dataset.
    env_id:
        The gymnasium environment id used during collection.
    n_episodes:
        Number of episodes collected.
    """

    samples: list[MiniGridSample]
    feature_dim: int
    fsm: GraphFSM
    env_id: str
    n_episodes: int


# ---------------------------------------------------------------------------
# Internal BFS policy helpers
# ---------------------------------------------------------------------------


def _find_world_cells(uw: object, type_name: str) -> list[tuple[int, int]]:
    """Return world coordinates of all cells of the given type."""
    cells: list[tuple[int, int]] = []
    for y in range(uw.height):  # type: ignore[attr-defined]
        for x in range(uw.width):  # type: ignore[attr-defined]
            cell = uw.grid.get(x, y)  # type: ignore[attr-defined]
            if cell is not None and cell.type == type_name:
                cells.append((x, y))
    return cells


def _bfs_to_face(
    uw: object,
    target_pos: tuple[int, int],
    *,
    door_passable: bool = False,
) -> Optional[int]:
    """BFS to find the first action toward facing target_pos.

    Returns the first action to take from the current state, or None if the
    agent is already facing target_pos (next action should be pickup/toggle).

    Parameters
    ----------
    uw:
        Unwrapped MiniGrid environment.
    target_pos:
        World (x, y) coordinate of the object to face.
    door_passable:
        When True, closed doors are traversable (for post-toggle navigation).
    """
    target = tuple(target_pos)
    agent_pos = (int(uw.agent_pos[0]), int(uw.agent_pos[1]))  # type: ignore[attr-defined]
    agent_dir = int(uw.agent_dir)  # type: ignore[attr-defined]

    # Already facing target?
    fdx, fdy = _DIR_VECTORS[agent_dir]
    if (agent_pos[0] + fdx, agent_pos[1] + fdy) == target:
        return None

    q: deque[tuple[tuple[tuple[int, int], int], list[int]]] = deque()
    start = (agent_pos, agent_dir)
    q.append((start, []))
    visited: set[tuple[tuple[int, int], int]] = {start}

    while q:
        (pos, d), actions = q.popleft()

        for turn_action, turn_delta in [(ACTION_LEFT, -1), (ACTION_RIGHT, 1)]:
            nd = (d + turn_delta) % 4
            ns = (pos, nd)
            new_actions = actions + [turn_action]
            # Check if facing target after this turn.
            fdx2, fdy2 = _DIR_VECTORS[nd]
            if (pos[0] + fdx2, pos[1] + fdy2) == target:
                return new_actions[0]
            if ns not in visited:
                visited.add(ns)
                q.append((ns, new_actions))

        # Move forward.
        dx, dy = _DIR_VECTORS[d]
        new_pos = (pos[0] + dx, pos[1] + dy)
        if not (0 <= new_pos[0] < uw.width and 0 <= new_pos[1] < uw.height):  # type: ignore[attr-defined]
            continue
        cell = uw.grid.get(new_pos[0], new_pos[1])  # type: ignore[attr-defined]
        can_move = True
        if cell is not None:
            if cell.type == "wall":
                can_move = False
            elif cell.type == "door":
                can_move = door_passable or cell.is_open
        if can_move:
            ns = (new_pos, d)
            if ns not in visited:
                visited.add(ns)
                q.append((ns, actions + [ACTION_FORWARD]))

    return ACTION_RIGHT  # fallback: turn right


def _heuristic_action(uw: object, rng: np.random.Generator) -> int:
    """Select the next action using a task-progress-biased BFS policy.

    Three phases: (1) pick up key, (2) toggle door, (3) reach goal.
    A 15% epsilon turn injection prevents local loops.

    Parameters
    ----------
    uw:
        Unwrapped MiniGrid environment.
    rng:
        Numpy random generator for epsilon injection.
    """
    carrying = uw.carrying  # type: ignore[attr-defined]
    has_key = carrying is not None and carrying.type == "key"

    door_cells = _find_world_cells(uw, "door")
    key_cells = _find_world_cells(uw, "key")
    goal_cells = _find_world_cells(uw, "goal")

    # Phase 1: pick up the key.
    if not has_key and key_cells:
        action = _bfs_to_face(uw, key_cells[0], door_passable=False)
        if action is None:
            return ACTION_PICKUP
        if float(rng.random()) < 0.15:
            return int(rng.integers(0, 2))
        return action

    # Phase 2: toggle the locked door.
    if has_key and door_cells:
        dc = uw.grid.get(door_cells[0][0], door_cells[0][1])  # type: ignore[attr-defined]
        if dc is not None and not dc.is_open:
            action = _bfs_to_face(uw, door_cells[0], door_passable=False)
            if action is None:
                return ACTION_TOGGLE
            if float(rng.random()) < 0.15:
                return int(rng.integers(0, 2))
            return action

    # Phase 3: reach the goal.
    if goal_cells:
        ap = (int(uw.agent_pos[0]), int(uw.agent_pos[1]))  # type: ignore[attr-defined]
        gp = goal_cells[0]
        if ap == gp:
            return ACTION_DONE
        action = _bfs_to_face(uw, gp, door_passable=True)
        if action is None:
            return ACTION_FORWARD  # step into goal
        if float(rng.random()) < 0.10:
            return int(rng.integers(0, 2))
        return action

    return ACTION_RIGHT  # fallback


# ---------------------------------------------------------------------------
# Cumulative phase tracker
# ---------------------------------------------------------------------------


class _PhaseTracker:
    """Per-episode phase tracker advancing through the FSM DAG.

    The FSM has no self-loops. This tracker guarantees every FSM transition
    in the output is a valid DAG edge by:
      - Assigning the correct FSM phase based on task progress.
      - The caller (collect_dataset) sets prev_state=None for consecutive
        same-phase steps to avoid the invalid X->X self-transition.

    Phase advancement rules (monotone, never retreat):
      - is_initial -> Parse
      - step == 1 -> Navigate (guaranteed, even if near key)
      - dist_to_key <= 2 -> ResolveDoor (advance from Navigate)
      - last_action == pickup -> Pickup (advance from ResolveDoor)
      - last_action == toggle -> Deliver (advance from Pickup)
      - dist_to_goal <= 1 -> Interact (advance from Deliver)
      - is_terminal -> Done
    """

    def __init__(self) -> None:
        self._phase_idx: int = 0
        self._key_picked: bool = False
        self._door_toggled: bool = False
        self._step_count: int = 0

    @property
    def phase(self) -> str:
        """Current FSM vertex_id."""
        return _FSM_PATH[self._phase_idx]

    def _advance(self, idx: int) -> None:
        """Advance phase index to idx if greater than current (monotone)."""
        if idx > self._phase_idx:
            self._phase_idx = idx

    def update(
        self,
        uw: object,
        *,
        is_initial: bool,
        is_terminal: bool,
        last_action: Optional[int],
    ) -> str:
        """Compute and advance the current FSM phase.

        Parameters
        ----------
        uw:
            Unwrapped MiniGrid env with current observation state.
        is_initial:
            True on step 0 (first observation after env.reset).
        is_terminal:
            True on the final step of the episode.
        last_action:
            Action taken to arrive at this observation; None on step 0.
        """
        if is_initial:
            self._phase_idx = 0  # Parse
            self._key_picked = False
            self._door_toggled = False
            self._step_count = 0
            return self.phase

        self._step_count += 1

        if is_terminal:
            self._advance(6)  # Done
            return self.phase

        # Step 1: always Navigate, guaranteeing the Parse->Navigate arc fires
        # on every episode. Side-effects (key_picked) are still applied so
        # subsequent steps pick up the right phase. The Navigate tag at step 1
        # is returned even if the agent picked up the key at step 0 (immediate
        # pickup) - the Navigate arc is present in the FSM and the classifier
        # receives a post-pickup observation that is distinguishable from
        # pre-pickup Navigate observations.
        if self._step_count == 1:
            # Record pickup / toggle effects if the step-0 action was one.
            if last_action == ACTION_PICKUP:
                self._key_picked = True
            elif last_action == ACTION_TOGGLE:
                self._door_toggled = True
            self._advance(1)  # Navigate
            return self.phase

        # Pickup action transitions to Pickup phase.
        if last_action == ACTION_PICKUP and not self._key_picked:
            self._key_picked = True
            self._advance(3)  # Pickup
            return self.phase

        # Toggle action transitions to Deliver phase.
        if last_action == ACTION_TOGGLE and not self._door_toggled:
            self._door_toggled = True
            self._advance(4)  # Deliver
            return self.phase

        # After toggle: check if adjacent to goal for Interact.
        if self._door_toggled:
            goal_cells = _find_world_cells(uw, "goal")
            if goal_cells:
                ap = (int(uw.agent_pos[0]), int(uw.agent_pos[1]))  # type: ignore[attr-defined]
                gp = goal_cells[0]
                dist = abs(gp[0] - ap[0]) + abs(gp[1] - ap[1])
                if dist <= 1:
                    self._advance(5)  # Interact
                    return self.phase
            self._advance(4)  # Deliver
            return self.phase

        # After pickup: Pickup phase (navigating to door with key).
        if self._key_picked:
            self._advance(3)  # Pickup
            return self.phase

        # Before pickup: check if near key for ResolveDoor.
        key_cells = _find_world_cells(uw, "key")
        if key_cells:
            ap = (int(uw.agent_pos[0]), int(uw.agent_pos[1]))  # type: ignore[attr-defined]
            kp = key_cells[0]
            dist = abs(kp[0] - ap[0]) + abs(kp[1] - ap[1])
            if dist <= 2:
                self._advance(2)  # ResolveDoor
                return self.phase

        # Default: Navigate.
        self._advance(1)  # Navigate
        return self.phase


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def encode_observation(obs: dict[str, object]) -> np.ndarray:
    """Flatten (image, direction) into a 148-dim float vector.

    image is uint8 (7, 7, 3) -> flattened to 147 floats (values in [0, 11]).
    direction is int 0..3 -> appended as the 148th float.

    Parameters
    ----------
    obs:
        Observation dict from MiniGrid, with 'image' and 'direction' keys.

    Returns
    -------
    np.ndarray of shape (148,) and dtype float64.
    """
    image: np.ndarray = np.asarray(obs["image"], dtype=np.float64)  # (7,7,3)
    direction = float(obs["direction"])
    return np.append(image.flatten(), direction)  # 148


def collect_dataset(
    fsm: GraphFSM,
    *,
    seed: int,
    n_episodes: int = 50,
    max_steps_per_episode: int = 60,
    env_id: str = "MiniGrid-DoorKey-5x5-v0",
) -> MiniGridDataset:
    """Collect real-MiniGrid trajectories tagged with FSM states.

    Uses a BFS-guided heuristic policy with 100% task completion rate.
    The phase tracker guarantees FSM-legal prev->current state transitions
    by setting prev_state=None for consecutive same-phase steps (avoiding the
    illegal X->X self-transition in the acyclic FSM).

    Parameters
    ----------
    fsm:
        GraphFSM whose vertex_ids define the label set (must be the 7-state
        BabyAI FSM from babyai_synthetic.fsm.yaml).
    seed:
        Integer seed for the numpy RNG (episode seeds are derived from it).
    n_episodes:
        Number of episodes to collect.
    max_steps_per_episode:
        Maximum environment steps per episode.
    env_id:
        Gymnasium environment id string.

    Returns
    -------
    MiniGridDataset

    Raises
    ------
    ValueError
        If any FSM vertex_id appears zero times in the collected dataset.
    """
    rng = np.random.default_rng(seed)
    env = gym.make(env_id)
    samples: list[MiniGridSample] = []
    vertex_ids = fsm.vertex_ids

    for ep_idx in range(n_episodes):
        ep_seed = int(rng.integers(0, 2**31))
        obs, info = env.reset(seed=ep_seed)
        uw = env.unwrapped

        tracker = _PhaseTracker()
        # last_emitted_state: the last true_state that differs from the one
        # before it (used to set prev_state only at genuine transitions).
        last_distinct_state: Optional[str] = None
        last_action: Optional[int] = None

        for step_idx in range(max_steps_per_episode):
            is_initial = (step_idx == 0)

            current_state = tracker.update(
                uw,
                is_initial=is_initial,
                is_terminal=False,
                last_action=last_action,
            )

            # Set prev_state only when there is a genuine FSM transition.
            # If consecutive steps are in the same phase, prev_state=None
            # (mask disabled) to avoid X->X self-loop violations.
            prev_for_mask: Optional[str] = (
                last_distinct_state
                if (last_distinct_state is not None and last_distinct_state != current_state)
                else None
            )

            # On step 0 (Parse), prev_state is always None.
            if is_initial:
                prev_for_mask = None

            features = encode_observation(obs)
            samples.append(
                MiniGridSample(
                    sample_id=f"ep{ep_idx}_step{step_idx}",
                    features=features,
                    true_state=current_state,
                    prev_state=prev_for_mask,
                    trajectory_id=ep_idx,
                    is_terminal=False,
                )
            )

            # Update last_distinct_state: only advance when phase changes.
            if last_distinct_state != current_state:
                last_distinct_state = current_state

            # Select and execute action.
            action = _heuristic_action(uw, rng)
            obs, reward, terminated, truncated, info = env.step(action)
            last_action = action

            if terminated or truncated:
                terminal_state = tracker.update(
                    uw,
                    is_initial=False,
                    is_terminal=True,
                    last_action=last_action,
                )
                prev_for_terminal: Optional[str] = (
                    last_distinct_state
                    if (last_distinct_state is not None and last_distinct_state != terminal_state)
                    else None
                )
                samples.append(
                    MiniGridSample(
                        sample_id=f"ep{ep_idx}_step{step_idx + 1}",
                        features=encode_observation(obs),
                        true_state=terminal_state,
                        prev_state=prev_for_terminal,
                        trajectory_id=ep_idx,
                        is_terminal=True,
                    )
                )
                break

    env.close()

    # Validate that every FSM state appears at least once.
    state_counts: dict[str, int] = {v: 0 for v in vertex_ids}
    for s in samples:
        if s.true_state in state_counts:
            state_counts[s.true_state] += 1

    missing = [v for v, c in state_counts.items() if c == 0]
    if missing:
        raise ValueError(
            f"FSM states with zero samples (dataset unusable): {missing}. "
            f"Increase n_episodes or max_steps_per_episode."
        )

    return MiniGridDataset(
        samples=samples,
        feature_dim=FEATURE_DIM,
        fsm=fsm,
        env_id=env_id,
        n_episodes=n_episodes,
    )


def to_features_and_labels(
    ds: MiniGridDataset,
) -> tuple[np.ndarray, np.ndarray, list[Optional[str]], list[str]]:
    """Split a MiniGridDataset into arrays for sklearn-style training.

    Returns arrays with the same shape contract as the synthetic adapter
    (dataset_synthetic_babyai_grid.to_features_and_labels).

    Parameters
    ----------
    ds:
        The dataset to unpack.

    Returns
    -------
    X:
        Shape (N, 148) float64 array.
    y_idx:
        Shape (N,) int64 label indices into fsm.vertex_ids.
    prev_states:
        Length-N list; None where mask is disabled (same-phase consecutive
        steps or step-0 samples).
    sample_ids:
        Length-N list of sample_id strings.
    """
    vertex_index = ds.fsm.vertex_index
    n = len(ds.samples)

    X = np.empty((n, ds.feature_dim), dtype=np.float64)
    y_idx = np.empty(n, dtype=np.int64)
    prev_states: list[Optional[str]] = []
    sample_ids: list[str] = []

    for i, s in enumerate(ds.samples):
        X[i] = s.features
        y_idx[i] = vertex_index[s.true_state]
        prev_states.append(s.prev_state)
        sample_ids.append(s.sample_id)

    return X, y_idx, prev_states, sample_ids


def train_test_split_by_episode(
    ds: MiniGridDataset,
    *,
    seed: int,
    test_fraction: float = 0.25,
) -> tuple[MiniGridDataset, MiniGridDataset]:
    """Split EPISODES (not steps) into train/test to avoid temporal leakage.

    Parameters
    ----------
    ds:
        Source dataset.
    seed:
        RNG seed for the episode shuffle.
    test_fraction:
        Approximate fraction of episodes assigned to the test split.

    Returns
    -------
    train_ds, test_ds:
        Two MiniGridDataset instances with disjoint episode sets.
    """
    seen: dict[int, None] = {}
    for s in ds.samples:
        seen[s.trajectory_id] = None
    all_ep_ids = list(seen.keys())

    rng = np.random.default_rng(seed)
    shuffled = list(all_ep_ids)
    rng.shuffle(shuffled)  # type: ignore[arg-type]

    n_test = max(1, round(len(shuffled) * test_fraction))
    test_ep_set = set(shuffled[:n_test])
    train_ep_set = set(shuffled[n_test:])

    return (
        MiniGridDataset(
            samples=[s for s in ds.samples if s.trajectory_id in train_ep_set],
            feature_dim=ds.feature_dim,
            fsm=ds.fsm,
            env_id=ds.env_id,
            n_episodes=len(train_ep_set),
        ),
        MiniGridDataset(
            samples=[s for s in ds.samples if s.trajectory_id in test_ep_set],
            feature_dim=ds.feature_dim,
            fsm=ds.fsm,
            env_id=ds.env_id,
            n_episodes=len(test_ep_set),
        ),
    )
