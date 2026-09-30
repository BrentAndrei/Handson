"""
PHASE 6 — E7 diagnostic: world->local quaternion composition order in
signavatars-service/fit_smplx.py (CCDIKChain.solve, final conversion block).

fit_smplx.py computes, for every non-root joint:

    local[i] = _multiply_quats(world_q, parent_inv)

and _multiply_quats(a, b) is documented "apply b first, then a", i.e. it
returns R(a) . R(b). So the code produces R(world) . R(parent)^-1.

A parent->child chain requires R(world_child) = R(parent_world) . R(local_child),
which inverts to R(local_child) = R(parent_world)^-1 . R(world_child).

Quaternion products do NOT commute, so the two are equal only when the parent is
unrotated. Roots are skipped by the `parent_idx is None` branch, which is why the
bug stayed dormant at the root and only corrupts non-root joints — the classic
"twisted limb" symptom, because the aim the solver solved for is not what
SMPL-X re-composes at playback.

This proves the order numerically with a round trip: build a chain, drive the
solver to a known asymmetric pose ("arms raised"), take the LOCAL rotations it
produced, re-compose them the way SMPL-X does (parent_world (x) local), and
compare against the WORLD rotations the solver actually solved for.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).parent))
from fit_smplx import CCDIKChain  # noqa: E402


def qmul(a, b):
    """R(a) . R(b) — same convention as CCDIKChain._multiply_quats."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def qconj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def angle_between(a, b):
    d = float(np.clip(abs(np.dot(a, b)), -1.0, 1.0))
    return np.degrees(np.arccos(d))


IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])


def drive_ik(chain, names, parents, targets, iterations=40):
    """Replay the solver's own CCD step so we can snapshot the WORLD rotations
    it intends, which solve() does not expose."""
    chain._capture_initial_directions()
    for _ in range(iterations):
        for j in range(len(names) - 1, -1, -1):
            nm = names[j]
            if nm not in targets:
                continue
            tgt = targets[nm]
            eff = chain.joint_positions[j]
            if np.linalg.norm(tgt - eff) < 1e-4:
                continue
            p = parents[j]
            if p is None:
                chain.joint_positions[j] += (tgt - eff) * 0.3
                continue
            ppos = chain.joint_positions[p]
            to_eff, to_tgt = eff - ppos, tgt - ppos
            ne, nt = np.linalg.norm(to_eff), np.linalg.norm(to_tgt)
            if ne < 1e-10 or nt < 1e-10:
                continue
            to_eff, to_tgt = to_eff / ne, to_tgt / nt
            axis = np.cross(to_eff, to_tgt)
            an = np.linalg.norm(axis)
            if an < 1e-10:
                continue
            Rm = Rot.from_rotvec(
                axis / an * np.arccos(np.clip(np.dot(to_eff, to_tgt), -1, 1))
            ).as_matrix()
            chain.joint_positions[j] = ppos + Rm @ to_eff
            for c in chain._get_children(j):
                rel = chain.joint_positions[c] - chain.joint_positions[j]
                chain.joint_positions[c] = chain.joint_positions[j] + Rm @ rel


def intended_world_rotations(chain, names, parents):
    """The initial_dir -> current_dir world rotation for every non-root joint."""
    world = {}
    for i in range(len(names)):
        p = parents[i]
        if p is None:
            continue
        init = chain._initial_dirs[i]
        cur = chain.joint_positions[i] - chain.joint_positions[p]
        cur = cur / np.linalg.norm(cur)
        axis = np.cross(init, cur)
        an = np.linalg.norm(axis)
        if an < 1e-10:
            world[i] = IDENTITY.copy()
        else:
            world[i] = Rot.from_rotvec(
                axis / an * np.arccos(np.clip(np.dot(init, cur), -1, 1))
            ).as_quat()
    return world



def main():
    # 4-joint chain root -> b1 -> b2 -> b3. A non-identity rotation on b1 is
    # what makes the parent term matter.
    names = ["root", "b1", "b2", "b3"]
    parents = [None, 0, 1, 2]
    chain = CCDIKChain(names, parents)
    chain.joint_positions = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
    ])

    # Deliberately asymmetric ("arms raised") so no order error can cancel out.
    targets = {
        "b1": np.array([1.0, 1.2, 0.4]),
        "b2": np.array([1.6, 1.9, 1.0]),
        "b3": np.array([1.4, 2.4, 1.9]),
    }

    # Run the PRODUCTION solver end to end. solve() leaves the solved pose in
    # joint_positions and the LOCAL rotations in joint_rotations.
    chain.solve(targets, iterations=40)

    # Recompute the world rotations the solver solved for, from its own final
    # state, so the comparison is self-consistent with whatever it produced.
    world = intended_world_rotations(chain, names, parents)
    # The root has no parent, so solve() never assigns it -> stays identity.
    world[0] = IDENTITY.copy()

    print("=== E7 gate: does solve() emit the CORRECTLY-ordered local rotation? ===")
    print("joint | parent world | emitted vs CORRECT | emitted vs SWAPPED | winner")
    print("-" * 84)

    worst_correct = 0.0
    worst_swapped = 0.0
    for i in range(1, len(names)):
        p = parents[i]
        emitted = chain.joint_rotations[i]
        correct = qmul(qconj(world[p]), world[i])    # chain relation
        swapped = qmul(world[i], qconj(world[p]))    # the old, wrong order
        e_c = angle_between(emitted, correct)
        e_s = angle_between(emitted, swapped)
        worst_correct = max(worst_correct, e_c)
        worst_swapped = max(worst_swapped, e_s)
        winner = "CORRECT" if e_c < e_s else "SWAPPED"
        print(
            f"{names[i]:>5} | {angle_between(world[p], IDENTITY):9.2f}deg | "
            f"{e_c:14.5f}deg | {e_s:16.5f}deg | {winner}"
        )

    print("-" * 84)
    print(f"worst |emitted - CORRECT| : {worst_correct:.6f} deg")
    print(f"worst |emitted - SWAPPED| : {worst_swapped:.6f} deg")
    print()
    if worst_correct < 1e-3 and worst_swapped > 1.0:
        print("VERDICT: PASS — fit_smplx.py now emits the correctly-ordered local")
        print("         rotation (conjugate(parent) * world). E7 is fixed.")
        return 0
    if worst_swapped < 1e-3:
        print("VERDICT: E7 STILL PRESENT — solve() is emitting the swapped order")
        print("         R(world).R(parent)^-1 instead of R(parent)^-1.R(world).")
        return 2
    print(f"INCONCLUSIVE — emitted rotations match neither order closely")
    print(f"(correct {worst_correct:.4f}deg, swapped {worst_swapped:.4f}deg).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
