"""环状双酶图谱重建：仅由实验片段长度重建环状分子上 A、B 两类切点的相对位置。

规则
----
- 只依据实验长度，不要求 DNA 序列或酶识别串：输入 A、B 各自单酶切及
  A+B 双酶切的片段长度多重集（重复长度按出现次数计算）。
- 假定同一环状分子被完全切开，且 A、B 切点互不重合：环状分子上 k 个切点
  产生 k 条片段，故 A 单酶 a 片 ⇒ A 有 a 个切点，B 单酶 b 片 ⇒ B 有 b 个
  切点，双酶切共 a+b 个切点、a+b 条片段。双酶片数不等于两组单酶片数之和
  时前提不成立，由上层返回 HTTP 400。
- 双酶片段即相邻切点之间的弧。把 m = a+b 条双酶片段排成环，并为 m 个边界
  各指定 A/B 归属（恰 a 个 A、b 个 B）：从每个 A 边界顺时针到下一个 A
  边界的弧长之和即 A 单酶切的一条片段，全部 A 弧须重现 A 长度多重集；
  B 同理。
- 枚举全部不同的片段环序（多重集排列）与边界归属组合，只保留两组单酶
  长度多重集都能重现的图谱。

旋转与镜像
----------
同一图谱旋转或镜像后应视为相同：用（切点标签, 其后间距）序列在全部旋转
与正/反两个方向下的字典序最小者作为规范表示去重；输出时按规范表示从 0
计坐标（首个切点位于 0），全部图谱之间也按规范表示排序。
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations

MIN_SINGLE_FRAGMENTS, MAX_SINGLE_FRAGMENTS = 1, 3
MIN_DOUBLE_FRAGMENTS, MAX_DOUBLE_FRAGMENTS = 2, 6
MIN_TOTAL_LENGTH, MAX_TOTAL_LENGTH = 4, 5000

LABEL_A, LABEL_B = "A", "B"


def _distinct_permutations(lengths: list[int]) -> list[tuple[int, ...]]:
    """多重集的全部不同排列（重复长度按出现次数计，相同排列只生成一次）。"""
    counts = Counter(lengths)
    values = sorted(counts)
    size = len(lengths)
    current = [0] * size
    result: list[tuple[int, ...]] = []

    def backtrack(pos: int) -> None:
        if pos == size:
            result.append(tuple(current))
            return
        for value in values:
            if counts[value] == 0:
                continue
            counts[value] -= 1
            current[pos] = value
            backtrack(pos + 1)
            counts[value] += 1

    backtrack(0)
    return result


def _arc_lengths(
    frag_lengths: tuple[int, ...], labels: list[str], target: str
) -> list[int]:
    """从每个 target 边界顺时针到下一个 target 边界的弧长列表。

    片段 j 位于边界 j 与边界 j+1（模 m）之间；从 target 边界出发累加片段，
    直到抵达下一个 target 边界为止。
    """
    m = len(frag_lengths)
    arcs: list[int] = []
    for i in range(m):
        if labels[i] != target:
            continue
        total = 0
        j = i
        while True:
            total += frag_lengths[j]
            j = (j + 1) % m
            if labels[j] == target:
                break
        arcs.append(total)
    return arcs


def _canonical(
    frag_lengths: tuple[int, ...], labels: list[str]
) -> tuple[tuple[str, int], ...]:
    """（切点标签, 其后间距）序列在全部旋转与镜像下的字典序最小表示。

    镜像即反向绕行：边界 i 之后的间距变为原序列中它之前的间距，故镜像
    序列第 t 项为 (labels[-t], frag_lengths[-t-1])。
    """
    m = len(frag_lengths)
    pairs = tuple((labels[i], frag_lengths[i]) for i in range(m))
    mirrored = tuple((labels[-t], frag_lengths[-t - 1]) for t in range(m))
    best: tuple[tuple[str, int], ...] | None = None
    for seq in (pairs, mirrored):
        for k in range(m):
            rotated = seq[k:] + seq[:k]
            if best is None or rotated < best:
                best = rotated
    return best


def circular_map_reconstruction(
    a_lengths: list[int], b_lengths: list[int], double_lengths: list[int]
) -> dict:
    """枚举候选图谱，返回全部能重现两组单酶长度多重集的不同图谱。

    三组长度均已通过校验：正整数、片数在范围内、总长相等且双酶片数等于
    两组单酶片数之和。
    """
    total = sum(double_lengths)
    a_sorted = sorted(a_lengths)
    b_sorted = sorted(b_lengths)
    m = len(double_lengths)
    a_count = len(a_lengths)

    # 规范表示 → 占位：同一图谱的不同旋转/镜像枚举结果在此去重
    found: dict[tuple[tuple[str, int], ...], None] = {}
    candidate_count = 0
    for frag_lengths in _distinct_permutations(double_lengths):
        # 边界归属：在 m 个边界中选 a_count 个标 A，其余标 B
        for a_positions in combinations(range(m), a_count):
            candidate_count += 1
            labels = [LABEL_B] * m
            for pos in a_positions:
                labels[pos] = LABEL_A
            if sorted(_arc_lengths(frag_lengths, labels, LABEL_A)) != a_sorted:
                continue
            if sorted(_arc_lengths(frag_lengths, labels, LABEL_B)) != b_sorted:
                continue
            found[_canonical(frag_lengths, labels)] = None

    maps: list[dict] = []
    for map_index, canonical in enumerate(sorted(found), start=1):
        labels = [label for label, _ in canonical]
        gaps = [gap for _, gap in canonical]
        # 规范表示的首个切点位于 0，其余坐标顺时针累加
        positions = [0]
        for gap in gaps[:-1]:
            positions.append(positions[-1] + gap)
        sites = [
            {"index": i + 1, "label": labels[i], "position": positions[i]}
            for i in range(m)
        ]
        fragments = [
            {
                "index": i + 1,
                "start": positions[i],
                "end": positions[i] + gaps[i],
                "length": gaps[i],
                "left_label": labels[i],
                "right_label": labels[(i + 1) % m],
            }
            for i in range(m)
        ]
        maps.append(
            {
                "map_index": map_index,
                "signature": " ".join(f"{label}{gap}" for label, gap in canonical),
                "sites": sites,
                "fragments": fragments,
            }
        )

    result: dict = {
        "total_length": total,
        "a_fragments": list(a_lengths),
        "b_fragments": list(b_lengths),
        "double_fragments": list(double_lengths),
        "a_site_count": a_count,
        "b_site_count": len(b_lengths),
        "site_count": m,
        "candidate_count": candidate_count,
        "map_count": len(maps),
        "maps": maps,
    }

    if not maps:
        result["solvable"] = False
        result["message"] = (
            f"无解：枚举了 {candidate_count} 种候选（{m} 条双酶片段的全部不同"
            f"环序 × 边界的 A/B 归属组合），没有任何一种能同时重现 A 单酶切 "
            f"{a_sorted} 与 B 单酶切 {b_sorted} 的长度多重集。"
        )
        return result

    result["solvable"] = True
    return result
