"""Tests for adapters/visualization/call_list.py — ranking helpers."""

from adapters.visualization.call_list import precision_at_capacity, rank_call_list


def _candidates() -> list[dict[str, object]]:
    return [
        {"id": "a", "risk_score": 0.10, "actual_no_show": 0},
        {"id": "b", "risk_score": 0.90, "actual_no_show": 1},
        {"id": "c", "risk_score": 0.50, "actual_no_show": 1},
        {"id": "d", "risk_score": 0.30, "actual_no_show": 0},
    ]


class TestRankCallList:
    def test_orders_by_score_descending(self) -> None:
        ranked = rank_call_list(_candidates(), capacity_k=4)
        assert [r["id"] for r in ranked] == ["b", "c", "d", "a"]

    def test_respects_capacity(self) -> None:
        ranked = rank_call_list(_candidates(), capacity_k=2)
        assert [r["id"] for r in ranked] == ["b", "c"]

    def test_zero_capacity_returns_empty(self) -> None:
        assert rank_call_list(_candidates(), capacity_k=0) == []

    def test_negative_capacity_clamped(self) -> None:
        assert rank_call_list(_candidates(), capacity_k=-5) == []


class TestPrecisionAtCapacity:
    def test_top_two_both_no_shows(self) -> None:
        ranked = rank_call_list(_candidates(), capacity_k=2)
        assert precision_at_capacity(ranked) == 1.0

    def test_full_list_precision(self) -> None:
        ranked = rank_call_list(_candidates(), capacity_k=4)
        assert precision_at_capacity(ranked) == 0.5

    def test_empty_list_is_zero(self) -> None:
        assert precision_at_capacity([]) == 0.0
