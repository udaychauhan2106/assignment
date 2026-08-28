import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel


ORDER_ID_PATTERN = re.compile(r"^ORD-\d{4}$")
DEFAULT_ORDERS_PATH = Path(__file__).resolve().parent.parent / "data" / "orders.json"
OrderLookupState = Literal["missing", "malformed", "not_found", "found"]


class OrderItem(BaseModel):
    name: str
    quantity: int
    final_sale: bool


class OrderLookupResult(BaseModel):
    found: bool
    state: OrderLookupState
    order_id: str | None = None
    membership_tier: str | None = None
    items: list[OrderItem] | None = None
    placed_at: str | None = None
    status: str | None = None
    status_updated_at: str | None = None
    shipped_at: str | None = None
    delivered_at: str | None = None
    carrier: str | None = None
    tracking_number: str | None = None
    estimated_delivery: str | None = None
    customer_safe_message: str | None = None


def _empty_result(state: Literal["missing", "malformed", "not_found"]) -> OrderLookupResult:
    return OrderLookupResult(found=False, state=state)


def _load_order_index(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)

    orders = payload.get("orders") if isinstance(payload, dict) else None
    if not isinstance(orders, list):
        raise ValueError("orders.json must contain an orders list")

    index: dict[str, dict[str, Any]] = {}
    for order in orders:
        if isinstance(order, dict) and isinstance(order.get("order_id"), str):
            index[order["order_id"].upper()] = order
    return index


def _safe_result(order: dict[str, Any]) -> OrderLookupResult:
    status = str(order.get("status") or "").lower()
    items = [
        OrderItem(
            name=str(item.get("name") or ""),
            quantity=int(item.get("quantity") or 0),
            final_sale=bool(item.get("final_sale", False)),
        )
        for item in order.get("items", [])
        if isinstance(item, dict)
    ]

    delivery_fields: dict[str, Any] = {
        "shipped_at": order.get("shipped_at"),
        "delivered_at": order.get("delivered_at"),
        "carrier": order.get("carrier"),
        "tracking_number": order.get("tracking_number"),
        "estimated_delivery": order.get("estimated_delivery"),
    }
    if status in {"cancelled", "returned"}:
        delivery_fields = {field: None for field in delivery_fields}

    return OrderLookupResult(
        found=True,
        state="found",
        order_id=str(order["order_id"]),
        membership_tier=order.get("membership_tier"),
        items=items,
        placed_at=order.get("placed_at"),
        status=order.get("status"),
        status_updated_at=order.get("status_updated_at"),
        customer_safe_message=order.get("customer_safe_message"),
        **delivery_fields,
    )


def lookup_order(
    order_id: str | None,
    *,
    orders_path: str | Path = DEFAULT_ORDERS_PATH,
) -> OrderLookupResult:
    """Return only customer-safe fields for a validated order ID."""
    if order_id is None or not order_id.strip():
        return _empty_result("missing")

    normalized_id = order_id.strip().upper()
    if not ORDER_ID_PATTERN.fullmatch(normalized_id):
        return _empty_result("malformed")

    order = _load_order_index(Path(orders_path)).get(normalized_id)
    if order is None:
        return _empty_result("not_found")
    return _safe_result(order)