from app.orders import OrderLookupResult, lookup_order


def test_exact_valid_order_id_returns_current_status() -> None:
    result = lookup_order("ORD-1007")

    assert isinstance(result, OrderLookupResult)
    assert result.found is True
    assert result.state == "found"
    assert result.order_id == "ORD-1007"
    assert result.status == "shipped"
    assert result.carrier == "UPS"


def test_order_id_matching_is_case_insensitive_and_strips_whitespace() -> None:
    result = lookup_order(" ord-1007 ")

    assert result.found is True
    assert result.order_id == "ORD-1007"


def test_missing_order_id_does_not_read_orders_file() -> None:
    result = lookup_order(None, orders_path="does-not-exist.json")

    assert result.model_dump(exclude_none=True) == {"found": False, "state": "missing"}


def test_blank_order_id_is_missing() -> None:
    assert lookup_order("   ").state == "missing"


def test_malformed_order_id_is_rejected() -> None:
    result = lookup_order("order 1007")

    assert result.found is False
    assert result.state == "malformed"


def test_unknown_valid_order_id_is_not_found() -> None:
    result = lookup_order("ORD-9999")

    assert result.found is False
    assert result.state == "not_found"


def test_missing_eta_remains_unavailable() -> None:
    result = lookup_order("ORD-1011")

    assert result.status == "shipped"
    assert result.estimated_delivery is None


def test_cancelled_order_hides_stale_delivery_fields() -> None:
    result = lookup_order("ORD-1004")

    assert result.status == "cancelled"
    assert result.carrier is None
    assert result.tracking_number is None
    assert result.estimated_delivery is None
    assert result.shipped_at is None
    assert result.delivered_at is None


def test_returned_order_hides_stale_delivery_fields() -> None:
    result = lookup_order("ORD-1008")

    assert result.status == "returned"
    assert result.carrier is None
    assert result.tracking_number is None
    assert result.estimated_delivery is None
    assert result.shipped_at is None
    assert result.delivered_at is None


def test_result_contains_no_sensitive_or_raw_order_fields() -> None:
    result = lookup_order("ORD-1007")
    output = result.model_dump()
    output_text = str(output)

    assert "email" not in output
    assert "shipping_address" not in output
    assert "internal" not in output
    assert "risk_score" not in output
    assert "warehouse_note" not in output
    assert "ava.morgan@example.test" not in output_text
    assert "220 King Street West" not in output_text
    assert not isinstance(result, dict)