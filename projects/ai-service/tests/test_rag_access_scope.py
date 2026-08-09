from app.rag.filters import RagAccessScope
from app.services.console_agent_service import (
    ROLE_TO_PERMISSION_GROUPS,
    build_rag_access_scope,
)


def test_role_mapping_covers_known_roles():
    assert ROLE_TO_PERMISSION_GROUPS["customer"] == ["customer_service", "public"]
    assert "internal_staff" in ROLE_TO_PERMISSION_GROUPS["staff"]


def test_unknown_role_falls_back_to_customer_scope():
    scope = build_rag_access_scope(user_id="U1", tenant_id="t1", roles=("unknown_role",))
    assert scope.permission_groups == ["customer_service", "public"]
    assert scope.user_id == "U1"
    assert scope.tenant_id == "t1"


def test_admin_role_includes_admin_group():
    scope = build_rag_access_scope(user_id="U1", tenant_id="t1", roles=("admin",))
    assert "admin" in scope.permission_groups


def test_multiple_roles_dedupe_permission_groups():
    scope = build_rag_access_scope(user_id="U1", tenant_id="t1", roles=("customer", "staff"))
    assert scope.permission_groups == ["customer_service", "public", "internal_staff"]


def test_build_returns_rag_access_scope_instance():
    scope = build_rag_access_scope(user_id="U1", tenant_id="t1", roles=("customer",))
    assert isinstance(scope, RagAccessScope)
