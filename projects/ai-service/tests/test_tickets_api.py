from fastapi.testclient import TestClient

from app.core.config import Settings
from app.routers.tickets import get_ticket_workflow_service
from app.services.ticket_workflow_service import TicketWorkflowService
from app.services.tool_confirmation_service import ToolConfirmationService
from app.tools.tool_confirmation import get_tool_confirmation_store
from tests.tool_fakes import FakeTicketCreator, FakeTicketExtractor


def test_ticket_api_plans_confirms_and_executes_one_ticket(
    client: TestClient,
    app,
) -> None:
    settings = Settings(_env_file=None)
    fake_creator = FakeTicketCreator()
    workflow = TicketWorkflowService(
        settings=settings,
        extractor=FakeTicketExtractor(),
        confirmation_service=ToolConfirmationService(
            settings,
            get_tool_confirmation_store(),
        ),
        ticket_creator=fake_creator,
    )
    app.dependency_overrides[get_ticket_workflow_service] = lambda: workflow

    plan_response = client.post(
        "/tickets/plans",
        json={
            "actor_id": "demo_user_001",
            "message": "订单 A1001 已付款一周仍未发货，请帮我处理。",
        },
    )
    confirmation_id = plan_response.json()["confirmation"]["confirmation_id"]

    confirm_response = client.post(
        f"/tools/confirmations/{confirmation_id}/confirm",
        json={"actor_id": "demo_user_001"},
    )
    execute_response = client.post(
        f"/tickets/confirmations/{confirmation_id}/execute",
        json={"actor_id": "demo_user_001"},
    )

    assert plan_response.status_code == 200
    assert confirm_response.status_code == 200
    assert execute_response.status_code == 201
    assert execute_response.json()["ticket"]["ticket_id"] == "T1001"
    assert len(fake_creator.calls) == 1
