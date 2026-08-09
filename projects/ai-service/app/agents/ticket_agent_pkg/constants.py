import re

from .state import TicketIssueType, TicketUrgencyLevel
from app.schemas.ticket import TicketCategory, TicketPriority
POLICY_KEYWORDS = (
    "规则",
    "政策",
    "faq",
    "退款规则",
    "退货规则",
    "售后政策",
    "账号安全",
    "异常登录",
    "身份验证",
    "会员积分",
    "积分",
    "兑换礼品",
    "怎么退款",
    "怎么退货",
    "怎么申请退款",
    "如何申请退款",
    "多久可以退款",
    "多久可以退货",
    "多久到账",
    "怎么取消订单",
    "如何取消订单",
    "取消政策",
    "取消规则",
    "取消多久",
    "取消条件",
)
ORDER_KEYWORDS = (
    "订单",
    "物流",
    "快递",
    "发货",
    "到哪",
    "到哪了",
    "支付",
    "付款",
    "签收",
)
TICKET_KEYWORDS = (
    "投诉",
    "工单",
    "售后处理",
    "人工处理",
    "人工客服",
    "创建工单",
    "商品坏了",
    "商品破损",
    "不发货",
    "一直不动",
    "帮我处理",
)
SMALLTALK_KEYWORDS = (
    "你好",
    "您好",
    "hello",
    "hi",
    "你是谁",
    "你能做什么",
)
UNSUPPORTED_KEYWORDS = (
    "黑客",
    "攻击脚本",
    "写小说",
    "股票",
    "天气",
    "忽略之前",
    "忽略所有规则",
    "系统提示词",
    "内部工具",
    "内部工具配置",
    "api key",
    "api_key",
)
# 明确要求执行取消订单的动词短语：命中即视为 cancel_request，而不是政策咨询
# 或退款请求。cancel 关键词优先级高于 refund（"取消订单"/"退单" 不含退款语义）。
CANCEL_ACTION_PHRASES = (
    "取消订单",
    "申请取消",
    "直接取消",
    "立刻取消",
    "马上取消",
    "帮我取消",
    "我要取消",
    "想取消",
    "要求取消",
    "给我取消",
    "请取消",
    "请帮我取消",
    "取消购物",
    "退单",
    "办理取消",
)
# 取消关键词 + 具体订单号：如 "取消订单 A1002"、"取消 A1002"。
CANCEL_KEYWORDS = ("取消订单", "退单", "取消购物")
# 单独动作表达：如 "取消 A1002"、"取消一下 A1002"——"取消" 后直接跟订单号。
CANCEL_WITH_ORDER_PATTERN = re.compile(
    r"取消(?:一下|掉|了)?\s*(?:订单|单|购物)?\s*[:：#-]?\s*(?:[A-Za-z][A-Za-z0-9_-]{3,}|\d{4,})",
    re.IGNORECASE,
)
# 疑问句式：用户是在咨询取消政策/流程，而不是要执行取消。命中后不判
# cancel_request，留给 policy_question 分支回答（如 "怎么取消订单"、
# "取消多久生效"、"取消需要什么条件"）。
CANCEL_QUERY_WORDS = ("怎么", "如何", "多久", "什么", "条件", "吗", "呢", "为什么")
# 明确要求执行退款的动词短语：命中即视为 refund_request，而不是政策咨询。
REFUND_ACTION_PHRASES = (
    "申请退款",
    "直接退款",
    "立刻退款",
    "马上退款",
    "帮我退款",
    "我要退款",
    "想退款",
    "要求退款",
    "给我退款",
    "请退款",
    "请帮我退款",
    "退款到账",
    "退货款",
    "退货退款",
    "退钱",
    "退单",
    "办理退款",
    "帮我退",
    "帮我退货",
)
# 退款关键词 + 具体订单号：如 "订单 A1002 我要退款"、"A1002 退钱"。
REFUND_KEYWORDS = ("退款", "退钱", "退货款", "退货退款")
# 单独动作表达：如 "退 A1002 的款"、"退一下 A1002"、"退A1002"——"退" 后直接跟
# 订单号。注意不能依赖 \b：Python re 的 \w 在 Unicode 模式下包含中文，中文字符
# 与 ASCII 字母之间不构成单词边界，会漏判 "退A1002"；改用显式订单号字符类。
# 同时排除 "退货 A1002" 这类政策句（"退" 后不允许跨过"货"字再接订单号）。
REFUND_WITH_ORDER_PATTERN = re.compile(
    r"退(?:一下|掉|了)?\s*(?:订单|单)?\s*[:：#-]?\s*(?:[A-Za-z][A-Za-z0-9_-]{3,}|\d{4,})",
    re.IGNORECASE,
)
# 疑问句式：用户是在咨询退款政策/流程，而不是要执行退款。命中后不判
# refund_request，留给 policy_question 分支回答（如 "怎么申请退款"、
# "退款多久到账"、"申请退款需要什么条件"）。
REFUND_QUERY_WORDS = ("怎么", "如何", "多久", "什么", "条件", "吗", "呢", "为什么")
UNCLEAR_MESSAGES = (
    "有问题",
    "帮我看看",
    "这个怎么办",
    "处理一下",
)
ORDER_ID_PATTERN = re.compile(
    r"(?:订单号?|order(?:_id)?)\s*[:：#-]?\s*([A-Za-z0-9_-]{3,64})",
    re.IGNORECASE,
)
FALLBACK_ORDER_ID_PATTERN = re.compile(r"\b([A-Za-z]\d{3,}|\d{4,})\b")
REFUND_ISSUE_KEYWORDS = ("退款", "退货", "售后")
LOGISTICS_ISSUE_KEYWORDS = ("物流", "快递", "发货", "未发货", "不发货", "一直不动", "到哪")
COMPLAINT_ISSUE_KEYWORDS = (
    "投诉",
    "人工处理",
    "人工客服",
    "帮我处理",
    "商品坏了",
    "商品破损",
    "破损",
)
HIGH_URGENCY_KEYWORDS = (
    "破损",
    "坏了",
    "一直不动",
    "一周",
    "加急",
    "着急",
    "催一下",
    "立刻",
    "马上",
)
ORDER_REQUIRED_ISSUE_TYPES: tuple[TicketIssueType, ...] = (
    "refund",
    "logistics",
    "complaint",
)
MISSING_TICKET_FIELD_QUESTIONS: dict[str, str] = {
    "order_id": "请补充相关订单号（例如 1001 或 A1001），这样我才能继续为你整理工单。",
    "issue_type": "请说明这是退款、物流、投诉，还是其他需要人工处理的问题。",
    "description": "请补充问题的具体描述，例如发生了什么、影响是什么。",
    "user_request": "请说明你希望客服帮你处理什么，例如投诉处理、退款处理或人工解释。",
    "reason": "请补充退款原因，例如商品质量问题或不想要了。",
}
TICKET_ISSUE_TYPE_LABELS: dict[TicketIssueType, str] = {
    "cancel": "取消订单",
    "refund": "退款/退货",
    "logistics": "物流/发货",
    "complaint": "投诉/异常处理",
    "policy_gap": "知识库缺口",
    "unknown": "未确定",
}
TICKET_URGENCY_LABELS: dict[TicketUrgencyLevel, str] = {
    "low": "低",
    "normal": "普通",
    "high": "高",
}
TICKET_ISSUE_TYPE_TO_CATEGORY: dict[TicketIssueType, TicketCategory] = {
    # 取消订单通常走 cancel_order 工具链路；ticket_request 意图 + LLM 抽取
    # issue_type=cancel 的边缘路径会创建人工工单，TicketCategory 无 cancel 值，
    # 归入最接近的"投诉/异常处理"人工工单类别。
    "cancel": TicketCategory.COMPLAINT,
    "refund": TicketCategory.REFUND,
    "logistics": TicketCategory.LOGISTICS,
    "complaint": TicketCategory.COMPLAINT,
    "policy_gap": TicketCategory.POLICY_GAP,
}
TICKET_URGENCY_TO_PRIORITY: dict[TicketUrgencyLevel, TicketPriority] = {
    "low": TicketPriority.LOW,
    "normal": TicketPriority.NORMAL,
    "high": TicketPriority.HIGH,
}
ORDER_STATUS_LABELS: dict[str, str] = {
    "waiting_shipment": "待发货",
    "shipped": "已发货",
    "delivered": "已签收",
    "canceled": "已取消",
}
PAYMENT_STATUS_LABELS: dict[str, str] = {
    "unpaid": "未支付",
    "paid": "已支付",
    "refunded": "已退款",
}
DEFAULT_TICKET_ACTOR_ID = "demo_user_001"
CREATE_TICKET_TOOL_NAME = "create_ticket"
TICKET_CONFIRMATION_NOT_FOUND_MESSAGE = "当前会话没有待确认工单，请先发起工单流程。"
TICKET_CONFIRMATION_INTERRUPT_NOT_FOUND_MESSAGE = "当前执行结果里没有待处理的工单确认中断。"
TICKET_CONFIRMATION_REJECTED_MESSAGE = "已取消创建工单；如需创建，请重新发起工单流程。"
TICKET_CONFIRMATION_INTERRUPT_KIND = "ticket_confirmation"
REFUND_CONFIRMATION_REJECTED_MESSAGE = "已取消退款申请；如需退款，请重新发起退款流程。"
REFUND_FIELDS_NOT_FOUND_MESSAGE = "没有找到可执行退款的确认字段，请重新发起退款流程。"
REFUND_CONFIRMATION_REQUIRED_MESSAGE = "执行退款前需要先得到用户确认。"
REFUND_UNEXPECTED_ERROR_CODE = "REFUND_UNEXPECTED_ERROR"
REFUND_UNEXPECTED_ERROR_MESSAGE = "执行退款时遇到异常，请稍后重试或联系人工客服。"
REFUND_REASON_MAX_LENGTH = 200
CANCEL_CONFIRMATION_REJECTED_MESSAGE = "已取消本次取消订单申请；如需取消订单，请重新发起取消流程。"
CANCEL_FIELDS_NOT_FOUND_MESSAGE = "没有找到可执行取消的确认字段，请重新发起取消流程。"
CANCEL_CONFIRMATION_REQUIRED_MESSAGE = "执行取消订单前需要先得到用户确认。"
CANCEL_UNEXPECTED_ERROR_CODE = "CANCEL_UNEXPECTED_ERROR"
CANCEL_UNEXPECTED_ERROR_MESSAGE = "执行取消订单时遇到异常，请稍后重试或联系人工客服。"
CANCEL_REASON_MAX_LENGTH = 200
TICKET_AGENT_FALLBACK_ERROR_CODE = "TICKET_AGENT_UNEXPECTED_ERROR"
TICKET_AGENT_FALLBACK_MESSAGE = "智能工单流程暂时遇到异常，请稍后重试或联系人工客服。"
TICKET_ORDER_QUERY_MISSING_ORDER_ID_MESSAGE = (
    "请提供要查询的订单号（例如 A1001 或 1001），我拿到订单号后才能查询订单状态和物流信息。"
)
TICKET_ORDER_QUERY_ARGUMENT_VALIDATION_ERROR_CODE = "TOOL_ARGUMENTS_VALIDATION_FAILED"
TICKET_ORDER_QUERY_ARGUMENT_VALIDATION_MESSAGE = (
    "订单号格式不符合查询工具要求，请提供清晰的订单号。"
)
TICKET_ORDER_QUERY_UNEXPECTED_ERROR_CODE = "TOOL_CALL_FAILED"
TICKET_ORDER_QUERY_UNEXPECTED_ERROR_MESSAGE = (
    "订单查询工具调用失败，请稍后重试或联系人工客服。"
)
TICKET_ORDER_QUERY_RESULT_VALIDATION_MESSAGE = (
    "订单查询服务返回的数据暂时无法处理，请稍后重试或联系人工客服。"
)
TICKET_ORDER_QUERY_NOT_FOUND_CODES = frozenset({"ORDER_NOT_FOUND"})
TICKET_ORDER_QUERY_TIMEOUT_CODES = frozenset({"TOOL_TIMEOUT"})
TICKET_ORDER_QUERY_UPSTREAM_ERROR_CODES = frozenset({"TOOL_UPSTREAM_ERROR"})
TICKET_ORDER_QUERY_RESULT_VALIDATION_FAILED_CODES = frozenset(
    {"TOOL_RESULT_VALIDATION_FAILED"}
)
TICKET_ORDER_QUERY_TOOL_ERROR_CODES = frozenset({"TOOL_CALL_FAILED"})
TICKET_CREATION_UNEXPECTED_ERROR_CODE = "TICKET_CREATION_UNEXPECTED_ERROR"
TICKET_CREATION_UNEXPECTED_ERROR_MESSAGE = "创建工单时遇到异常，请稍后重试或联系人工客服。"
TICKET_THREAD_ID_INVALID_ERROR_CODE = "TICKET_THREAD_ID_INVALID"
TICKET_AGENT_LOG_VALUE_EMPTY = "-"
TICKET_AGENT_MODEL_EMPTY_RESPONSE_CODES = frozenset(
    {
        "LLM_EMPTY_RESPONSE",
        "TICKET_INTENT_LLM_EMPTY_RESPONSE",
        "TICKET_FIELD_LLM_EMPTY_RESPONSE",
    }
)
TICKET_AGENT_MODEL_SCHEMA_VALIDATION_FAILED_CODES = frozenset(
    {
        "TICKET_INTENT_LLM_VALIDATION_FAILED",
        "TICKET_FIELD_LLM_VALIDATION_FAILED",
    }
)
TICKET_AGENT_MODEL_TRANSIENT_PROVIDER_ERROR_CODES = frozenset(
    {
        "LLM_TIMEOUT",
        "LLM_RATE_LIMITED",
        "LLM_PROVIDER_ERROR",
        "LLM_CONNECTION_ERROR",
        "LLM_PROVIDER_STATUS_ERROR",
        "LLM_BAD_RESPONSE",
        "LLM_CALL_FAILED",
    }
)
TICKET_AGENT_MODEL_CONFIGURATION_ERROR_CODES = frozenset(
    {
        "LLM_API_KEY_MISSING",
        "LLM_AUTHENTICATION_FAILED",
        "LLM_PERMISSION_DENIED",
        "LLM_RESOURCE_NOT_FOUND",
        "LLM_BAD_REQUEST",
    }
)


