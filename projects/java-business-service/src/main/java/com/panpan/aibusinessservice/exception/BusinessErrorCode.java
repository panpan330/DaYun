package com.panpan.aibusinessservice.exception;

import org.springframework.http.HttpStatus;

public enum BusinessErrorCode {
    TOOL_RATE_LIMITED(HttpStatus.TOO_MANY_REQUESTS, "工具调用过于频繁，请稍后再试。"),
    INTERNAL_AUTH_FAILED(HttpStatus.UNAUTHORIZED, "内部服务鉴权失败。"),
    ORDER_ID_INVALID(HttpStatus.UNPROCESSABLE_ENTITY, "订单号格式不正确。"),
    ORDER_NOT_FOUND(HttpStatus.NOT_FOUND, "订单不存在，请确认订单号是否正确。"),
    ORDER_ACCESS_DENIED(HttpStatus.FORBIDDEN, "当前用户无权查看或操作该订单。"),
    IDEMPOTENCY_KEY_REQUIRED(HttpStatus.BAD_REQUEST, "写操作缺少幂等键。"),
    INVALID_REQUEST_PARAM(HttpStatus.BAD_REQUEST, "请求参数不合法。"),
    IDEMPOTENCY_KEY_INVALID(HttpStatus.UNPROCESSABLE_ENTITY, "幂等键格式不正确。"),
    IDEMPOTENCY_KEY_CONFLICT(HttpStatus.CONFLICT, "同一个幂等键不能用于不同的请求参数。"),
    TICKET_REQUEST_INVALID(HttpStatus.UNPROCESSABLE_ENTITY, "工单请求参数不合法。"),
    TICKET_ALREADY_EXISTS(HttpStatus.CONFLICT, "已存在类似工单。"),
    ORDER_NOT_SUPPORT_TICKET(HttpStatus.CONFLICT, "当前订单不支持创建该类工单。"),
    ORDER_NOT_REFUNDABLE(HttpStatus.CONFLICT, "当前订单状态不支持退款。"),
    REFUND_ALREADY_EXISTS(HttpStatus.CONFLICT, "订单已退款，请勿重复操作。"),
    HANDOFF_STATE_CONFLICT(HttpStatus.CONFLICT, "转人工状态不允许该操作。"),
    HANDOFF_NOT_ASSIGNED(HttpStatus.FORBIDDEN, "请先接手该会话。"),
    REFUND_REASON_REQUIRED(HttpStatus.UNPROCESSABLE_ENTITY, "退款原因不能为空。"),
    REFUND_REASON_TOO_LONG(HttpStatus.UNPROCESSABLE_ENTITY, "退款原因不能超过 200 字。"),
    ORDER_NOT_CANCELABLE(HttpStatus.CONFLICT, "当前订单状态不支持取消。"),
    CANCEL_ALREADY_EXISTS(HttpStatus.CONFLICT, "订单已取消，请勿重复操作。"),
    CANCEL_REASON_REQUIRED(HttpStatus.UNPROCESSABLE_ENTITY, "取消原因不能为空。"),
    CANCEL_REASON_TOO_LONG(HttpStatus.UNPROCESSABLE_ENTITY, "取消原因不能超过 200 字。"),
    TICKET_NOT_FOUND(HttpStatus.NOT_FOUND, "工单不存在。"),
    TICKET_ACCESS_DENIED(HttpStatus.FORBIDDEN, "当前用户无权查看或处理该工单。"),
    TICKET_STATUS_TRANSITION_INVALID(HttpStatus.CONFLICT, "工单状态流转不合法。"),
    TICKET_ALREADY_ASSIGNED(HttpStatus.CONFLICT, "工单已有处理人。"),
    TICKET_ASSIGNEE_INVALID(HttpStatus.UNPROCESSABLE_ENTITY, "工单处理人不存在或不是可分配的客服人员。"),
    TICKET_MESSAGE_VISIBILITY_INVALID(HttpStatus.FORBIDDEN, "当前用户不能提交该可见范围的工单留言。"),
    TICKET_CUSTOMER_REPLY_NOT_ALLOWED(HttpStatus.CONFLICT, "当前工单状态不允许用户继续补充信息。"),
    TICKET_REOPEN_NOT_ALLOWED(HttpStatus.CONFLICT, "只有已解决的工单可以由用户申请重开。"),
    LOGIN_REQUEST_INVALID(HttpStatus.UNPROCESSABLE_ENTITY, "登录请求参数不合法。"),
    LOGIN_FAILED(HttpStatus.UNAUTHORIZED, "用户名或密码不正确。"),
    AUTH_REQUIRED(HttpStatus.UNAUTHORIZED, "请先登录。"),
    USER_NOT_FOUND(HttpStatus.NOT_FOUND, "用户不存在。"),
    AI_FEEDBACK_ACCESS_DENIED(HttpStatus.FORBIDDEN, "当前账号无权查看 AI 反馈数据。"),
    AI_FEEDBACK_NOT_FOUND(HttpStatus.NOT_FOUND, "AI 反馈记录不存在。"),
    DOCUMENT_ID_REQUIRED(HttpStatus.UNPROCESSABLE_ENTITY, "知识文档 ID 不能为空。"),
    DOCUMENT_TITLE_REQUIRED(HttpStatus.UNPROCESSABLE_ENTITY, "知识文档标题不能为空。");

    private final HttpStatus status;
    private final String defaultMessage;

    BusinessErrorCode(HttpStatus status, String defaultMessage) {
        this.status = status;
        this.defaultMessage = defaultMessage;
    }

    public HttpStatus status() {
        return status;
    }

    public String defaultMessage() {
        return defaultMessage;
    }
}
