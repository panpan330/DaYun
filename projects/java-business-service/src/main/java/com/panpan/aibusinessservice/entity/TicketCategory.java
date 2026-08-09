package com.panpan.aibusinessservice.entity;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonValue;

public enum TicketCategory {
    REFUND("refund"),
    ORDER_QUERY("order_query"),
    LOGISTICS("logistics"),
    COMPLAINT("complaint"),
    POLICY_GAP("policy_gap");

    private final String code;

    TicketCategory(String code) {
        this.code = code;
    }

    @JsonValue
    public String code() {
        return code;
    }

    @JsonCreator
    public static TicketCategory fromCode(String rawCode) {
        for (TicketCategory category : values()) {
            if (category.code.equals(rawCode)) {
                return category;
            }
        }
        throw new IllegalArgumentException("Unsupported ticket category: " + rawCode);
    }
}
