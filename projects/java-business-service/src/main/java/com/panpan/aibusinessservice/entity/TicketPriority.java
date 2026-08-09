package com.panpan.aibusinessservice.entity;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonValue;

public enum TicketPriority {
    LOW("low"),
    NORMAL("normal"),
    HIGH("high");

    private final String code;

    TicketPriority(String code) {
        this.code = code;
    }

    @JsonValue
    public String code() {
        return code;
    }

    @JsonCreator
    public static TicketPriority fromCode(String rawCode) {
        for (TicketPriority priority : values()) {
            if (priority.code.equals(rawCode)) {
                return priority;
            }
        }
        throw new IllegalArgumentException("Unsupported ticket priority: " + rawCode);
    }
}
