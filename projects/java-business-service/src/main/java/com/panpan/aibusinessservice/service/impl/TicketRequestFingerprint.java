package com.panpan.aibusinessservice.service.impl;

import com.panpan.aibusinessservice.dto.CreateTicketCommand;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

final class TicketRequestFingerprint {
    private TicketRequestFingerprint() {
    }

    static String from(CreateTicketCommand command, String requesterUserId, String tenantId) {
        StringBuilder source = new StringBuilder();
        append(source, requesterUserId);
        append(source, tenantId);
        append(source, command.title());
        append(source, command.description());
        append(source, command.category().code());
        append(source, command.priority().code());
        append(source, command.relatedOrderId());
        append(source, command.source());
        append(source, command.confirmationId());
        return sha256(source.toString());
    }

    private static void append(StringBuilder builder, String value) {
        builder.append(value == null ? "" : value).append('\n');
    }

    private static String sha256(String value) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] hash = digest.digest(value.getBytes(StandardCharsets.UTF_8));
            return toHex(hash);
        } catch (NoSuchAlgorithmException exception) {
            throw new IllegalStateException("SHA-256 is not available", exception);
        }
    }

    private static String toHex(byte[] bytes) {
        StringBuilder hex = new StringBuilder(bytes.length * 2);
        for (byte item : bytes) {
            hex.append(String.format("%02x", item));
        }
        return hex.toString();
    }
}
