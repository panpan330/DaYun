package com.panpan.aibusinessservice.config;

import jakarta.annotation.PostConstruct;
import java.sql.Connection;
import java.sql.DatabaseMetaData;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.List;
import javax.sql.DataSource;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Component;

/** Adds feedback-review columns for an existing local database without vendor-specific SQL syntax. */
@Component
public class AiFeedbackSchemaMigration {
    private record Column(String name, String definition) {}

    private static final List<Column> REQUIRED_COLUMNS = List.of(
            new Column("user_message_excerpt", "VARCHAR(1000) NULL"),
            new Column("assistant_answer_excerpt", "VARCHAR(2000) NULL"),
            new Column("citation_summary_json", "TEXT NULL"),
            new Column("review_status", "VARCHAR(32) NOT NULL DEFAULT 'candidate'"),
            new Column("bad_case_id", "VARCHAR(160) NULL"),
            new Column("reviewed_by_user_id", "VARCHAR(64) NULL"),
            new Column("reviewed_at", "DATETIME(6) NULL"),
            new Column("review_note", "VARCHAR(1000) NULL")
    );

    private final DataSource dataSource;
    private final JdbcTemplate jdbcTemplate;

    public AiFeedbackSchemaMigration(DataSource dataSource, JdbcTemplate jdbcTemplate) {
        this.dataSource = dataSource;
        this.jdbcTemplate = jdbcTemplate;
    }

    @PostConstruct
    void migrate() {
        for (Column column : REQUIRED_COLUMNS) {
            if (!columnExists(column.name())) {
                jdbcTemplate.execute("ALTER TABLE ai_response_feedback ADD COLUMN " + column.name() + " " + column.definition());
            }
        }
    }

    private boolean columnExists(String columnName) {
        try (Connection connection = dataSource.getConnection()) {
            DatabaseMetaData metadata = connection.getMetaData();
            return exists(metadata, "ai_response_feedback", columnName)
                    || exists(metadata, "AI_RESPONSE_FEEDBACK", columnName);
        } catch (SQLException exception) {
            throw new IllegalStateException("Failed to inspect AI feedback schema", exception);
        }
    }

    private boolean exists(DatabaseMetaData metadata, String tableName, String columnName) throws SQLException {
        try (ResultSet columns = metadata.getColumns(null, null, tableName, columnName)) {
            return columns.next();
        }
    }
}
