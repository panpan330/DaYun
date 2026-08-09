package com.panpan.aibusinessservice;

import static com.panpan.aibusinessservice.InternalApiTestSupport.TRACE_ID;
import static com.panpan.aibusinessservice.InternalApiTestSupport.withInternalHeaders;
import static org.assertj.core.api.Assertions.assertThat;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.delete;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.header;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import com.panpan.aibusinessservice.common.trace.TraceHeaders;
import com.panpan.aibusinessservice.entity.KnowledgeDocument;
import com.panpan.aibusinessservice.mapper.KnowledgeDocumentMapper;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.transaction.annotation.Transactional;

@SpringBootTest
@AutoConfigureMockMvc
@Transactional
class InternalKnowledgeDocumentControllerTest {
    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private KnowledgeDocumentMapper mapper;

    private static String payload(String documentId, int chunkCount) {
        return """
                {"document_id":"%s","title":"Test Doc","doc_type":"policy",
                 "business_domain":"refund","permission_group":"public",
                 "status":"enabled","source_file_name":"%s.md",
                 "chunk_count":%d,"updated_by":"U1001"}
                """.formatted(documentId, documentId, chunkCount);
    }

    @Test
    void upsertCreatesDocument() throws Exception {
        mockMvc.perform(withInternalHeaders(post("/internal/knowledge-documents"))
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(payload("doc-001", 3)))
                .andExpect(status().isOk())
                .andExpect(header().string(TraceHeaders.TRACE_ID, TRACE_ID))
                .andExpect(jsonPath("$.success").value(true))
                .andExpect(jsonPath("$.data.document_id").value("doc-001"))
                .andExpect(jsonPath("$.data.chunk_count").value(3));
    }

    @Test
    void upsertUpdatesExistingDocument() throws Exception {
        mockMvc.perform(withInternalHeaders(post("/internal/knowledge-documents"))
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(payload("doc-002", 3)))
                .andExpect(status().isOk());

        mockMvc.perform(withInternalHeaders(post("/internal/knowledge-documents"))
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(payload("doc-002", 5)))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.data.chunk_count").value(5));

        KnowledgeDocument loaded = mapper.selectByTenantIdAndDocumentId("default", "doc-002");
        assertThat(loaded).isNotNull();
        assertThat(loaded.getChunkCount()).isEqualTo(5);
    }

    @Test
    void deleteRemovesDocument() throws Exception {
        mockMvc.perform(withInternalHeaders(post("/internal/knowledge-documents"))
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(payload("doc-003", 3)))
                .andExpect(status().isOk());

        mockMvc.perform(withInternalHeaders(delete("/internal/knowledge-documents/doc-003")))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.success").value(true))
                .andExpect(jsonPath("$.data").value(true));

        assertThat(mapper.selectByTenantIdAndDocumentId("default", "doc-003")).isNull();
    }

    @Test
    void listReturnsAllDocuments() throws Exception {
        mockMvc.perform(withInternalHeaders(post("/internal/knowledge-documents"))
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(payload("doc-list-001", 2)))
                .andExpect(status().isOk());

        mockMvc.perform(withInternalHeaders(get("/internal/knowledge-documents")))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.success").value(true))
                .andExpect(jsonPath("$.data[?(@.document_id == 'doc-list-001')]").exists());
    }

    @Test
    void rejectsMissingInternalToken() throws Exception {
        mockMvc.perform(post("/internal/knowledge-documents")
                        .header(TraceHeaders.TRACE_ID, TRACE_ID)
                        .header(TraceHeaders.CALLER, "ai-service")
                        .header(TraceHeaders.USER_ID, "U1001")
                        .header(TraceHeaders.TENANT_ID, "default")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{}"))
                .andExpect(status().isUnauthorized())
                .andExpect(header().string(TraceHeaders.TRACE_ID, TRACE_ID))
                .andExpect(jsonPath("$.success").value(false))
                .andExpect(jsonPath("$.code").value("INTERNAL_AUTH_FAILED"));
    }
}
