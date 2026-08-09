package com.panpan.aibusinessservice;

import static org.assertj.core.api.Assertions.assertThat;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.header;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import com.panpan.aibusinessservice.common.trace.TraceHeaders;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.system.CapturedOutput;
import org.springframework.boot.test.system.OutputCaptureExtension;
import org.springframework.test.web.servlet.MockMvc;

@SpringBootTest
@AutoConfigureMockMvc
@ExtendWith(OutputCaptureExtension.class)
class TraceFilterLoggingTest {
    @Autowired
    private MockMvc mockMvc;

    @Test
    void traceFilterLogsRequestLifecycleWithTraceId(CapturedOutput output) throws Exception {
        mockMvc.perform(get("/health").header(TraceHeaders.TRACE_ID, "trace-java-log-001"))
                .andExpect(status().isOk())
                .andExpect(header().string(TraceHeaders.TRACE_ID, "trace-java-log-001"));

        assertThat(output)
                .contains("trace_id=trace-java-log-001")
                .contains("java_request_started trace_id=trace-java-log-001 method=GET path=/health")
                .contains("java_request_finished trace_id=trace-java-log-001 method=GET path=/health status_code=200");
    }
}
