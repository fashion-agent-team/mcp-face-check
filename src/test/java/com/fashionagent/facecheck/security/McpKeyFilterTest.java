package com.fashionagent.facecheck.security;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;
import org.springframework.mock.web.MockFilterChain;
import org.springframework.mock.web.MockHttpServletRequest;
import org.springframework.mock.web.MockHttpServletResponse;

class McpKeyFilterTest {

    static MockHttpServletResponse run(McpKeyFilter filter, String path, String key) throws Exception {
        MockHttpServletRequest req = new MockHttpServletRequest("POST", path);
        if (key != null) {
            req.addHeader(McpKeyFilter.HEADER, key);
        }
        MockHttpServletResponse res = new MockHttpServletResponse();
        filter.doFilter(req, res, new MockFilterChain());
        return res;
    }

    @Test
    void 키를_넣으면_mcp는_같은_키만_받는다() throws Exception {
        McpKeyFilter filter = new McpKeyFilter("inner-key-1");
        assertThat(run(filter, "/mcp", null).getStatus()).isEqualTo(401);
        assertThat(run(filter, "/mcp", "wrong").getStatus()).isEqualTo(401);
        assertThat(run(filter, "/mcp", "inner-key-1").getStatus()).isEqualTo(200);
        // health는 키 없이도
        assertThat(run(filter, "/actuator/health", null).getStatus()).isEqualTo(200);
    }

    @Test
    void 키를_비우면_예전처럼_다_받는다() throws Exception {
        assertThat(run(new McpKeyFilter(""), "/mcp", null).getStatus()).isEqualTo(200);
    }
}
