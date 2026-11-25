package com.fashionagent.facecheck.security;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;

/**
 * MCP 입구(/mcp) 내부 호출 키(2026-11-25 보안 점검).
 *
 * <p>MCP_SHARED_KEY를 넣으면 /mcp 요청은 X-Mcp-Key 헤더가 같은 값이어야 받습니다(agent가 같은 값을 보냄).
 * 비우면 예전처럼 누구나 받습니다(키를 아직 안 넣은 서버와 함께 돌 수 있게). health 등 다른 경로는 건드리지 않습니다.
 */
@Component
public class McpKeyFilter extends OncePerRequestFilter {

    public static final String HEADER = "X-Mcp-Key";

    private final byte[] key;

    public McpKeyFilter(@Value("${MCP_SHARED_KEY:}") String key) {
        this.key = key == null || key.isBlank() ? null : key.strip().getBytes(StandardCharsets.UTF_8);
    }

    @Override
    protected boolean shouldNotFilter(HttpServletRequest request) {
        String path = request.getRequestURI();
        return key == null || !(path.equals("/mcp") || path.startsWith("/mcp/"));
    }

    @Override
    protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response, FilterChain chain)
            throws ServletException, IOException {
        String given = request.getHeader(HEADER);
        if (given == null || !MessageDigest.isEqual(key, given.strip().getBytes(StandardCharsets.UTF_8))) {
            response.setStatus(HttpServletResponse.SC_UNAUTHORIZED);
            response.setContentType("application/json;charset=UTF-8");
            response.getWriter().write("{\"code\":\"unauthorized\",\"message\":\"내부 호출 키가 없거나 틀려요.\"}");
            return;
        }
        chain.doFilter(request, response);
    }
}
