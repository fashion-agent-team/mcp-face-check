"""mcp-face-check 서버의 /mcp에 붙는 아주 작은 클라이언트(표준 라이브러리만). 학습 측정값을 서버와 같은 코드로 얻으려고 씀."""
import json
import urllib.request


class Mcp:
    def __init__(self, url="http://localhost:8094/mcp"):
        self.url, self.sid, self.n = url, None, 0
        self._post({"jsonrpc": "2.0", "id": self._id(), "method": "initialize",
                    "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                               "clientInfo": {"name": "training", "version": "1"}}})
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def _id(self):
        self.n += 1
        return self.n

    def _post(self, body):
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        if self.sid:
            headers["Mcp-Session-Id"] = self.sid
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), method="POST", headers=headers)
        with urllib.request.urlopen(req, timeout=60) as r:
            self.sid = r.headers.get("Mcp-Session-Id") or self.sid
            raw = r.read().decode()
        if not raw.strip():
            return None
        if raw.lstrip().startswith("{"):
            return json.loads(raw)
        for line in raw.splitlines():
            if line.startswith("data:"):
                return json.loads(line[5:])

    def call(self, tool, args):
        r = self._post({"jsonrpc": "2.0", "id": self._id(), "method": "tools/call",
                        "params": {"name": tool, "arguments": args}})
        return json.loads(r["result"]["content"][0]["text"])
