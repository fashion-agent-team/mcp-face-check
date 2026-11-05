package com.fashionagent.facecheck.teeth;

import com.fashionagent.facecheck.teeth.TeethClassifier.TeethFinding;

/** teeth_check 결과 (앱 레포 docs/contracts/mcp-tools.md). framing이나 dark면 teeth는 null. */
public record TeethResult(String framing, String lighting, TeethFinding teeth, String source) {
}
