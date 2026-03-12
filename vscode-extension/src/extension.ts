import * as vscode from "vscode";

// ── Types ────────────────────────────────────────────────────────────────────

interface SecurityIssue {
  type: string;
  severity: "critical" | "high" | "medium" | "low" | "info";
  message: string;
  line: number;
  column?: number;
  endLine?: number;
  endColumn?: number;
  suggestion?: string;
}

interface VerifyRequest {
  code: string;
  language: string;
  filename?: string;
}

interface VerifyResponse {
  overall_score: number;
  security_score: number;
  reliability_score: number;
  issues: SecurityIssue[];
  summary: string;
  error?: string;
}

// ── ReasonForge API Client ───────────────────────────────────────────────────

class ReasonForgeClient {
  private serverUrl: string;
  private timeoutMs: number;

  constructor(serverUrl: string, timeoutSeconds: number) {
    this.serverUrl = serverUrl.replace(/\/+$/, "");
    this.timeoutMs = timeoutSeconds * 1000;
  }

  async verify(request: VerifyRequest): Promise<VerifyResponse> {
    const url = `${this.serverUrl}/verify`;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);

    try {
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(request),
        signal: controller.signal,
      });

      if (!response.ok) {
        const body = await response.text().catch(() => "");
        throw new Error(
          `Server returned ${response.status}: ${body || response.statusText}`
        );
      }

      return (await response.json()) as VerifyResponse;
    } catch (err: unknown) {
      if (err instanceof DOMException && err.name === "AbortError") {
        throw new Error(
          `Request timed out after ${this.timeoutMs / 1000} seconds`
        );
      }
      throw err;
    } finally {
      clearTimeout(timer);
    }
  }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function mapSeverity(
  severity: SecurityIssue["severity"]
): vscode.DiagnosticSeverity {
  switch (severity) {
    case "critical":
    case "high":
      return vscode.DiagnosticSeverity.Error;
    case "medium":
      return vscode.DiagnosticSeverity.Warning;
    case "low":
      return vscode.DiagnosticSeverity.Information;
    case "info":
      return vscode.DiagnosticSeverity.Hint;
    default:
      return vscode.DiagnosticSeverity.Warning;
  }
}

function languageIdFromFilename(filename: string): string {
  const ext = filename.split(".").pop()?.toLowerCase() ?? "";
  const map: Record<string, string> = {
    ts: "typescript",
    tsx: "typescriptreact",
    js: "javascript",
    jsx: "javascriptreact",
    py: "python",
    rs: "rust",
    go: "go",
    java: "java",
    c: "c",
    cpp: "cpp",
    cs: "csharp",
    rb: "ruby",
    php: "php",
    swift: "swift",
    kt: "kotlin",
  };
  return map[ext] ?? ext;
}

// ── Extension lifecycle ──────────────────────────────────────────────────────

let outputChannel: vscode.OutputChannel;
let diagnosticCollection: vscode.DiagnosticCollection;

export function activate(context: vscode.ExtensionContext): void {
  outputChannel = vscode.window.createOutputChannel("ReasonForge");
  diagnosticCollection =
    vscode.languages.createDiagnosticCollection("reasonforge");

  context.subscriptions.push(outputChannel, diagnosticCollection);

  // ── Verify Current File ──────────────────────────────────────────────────
  const verifyFileCmd = vscode.commands.registerCommand(
    "reasonforge.verifyFile",
    async () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        vscode.window.showInformationMessage(
          "ReasonForge: No active editor found."
        );
        return;
      }

      const document = editor.document;
      const code = document.getText();
      if (!code.trim()) {
        vscode.window.showInformationMessage(
          "ReasonForge: The current file is empty."
        );
        return;
      }

      await runVerification(
        code,
        document.languageId,
        document.fileName,
        document.uri
      );
    }
  );

  // ── Verify Selection ─────────────────────────────────────────────────────
  const verifySelectionCmd = vscode.commands.registerCommand(
    "reasonforge.verifySelection",
    async () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        vscode.window.showInformationMessage(
          "ReasonForge: No active editor found."
        );
        return;
      }

      const selection = editor.selection;
      if (selection.isEmpty) {
        vscode.window.showInformationMessage(
          "ReasonForge: No text selected. Select code to verify."
        );
        return;
      }

      const code = editor.document.getText(selection);
      await runVerification(
        code,
        editor.document.languageId,
        editor.document.fileName,
        editor.document.uri,
        selection.start.line
      );
    }
  );

  context.subscriptions.push(verifyFileCmd, verifySelectionCmd);

  outputChannel.appendLine("ReasonForge extension activated.");
}

export function deactivate(): void {
  if (diagnosticCollection) {
    diagnosticCollection.clear();
    diagnosticCollection.dispose();
  }
  if (outputChannel) {
    outputChannel.dispose();
  }
}

// ── Core verification logic ──────────────────────────────────────────────────

async function runVerification(
  code: string,
  languageId: string,
  filename: string,
  uri: vscode.Uri,
  lineOffset: number = 0
): Promise<void> {
  const config = vscode.workspace.getConfiguration("reasonforge");
  const serverUrl = config.get<string>("serverUrl", "http://localhost:8000");
  const timeout = config.get<number>("timeout", 30);
  const showSecurityWarnings = config.get<boolean>(
    "showSecurityWarnings",
    true
  );

  const client = new ReasonForgeClient(serverUrl, timeout);
  const language = languageId || languageIdFromFilename(filename);

  outputChannel.clear();
  outputChannel.appendLine(`Verifying: ${filename}`);
  outputChannel.appendLine(`Language:  ${language}`);
  outputChannel.appendLine(`Server:    ${serverUrl}`);
  outputChannel.appendLine("─".repeat(60));
  outputChannel.show(true);

  await vscode.window.withProgress(
    {
      location: vscode.ProgressLocation.Notification,
      title: "ReasonForge: Verifying code...",
      cancellable: false,
    },
    async () => {
      try {
        const result = await client.verify({
          code,
          language,
          filename,
        });

        // ── Display summary in output channel ──────────────────────────────
        outputChannel.appendLine("");
        outputChannel.appendLine("VERIFICATION RESULTS");
        outputChannel.appendLine("─".repeat(60));
        outputChannel.appendLine(
          `Overall Score:     ${result.overall_score}/100`
        );
        outputChannel.appendLine(
          `Security Score:    ${result.security_score}/100`
        );
        outputChannel.appendLine(
          `Reliability Score: ${result.reliability_score}/100`
        );
        outputChannel.appendLine("");
        outputChannel.appendLine(`Summary: ${result.summary}`);

        if (result.issues.length > 0) {
          outputChannel.appendLine("");
          outputChannel.appendLine(`Issues found: ${result.issues.length}`);
          outputChannel.appendLine("─".repeat(60));

          for (const issue of result.issues) {
            outputChannel.appendLine(
              `  [${issue.severity.toUpperCase()}] Line ${issue.line}: ${issue.message}`
            );
            if (issue.suggestion) {
              outputChannel.appendLine(
                `    Suggestion: ${issue.suggestion}`
              );
            }
          }
        } else {
          outputChannel.appendLine("");
          outputChannel.appendLine("No issues found.");
        }

        // ── Map issues to VS Code diagnostics ─────────────────────────────
        if (showSecurityWarnings) {
          const diagnostics: vscode.Diagnostic[] = result.issues.map(
            (issue) => {
              const startLine = Math.max(
                0,
                (issue.line ?? 1) - 1 + lineOffset
              );
              const startCol = Math.max(0, (issue.column ?? 1) - 1);
              const endLine = issue.endLine
                ? issue.endLine - 1 + lineOffset
                : startLine;
              const endCol = issue.endColumn
                ? issue.endColumn - 1
                : Number.MAX_SAFE_INTEGER;

              const range = new vscode.Range(
                startLine,
                startCol,
                endLine,
                endCol
              );

              const diag = new vscode.Diagnostic(
                range,
                issue.message,
                mapSeverity(issue.severity)
              );
              diag.source = "ReasonForge";
              diag.code = issue.type;

              return diag;
            }
          );

          diagnosticCollection.set(uri, diagnostics);
        }

        // ── Notification summary ───────────────────────────────────────────
        const criticalCount = result.issues.filter(
          (i) => i.severity === "critical" || i.severity === "high"
        ).length;

        if (criticalCount > 0) {
          vscode.window.showWarningMessage(
            `ReasonForge: Score ${result.overall_score}/100 — ${criticalCount} critical/high issue(s) found.`
          );
        } else if (result.issues.length > 0) {
          vscode.window.showInformationMessage(
            `ReasonForge: Score ${result.overall_score}/100 — ${result.issues.length} issue(s) found.`
          );
        } else {
          vscode.window.showInformationMessage(
            `ReasonForge: Score ${result.overall_score}/100 — No issues found.`
          );
        }
      } catch (err: unknown) {
        const message =
          err instanceof Error ? err.message : String(err);
        outputChannel.appendLine("");
        outputChannel.appendLine(`ERROR: ${message}`);
        vscode.window.showErrorMessage(
          `ReasonForge verification failed: ${message}`
        );
      }
    }
  );
}
