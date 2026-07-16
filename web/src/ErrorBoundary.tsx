import { Component, type ReactNode } from "react"
interface State { error: Error | null; stack: string }
export class ErrorBoundary extends Component<{ children: ReactNode }, State> {
  state: State = { error: null, stack: "" }
  static getDerivedStateFromError(error: Error): Partial<State> { return { error } }
  componentDidCatch(error: Error, info: { componentStack?: string }) {
    console.error("[colibrì] render crash:", error, info.componentStack)
    this.setState({ stack: info.componentStack ?? "" })
  }
  render() {
    if (!this.state.error) return this.props.children
    return <div style={{ padding: "2rem", fontFamily: "ui-monospace, monospace", color: "var(--foreground)", background: "var(--background)", minHeight: "100vh" }}>
      <h2 style={{ color: "var(--primary)" }}>colibrì UI hit an error</h2>
      <p style={{ color: "var(--muted-foreground)" }}>The engine is unaffected. Try refreshing.</p>
      <pre style={{ whiteSpace: "pre-wrap", color: "var(--destructive)" }}>{String(this.state.error)}</pre>
      <button onClick={() => this.setState({ error: null, stack: "" })} style={{ marginTop: "1rem", padding: "0.5rem 1rem", background: "var(--secondary)", color: "var(--foreground)", border: "1px solid var(--border)", borderRadius: 8, cursor: "pointer" }}>Retry</button>
    </div>
  }
}
