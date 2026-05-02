"""
Streamlit Dashboard — AI-Based Cost Management System
Run: streamlit run dashboard/app.py
"""
import streamlit as st
import requests
import pandas as pd
import json

API_BASE = "http://localhost:8000/api"

st.set_page_config(
    page_title="Cost Management System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar navigation ────────────────────────────────────────────────────────
page = st.sidebar.selectbox(
    "Navigation",
    ["Upload & Process", "Dashboard Overview", "Human Review", "Audit Log",
     "FL Training Monitor", "LayoutLM-FL Training"],
    index=0,
)
st.sidebar.markdown("---")
st.sidebar.markdown("**AI-Based Cost Management**")
st.sidebar.markdown("FYP — IBA Karachi")


# ── Helper functions (defined before the page routing chain) ──────────────────

def _render_result(result: dict):
    status = result["overall_status"]

    if status == "VERIFIED":
        st.success(f"✅ VERIFIED — Invoice passed all checks (Match Score: {result['match_result']['match_score']}%)")
    elif status == "FLAGGED":
        st.error("🚨 FLAGGED — Anomalies or discrepancies detected")
    else:
        st.warning("⚠️ UNMATCHED — No matching Purchase Order found")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Extracted Fields")
        fields = result["extracted_fields"]
        st.markdown(f"**Invoice #:** {fields.get('invoice_number') or '—'}")
        st.markdown(f"**Vendor:** {fields.get('vendor') or '—'}")
        st.markdown(f"**Date:** {fields.get('date') or '—'}")
        st.markdown(f"**PO Reference:** {fields.get('po_number') or '—'}")
        total    = fields.get("total_amount")
        currency = fields.get("currency") or "$"
        st.markdown(f"**Total Amount:** {f'{currency}{total:,.2f}' if total is not None else '—'}")
        st.caption(f"Extraction method: `{result.get('extraction_method', 'unknown')}`")

    with col2:
        st.subheader("Verification Details")
        match = result["match_result"]
        st.markdown(f"**Match Status:** `{match['status']}`")
        st.markdown(f"**Match Score:** {match['match_score']}%")
        if match["discrepancies"]:
            st.markdown("**Discrepancies:**")
            for d in match["discrepancies"]:
                st.markdown(f"- {d}")

        anomaly = result["anomaly_result"]
        if anomaly["rule_flags"]:
            st.markdown("**Rule-Based Flags:**")
            for f in anomaly["rule_flags"]:
                st.markdown(f"- {f}")
        if anomaly["is_statistical_outlier"]:
            st.markdown(
                f"**ML Anomaly:** Statistical outlier detected "
                f"(score: {anomaly['anomaly_score']:.4f})"
            )
        if not anomaly["rule_flags"] and not match["discrepancies"] and not anomaly["is_statistical_outlier"]:
            st.markdown("No anomalies detected.")

    with st.expander("Full JSON Response"):
        st.json(result)


def _show_sample_instructions():
    st.markdown("### Sample Test Files")
    st.markdown("""
After running `python scripts/generate_synthetic_data.py`, sample invoice PDFs
are in `data/sample_invoices/`. Try uploading one!

**Quick test:** Upload any `INV-*.pdf` to see the pipeline in action:
- A `correct` scenario invoice → should show **VERIFIED**
- An anomalous invoice → should show **FLAGGED**
""")


def _get_invoices():
    try:
        return requests.get(f"{API_BASE}/invoices", timeout=5).json()
    except Exception:
        return []


def _get_stats():
    try:
        return requests.get(f"{API_BASE}/dashboard-stats", timeout=5).json()
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: Upload & Process
# ══════════════════════════════════════════════════════════════════════════════
if page == "Upload & Process":
    st.title("Invoice Upload & Processing")
    st.markdown("Upload an invoice in PDF, image, CSV, or Excel format to run the full verification pipeline.")

    uploaded = st.file_uploader(
        "Choose an invoice file",
        type=["pdf", "png", "jpg", "jpeg", "csv", "xlsx"],
        help="Supported: PDF (digital or scanned), PNG/JPG images, CSV, Excel"
    )

    if uploaded:
        with st.spinner("Processing invoice through the AI pipeline..."):
            try:
                response = requests.post(
                    f"{API_BASE}/process-invoice",
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                    timeout=300,  # 5 min — Donut inference on CPU can take ~60s
                )
                if response.status_code == 200:
                    _render_result(response.json())
                else:
                    st.error(f"API Error {response.status_code}: {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to FastAPI backend. Start it with: `uvicorn app.main:app --reload`")
            except requests.exceptions.ReadTimeout:
                st.error("Request timed out. The backend is still loading the Donut model — wait 30s and try again.")
    else:
        st.info("Upload a file above to begin. You can also drag-and-drop.")
        _show_sample_instructions()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: Dashboard Overview
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Dashboard Overview":
    st.title("Dashboard Overview")

    stats = _get_stats()
    if stats is None:
        st.error("Cannot connect to FastAPI backend. Is it running? (`uvicorn app.main:app --reload`)")
        st.stop()

    total     = stats["total_processed"]
    verified  = stats["verified_count"]
    flagged   = stats["flagged_count"]
    unmatched = stats["unmatched_count"]
    savings   = stats["estimated_savings"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Processed", total)
    col2.metric("Verified",  verified,  f"{verified/max(total,1)*100:.0f}%")
    col3.metric("Flagged",   flagged,   f"{flagged/max(total,1)*100:.0f}%")
    col4.metric("Est. Savings", f"${savings:,.0f}")

    st.markdown("---")
    st.subheader("Invoice History")

    invoices = _get_invoices()
    if invoices:
        df = pd.DataFrame(invoices)
        display_cols = ["id", "filename", "uploaded_at", "overall_status",
                        "vendor", "po_number", "total_amount", "match_score"]
        display_cols = [c for c in display_cols if c in df.columns]

        search = st.text_input("Search by vendor, PO number, or invoice ID")
        if search:
            mask = df.apply(
                lambda row: search.lower() in row.astype(str).str.lower().to_string(), axis=1
            )
            df = df[mask]

        status_filter = st.multiselect(
            "Filter by status", ["VERIFIED", "FLAGGED", "UNMATCHED"],
            default=["VERIFIED", "FLAGGED", "UNMATCHED"]
        )
        if status_filter and "overall_status" in df.columns:
            df = df[df["overall_status"].isin(status_filter)]

        st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

        if "overall_status" in df.columns:
            st.subheader("Status Distribution")
            status_counts = df["overall_status"].value_counts().reset_index()
            status_counts.columns = ["Status", "Count"]
            st.bar_chart(status_counts.set_index("Status"))

        if stats.get("top_anomaly_types"):
            st.subheader("Top Anomaly Types")
            adf = pd.DataFrame(stats["top_anomaly_types"])
            st.bar_chart(adf.set_index("type")["count"])
    else:
        st.info("No invoices processed yet. Go to **Upload & Process** to get started.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: Human Review
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Human Review":
    st.title("Human Review — Flagged Invoices")
    st.markdown("Review flagged invoices, correct extraction errors, and approve or override system decisions.")

    invoices = _get_invoices()
    if not invoices:
        st.error("Cannot connect to backend.")
        st.stop()

    flagged = [i for i in invoices if i.get("overall_status") == "FLAGGED"]

    if not flagged:
        st.success("No flagged invoices pending review.")
        st.stop()

    st.info(f"{len(flagged)} flagged invoice(s) awaiting review")

    for inv in flagged:
        _cur = inv.get('currency') or '$'
        label = f"🚨 {inv['id']} — {inv.get('vendor', 'Unknown Vendor')} | {_cur}{inv.get('total_amount') or 0:,.2f}"
        with st.expander(label):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Extracted Fields** (editable)")
                new_vendor = st.text_input("Vendor",       value=inv.get("vendor") or "",       key=f"vendor_{inv['id']}")
                new_total  = st.text_input("Total Amount", value=str(inv.get("total_amount") or ""), key=f"total_{inv['id']}")
                new_po     = st.text_input("PO Number",    value=inv.get("po_number") or "",     key=f"po_{inv['id']}")
                new_date   = st.text_input("Date",         value=inv.get("date") or "",           key=f"date_{inv['id']}")

            with col2:
                st.markdown("**System Flags**")
                flags = json.loads(inv.get("rule_flags") or "[]")
                for f in flags:
                    st.markdown(f"- {f}")
                if inv.get("is_outlier"):
                    st.markdown(f"- ML outlier (score: {inv.get('anomaly_score', 0):.4f})")
                st.markdown(f"**Match Score:** {inv.get('match_score', 0)}%")

            bcol1, bcol2 = st.columns(2)
            with bcol1:
                if st.button("✅ Approve (override flag)", key=f"approve_{inv['id']}"):
                    for field, val in [("vendor", new_vendor), ("total_amount", new_total),
                                       ("po_number", new_po), ("date", new_date)]:
                        requests.post(f"{API_BASE}/feedback", json={
                            "invoice_id": inv["id"], "field_name": field,
                            "corrected_value": val, "user": "reviewer"
                        })
                    st.success("Correction submitted.")
            with bcol2:
                if st.button("🗑 Reject (confirm flag)", key=f"reject_{inv['id']}"):
                    requests.post(f"{API_BASE}/feedback", json={
                        "invoice_id": inv["id"], "field_name": "status",
                        "corrected_value": "REJECTED", "user": "reviewer"
                    })
                    st.warning("Invoice confirmed as rejected.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: Audit Log
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Audit Log":
    st.title("Audit Log")
    st.markdown("Complete audit trail of all verification actions.")

    try:
        logs = requests.get(f"{API_BASE}/audit-log", timeout=5).json()
    except Exception:
        st.error("Cannot connect to backend.")
        st.stop()

    if logs:
        df = pd.DataFrame(logs)
        st.dataframe(df, use_container_width=True, hide_index=True)

        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Audit Log as CSV",
            data=csv_data,
            file_name="audit_log.csv",
            mime="text/csv",
        )
    else:
        st.info("No audit log entries yet.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5: FL Training Monitor
# ══════════════════════════════════════════════════════════════════════════════
elif page == "FL Training Monitor":
    st.title("Federated Learning Training Monitor")
    st.markdown(
        "Train and monitor a privacy-preserving federated learning model across 3 simulated "
        "healthcare clients (Hospital, Clinic, Insurance) with differential privacy guarantees."
    )

    # ── Helper fetchers ───────────────────────────────────────────────────────
    def _get_fl_status():
        try:
            r = requests.get(f"{API_BASE}/fl/status", timeout=5)
            return r.json() if r.text.strip() else {}
        except Exception:
            return {}

    def _get_fl_history():
        try:
            r = requests.get(f"{API_BASE}/fl/history", timeout=5)
            return r.json() if r.text.strip() else {"runs": []}
        except Exception:
            return {"runs": []}

    def _trigger_fl_training(params: dict):
        r = requests.post(f"{API_BASE}/fl/train", json=params, timeout=300)
        if not r.text.strip():
            return {"error": f"Server returned empty response (HTTP {r.status_code}). Check backend logs."}
        try:
            data = r.json()
            # FastAPI wraps errors as {"detail": "..."}
            if r.status_code >= 400 and "detail" in data:
                return {"error": data["detail"]}
            return data
        except Exception:
            return {"error": f"HTTP {r.status_code}: {r.text[:400]}"}

    # ── Section 1: Status row ─────────────────────────────────────────────────
    fl_status = _get_fl_status()
    active = fl_status.get("fl_model_active", False)
    final_f1  = fl_status.get("final_f1")
    epsilon   = fl_status.get("epsilon")
    sigma_val = fl_status.get("sigma", 1.0)
    delta_val = fl_status.get("delta", 1e-5)
    n_rounds  = fl_status.get("n_rounds")

    mode = fl_status.get("mode", "")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "FL Model Active",
        "Yes" if active else "No",
        f"F1 = {final_f1:.4f}" if final_f1 is not None else "Not trained yet",
    )
    col2.metric(
        "Privacy Budget (ε)",
        f"{epsilon:.4f}" if epsilon is not None else "—",
        f"δ=1e-5  σ={sigma_val}",
    )
    col3.metric(
        "Active Model",
        "FL-SGD" if active else "Isolation Forest",
        f"{n_rounds} rounds" if n_rounds else "",
    )
    col4.metric(
        "Training Mode",
        "Distributed" if mode == "distributed" else ("Simulation" if mode == "simulation" else "—"),
        "3 client servers" if mode == "distributed" else ("in-process" if mode == "simulation" else ""),
    )

    if mode == "distributed":
        st.success("Distributed FL — 3 separate client processes, weights-only communication")
    elif mode == "simulation":
        st.info("Simulation mode — start `python start_fl_clients.py` for distributed FL")

    # ── Model toggle ──────────────────────────────────────────────────────────
    st.markdown("---")
    tcol1, tcol2 = st.columns(2)
    with tcol1:
        if st.button("Activate FL Model", disabled=active or not fl_status.get("run_id")):
            try:
                r = requests.post(f"{API_BASE}/fl/apply-model", json={"use_fl_model": True}, timeout=10)
                st.success(f"Switched to FL-SGD model ({r.json().get('active_model')})")
                st.rerun()
            except Exception as e:
                st.error(str(e))
    with tcol2:
        if st.button("Revert to Isolation Forest", disabled=not active):
            try:
                r = requests.post(f"{API_BASE}/fl/apply-model", json={"use_fl_model": False}, timeout=10)
                st.success(f"Reverted to Isolation Forest ({r.json().get('active_model')})")
                st.rerun()
            except Exception as e:
                st.error(str(e))

    # ── Section 2: Training controls ─────────────────────────────────────────
    st.markdown("---")
    st.subheader("Start Federated Training")

    pcol1, pcol2, pcol3, pcol4 = st.columns(4)
    with pcol1:
        num_rounds   = st.slider("Training Rounds",          5,  30, 15)
    with pcol2:
        noise_sigma  = st.slider("DP Noise Scale (σ)",    0.1, 5.0, 1.0, step=0.1)
    with pcol3:
        clip_norm    = st.slider("Gradient Clip Norm",    0.5, 2.0, 1.0, step=0.1)
    with pcol4:
        local_epochs = st.slider("Local Epochs per Round",  1,  10,   5)

    if st.button("▶ Start Federated Training"):
        with st.spinner(f"Running {num_rounds} FL rounds across 3 clients... (this takes ~10–30 seconds)"):
            try:
                result = _trigger_fl_training({
                    "n_rounds":     num_rounds,
                    "sigma":        noise_sigma,
                    "clip_norm":    clip_norm,
                    "local_epochs": local_epochs,
                })
                if "error" in result or result.get("status") == "failed":
                    st.error(f"Training failed: {result.get('error', result)}")
                else:
                    st.success(
                        f"Training complete! Final F1: {result.get('final_f1', 0):.4f} | "
                        f"ε = {result.get('epsilon', 0):.4f} | "
                        f"Time: {result.get('training_time_seconds', 0):.1f}s"
                    )
                    st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    # ── Fetch history for charts ──────────────────────────────────────────────
    history_data = _get_fl_history()
    runs = history_data.get("runs", [])

    if runs:
        latest_run = runs[0]
        round_metrics = latest_run.get("round_metrics", [])

        # ── Section 3: Convergence chart ─────────────────────────────────────
        if round_metrics:
            st.markdown("---")
            st.subheader("Model Convergence (Latest Run)")
            rm_df = pd.DataFrame(round_metrics)

            if "f1_score" in rm_df.columns and "round_number" in rm_df.columns:
                chart_df = rm_df.set_index("round_number")[["f1_score", "accuracy"]].rename(
                    columns={"f1_score": "FL F1 Score", "accuracy": "FL Accuracy"}
                )
                st.line_chart(chart_df)

            # ── Section 4: Privacy budget chart ──────────────────────────────
            st.subheader("Privacy Budget Consumption")
            if "epsilon_so_far" in rm_df.columns:
                eps_df = rm_df.set_index("round_number")[["epsilon_so_far"]].rename(
                    columns={"epsilon_so_far": "ε (epsilon)"}
                )
                st.area_chart(eps_df)
                max_eps = rm_df["epsilon_so_far"].max()
                if max_eps >= 1.0:
                    st.error(f"Privacy budget EXCEEDED: ε = {max_eps:.4f} ≥ 1.0")
                elif max_eps >= 0.9:
                    st.warning(f"Privacy budget near limit: ε = {max_eps:.4f} (threshold: 1.0)")
                else:
                    st.success(f"Privacy budget within limit: ε = {max_eps:.4f} < 1.0")

        # ── Section 5: Client statistics ─────────────────────────────────────
        st.markdown("---")
        st.subheader("Client Statistics")
        client_stats = latest_run.get("client_stats", {})
        if client_stats:
            ccol1, ccol2, ccol3 = st.columns(3)
            for col, (cid, stats_data) in zip([ccol1, ccol2, ccol3], client_stats.items()):
                with col:
                    st.markdown(f"**{stats_data.get('sector', cid)}**")
                    st.metric("Samples", stats_data.get("n_samples", "—"))
                    st.metric("Anomaly Rate", f"{stats_data.get('anomaly_rate', 0)*100:.1f}%")
                    amt = stats_data.get("amount_range", [0, 0])
                    if amt and len(amt) == 2:
                        st.markdown(f"Amount: ${amt[0]:,.0f} – ${amt[1]:,.0f}")
                    st.metric("Local F1", f"{stats_data.get('local_f1', 0):.4f}")

    # ── Section 6: Training history table ────────────────────────────────────
    st.markdown("---")
    st.subheader("Training Run History")
    if runs:
        hist_df = pd.DataFrame([
            {
                "Run ID":        r.get("run_id", ""),
                "Started":       r.get("started_at", "")[:19] if r.get("started_at") else "",
                "Rounds":        r.get("n_rounds", ""),
                "Final F1":      f"{r['final_f1']:.4f}" if r.get("final_f1") is not None else "—",
                "Epsilon (ε)":   f"{r['epsilon']:.4f}"  if r.get("epsilon")  is not None else "—",
                "Time (s)":      r.get("training_time_s", "—"),
                "Status":        r.get("status", ""),
            }
            for r in runs
        ])
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
    else:
        st.info("No training runs yet. Start a training run above.")

    # ── Section 7: Privacy mechanism explanation ──────────────────────────────
    st.markdown("---")
    with st.expander("How Differential Privacy Works in This System"):
        st.markdown("""
### 3-Step Privacy Protection

**Step 1 — Gradient Clipping**
Each client's model parameters are projected onto an L2 ball of radius `clip_norm`.
This *bounds the sensitivity* — no single client's update can shift the global model
by more than `clip_norm` in L2 distance.

**Step 2 — Gaussian Noise Injection**
Zero-mean Gaussian noise with `noise_std = clip_norm × σ` is added to each client's
clipped parameters before sending them to the server.
The server never sees any individual client's true model parameters.

**Step 3 — Rényi DP Budget Tracking (RDP)**
Cumulative privacy cost is tracked using Rényi Differential Privacy (Balle et al. 2020):

> ε_RDP(α) = α / (2σ²) per round → composed over n_rounds → converted to (ε, δ)-DP

With default settings (σ=1.0, 15 rounds, δ=1e-5), this gives **ε ≈ 0.90 < 1.0** ✓

**Secure Aggregation Simulation**
Pairwise random masks are added to client updates before aggregation:
`mask_ij = -mask_ji` so all masks cancel in the weighted sum.
The server aggregates masked parameters and never sees individual client values.
        """)
        st.markdown(f"""
**Current Settings:** σ={sigma_val} | clip_norm={fl_status.get('clip_norm', '—')} | δ={delta_val}

**Privacy Guarantee:** (ε={f"{epsilon:.4f}" if epsilon else "—"}, δ={delta_val})-DP
        """)


# ── PAGE: LayoutLM-FL Training ────────────────────────────────────────────────

elif page == "LayoutLM-FL Training":
    import time
    import streamlit.components.v1 as components

    st.title("LayoutLM-FL Training")
    st.markdown(
        "Federated fine-tuning of **LayoutLMv3 + LoRA** across 3 simulated company clients. "
        "Only LoRA adapter weights (~1.2 MB) are transmitted each round — raw invoices never leave each client."
    )

    # ── Helper functions ──────────────────────────────────────────────────────
    def _lora_status():
        try:
            r = requests.get(f"{API_BASE}/fl/layoutlm/status", timeout=5)
            return r.json() if r.ok else {}
        except Exception:
            return {}

    def _lora_live_status():
        try:
            r = requests.get(f"{API_BASE}/fl/layoutlm/live-status", timeout=5)
            return r.json() if r.ok else {}
        except Exception:
            return {}

    # ── Section 1: Status banner ──────────────────────────────────────────────
    lora_status    = _lora_status()
    lora_available = lora_status.get("fl_model_available", False)
    lora_active    = lora_status.get("fl_model_active",    False)
    lora_f1        = lora_status.get("final_macro_f1")
    lora_eps       = lora_status.get("epsilon")

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("FL Model Available", "Yes" if lora_available else "No")
    s2.metric("Active for Extraction", "Yes" if lora_active else "No")
    s3.metric("Final Macro F1",  f"{lora_f1:.4f}" if lora_f1 is not None else "—")
    s4.metric("Privacy ε",       f"{lora_eps:.4f}" if lora_eps is not None else "—")

    # ── Section 2: Toggle FL model ────────────────────────────────────────────
    st.markdown("---")
    col_on, col_off = st.columns(2)
    with col_on:
        if st.button("Activate FL-LoRA Model", disabled=lora_active or not lora_available):
            try:
                r = requests.post(f"{API_BASE}/fl/layoutlm/apply-model",
                                  params={"use_fl_model": True}, timeout=10)
                if r.ok:
                    st.success("FL-LoRA model activated — extraction now uses federated model")
                    st.rerun()
                else:
                    st.error(r.text)
            except Exception as e:
                st.error(f"Error: {e}")
    with col_off:
        if st.button("Deactivate FL-LoRA Model", disabled=not lora_active):
            try:
                r = requests.post(f"{API_BASE}/fl/layoutlm/apply-model",
                                  params={"use_fl_model": False}, timeout=10)
                if r.ok:
                    st.info("Reverted to base LayoutLMv3 model")
                    st.rerun()
                else:
                    st.error(r.text)
            except Exception as e:
                st.error(f"Error: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # REAL FL TRAINING — Live streaming panel
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("Federated Learning — LayoutLMv3 + LoRA")
    st.markdown(
        "Uses the **LayoutLMv3 + LoRA weights** in a training loop. "
        "Metrics stream in round-by-round as each client trains locally and the "
        "server aggregates via FedAvg with differential privacy."
    )

    # Training configuration inputs
    t1, t2, t3 = st.columns(3)
    lora_n_rounds  = t1.number_input("Rounds",        min_value=1, max_value=30, value=5,
                                     help="Keep low (3–5); each round takes minutes on CPU")
    lora_local_eps = t2.number_input("Epochs/round",  min_value=1, max_value=5,  value=1)
    lora_r_val     = t3.number_input("LoRA rank (r)", min_value=4, max_value=32, value=8)

    d1, d2, d3 = st.columns(3)
    lora_sigma_val = d1.number_input("DP sigma (σ)",  min_value=0.0, max_value=5.0, value=0.5, step=0.1)
    lora_clip_val  = d2.number_input("DP clip_norm",  min_value=0.05, max_value=2.0, value=0.3, step=0.05)
    lora_device    = d3.selectbox("Device", ["cpu", "cuda"], index=0)

    # Check current live state
    live = _lora_live_status()
    currently_training = live.get("is_training", False)

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        start_clicked = st.button(
            "▶ Start FL Training",
            type="primary",
            disabled=currently_training,
        )
    with btn_col2:
        stop_clicked = st.button(
            "⏹ Stop Training",
            disabled=not currently_training,
        )

    if stop_clicked:
        try:
            requests.post(f"{API_BASE}/fl/layoutlm/stop", timeout=5)
            st.warning("Stop signal sent — training will halt after the current round.")
        except Exception as e:
            st.error(f"Could not send stop signal: {e}")

    if start_clicked:
        try:
            r = requests.post(
                f"{API_BASE}/fl/layoutlm/train-async",
                params={
                    "n_rounds":     int(lora_n_rounds),
                    "local_epochs": int(lora_local_eps),
                    "lora_r":       int(lora_r_val),
                    "sigma":        float(lora_sigma_val),
                    "clip_norm":    float(lora_clip_val),
                    "device":       lora_device,
                },
                timeout=10,
            )
            if r.ok:
                data = r.json()
                st.success(f"Training started — Run ID: `{data.get('run_id')}`")
                currently_training = True
                live = _lora_live_status()
            else:
                st.error(f"Failed to start: {r.text}")
        except Exception as e:
            st.error(f"Error: {e}")

    # ── Live metrics (rendered every rerun while training is active) ─────────
    rh = live.get("round_history", [])

    if currently_training:
        cr = live.get("current_round", 0)
        tr = live.get("total_rounds", max(int(lora_n_rounds), 1))
        st.info(f"Training in progress — auto-refreshing every 5 s  |  Round {cr} / {tr}")
        st.progress(cr / max(tr, 1))

    if rh:
        last      = rh[-1]
        cl_losses = last.get("client_losses", {})
        cl_f1s    = last.get("client_f1s",    {})

        # Per-client stat cards
        cc1, cc2, cc3 = st.columns(3)
        for col, cid, label in zip(
            [cc1, cc2, cc3],
            ["client_a", "client_b", "client_c"],
            ["Client A — EuroSupplier", "Client B — AsiaRetail_A", "Client C — AsiaRetail_B"],
        ):
            loss_val = cl_losses.get(cid)
            f1_val   = cl_f1s.get(cid)
            col.metric(label,
                       f"Loss: {loss_val:.4f}" if loss_val is not None else "Loss: —",
                       f"F1: {f1_val:.4f}"     if f1_val   is not None else "F1: —")

        # Round-by-round table
        rh_df = pd.DataFrame(rh)
        display_cols = [c for c in ["round", "macro_f1", "epsilon", "param_change", "elapsed_s"]
                        if c in rh_df.columns]
        renamed = {"macro_f1": "Macro F1", "epsilon": "ε Budget",
                   "param_change": "ΔW Norm", "elapsed_s": "Time (s)"}
        st.markdown("**Round-by-round metrics**")
        st.dataframe(rh_df[display_cols].rename(columns=renamed).iloc[::-1],
                     use_container_width=True, hide_index=True)

        # Live charts — full-width vertical flow
        if "macro_f1" in rh_df.columns:
            st.markdown("**Macro F1 per Round**")
            st.line_chart(
                rh_df.set_index("round")[["macro_f1"]].rename(columns={"macro_f1": "Macro F1"}),
                height=350,
                use_container_width=True,
            )

        st.markdown("---")

        if "epsilon" in rh_df.columns:
            st.markdown("**Privacy Budget ε per Round**")
            st.area_chart(
                rh_df.set_index("round")[["epsilon"]].rename(columns={"epsilon": "ε"}),
                height=350,
                use_container_width=True,
            )

        st.markdown("---")

        if "param_change" in rh_df.columns:
            st.markdown("**ΔW Norm per Round**")
            st.bar_chart(
                rh_df.set_index("round")[["param_change"]].rename(columns={"param_change": "ΔW"}),
                height=350,
                use_container_width=True,
            )

    elif not currently_training:
        st.info("Configure parameters above and click **▶ Start FL Training** to begin.")

    # ── Visual FL Simulation (embedded iframe — always visible) ───────────────
    st.markdown("---")
    st.subheader("Visual FL Simulation")
    st.markdown(
        "The animation below shows how federated learning distributes training "
        "across clients, exchanges weight updates, and tracks the privacy budget. "
        "This runs independently of the training above."
    )
    components.iframe("http://localhost:8000/fl-simulation", height=680, scrolling=False)

    # ── Historical convergence charts (from saved metadata, shown after run) ──
    round_history = lora_status.get("round_history", [])
    if round_history and not currently_training:
        st.markdown("---")
        st.subheader("Convergence Curves (Last Completed Run)")
        try:
            rh_df = pd.DataFrame(round_history)

            if "macro_f1" in rh_df.columns:
                st.markdown("**Macro F1 per Round**")
                st.line_chart(
                    rh_df.set_index("round")[["macro_f1"]].rename(columns={"macro_f1": "Macro F1"}),
                    height=380,
                    use_container_width=True,
                )
                st.markdown("---")

            if "per_class_f1" in rh_df.columns:
                per_class_rows = rh_df["per_class_f1"].tolist()
                if per_class_rows and isinstance(per_class_rows[0], dict):
                    pc_df = pd.DataFrame(per_class_rows, index=rh_df["round"])
                    keep_cols = [c for c in pc_df.columns
                                 if any(e in c for e in ("VENDOR", "DATE", "TOTAL",
                                                          "INVOICE", "PO_NO", "CURRENCY"))]
                    if keep_cols:
                        st.markdown("**Per-Entity F1 Score by Round**")
                        st.line_chart(pc_df[keep_cols], height=380, use_container_width=True)
                        st.markdown("---")

            if "epsilon" in rh_df.columns:
                st.markdown("**Privacy Budget (ε) over Rounds**")
                st.area_chart(
                    rh_df.set_index("round")[["epsilon"]].rename(columns={"epsilon": "ε (epsilon)"}),
                    height=380,
                    use_container_width=True,
                )
                max_eps = rh_df["epsilon"].max()
                if max_eps >= 1.0:
                    st.error(f"Privacy budget EXCEEDED: ε = {max_eps:.4f} ≥ 1.0")
                elif max_eps >= 0.9:
                    st.warning(f"Privacy budget near limit: ε = {max_eps:.4f}")
                else:
                    st.success(f"Privacy budget within limit: ε = {max_eps:.4f} < 1.0")
        except Exception as e:
            st.warning(f"Could not render charts: {e}")

    # ── Auto-refresh while training (must be last — triggers full page rerun) ──
    if currently_training:
        time.sleep(5)
        st.rerun()

    # ── Groq vs FL-LoRA comparison ─────────────────────────────────────────────
    st.markdown("---")
    with st.expander("Groq API vs FL-LoRA Comparison"):
        st.markdown("""
| | **Groq API (LLaMA-4 Vision)** | **Federated LayoutLMv3 + LoRA** |
|---|---|---|
| **Privacy** | Sends invoice to external API | Data never leaves device |
| **Accuracy** | High out-of-box | Improves over FL rounds |
| **Training** | None required | Federated fine-tuning |
| **Communication** | Full invoice image | ~1.2 MB LoRA delta/round |
| **DP Guarantee** | None | (ε, δ)-DP via RDP accounting |
| **Offline** | Requires internet | Runs fully offline |
        """)

    with st.expander("Non-IID Data Distribution across Clients"):
        st.markdown("""
| Client | Data | Size | Labels | Language |
|---|---|---|---|---|
| **Client A** (EuroSupplier) | French FACTU invoices | 900 | VENDOR, DATE, TOTAL, INVOICE_NO, PO_NO, CURRENCY | French → English |
| **Client B** (AsiaRetail_A) | SROIE receipts (half) | ~487 | VENDOR, DATE, TOTAL | English |
| **Client C** (AsiaRetail_B) | SROIE receipts (half) | ~486 | VENDOR, DATE, TOTAL | English |

Client A has a richer label distribution (6 entity types vs 3).
This **label distribution non-IID** is the realistic FL scenario — different
companies hold genuinely different invoice types.
        """)
