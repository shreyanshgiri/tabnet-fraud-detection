import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
from pytorch_tabnet.tab_model import TabNetClassifier

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="FinGuard Enterprise | TabNet Fraud & Risk Ops",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main-header {
        font-size: 2.15rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.15rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .stop-switch-banner {
        background-color: #7F1D1D;
        color: white;
        padding: 0.85rem 1.2rem;
        border-radius: 0.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- LOAD SAVED TABNET MODEL ---
@st.cache_resource
def load_model():
    clf = TabNetClassifier()
    clf.load_model('tabnet_fraud_model.zip')
    return clf

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model weights (`tabnet_fraud_model.zip`): {e}")
    st.stop()

# --- SESSION STATE INITIALIZATION ---
defaults = {
    'tx_type': "PAYMENT",
    'amount': 250.0,
    'old_orig': 5000.0,
    'is_new_recipient': False,
    'new_device': False,
    'geo_anomaly': False,
    'high_velocity': False,
    'stop_switch_active': False,
    'customer_tx_limit': 25000.0,
    'trusted_person_enabled': True,
    'audit_log': []
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def set_preset(tx_type, amt, o_orig, new_recip=False, new_dev=False, geo=False, vel=False):
    st.session_state.tx_type = tx_type
    st.session_state.amount = amt
    st.session_state.old_orig = o_orig
    st.session_state.is_new_recipient = new_recip
    st.session_state.new_device = new_dev
    st.session_state.geo_anomaly = geo
    st.session_state.high_velocity = vel

# --- FRAUD TYPOLOGY & CUSTOMER ADVISORY ENGINE ---
def diagnose_fraud_and_advise(tx_type, amount, old_orig, new_orig, is_new_recipient, ctx_flags, decision_zone):
    drain_ratio = (amount / old_orig * 100) if (old_orig > 0 and tx_type != "CASH_IN") else 0.0

    if decision_zone == "GREEN":
        return {
            "typology": "Legitimate Transaction Pattern",
            "severity": "LOW",
            "summary": "TabNet neural score and contextual device signals align with normal user behavior.",
            "customer_actions": [
                "Cleared for instant settlement with zero step-up friction.",
                "Keep real-time SMS/App alerts enabled for continuous account monitoring.",
                "Never share UPI PIN, OTP, or banking credentials with unverified callers."
            ]
        }

    if decision_zone == "AMBER":
        return {
            "typology": "Moderate Anomaly — Step-Up Verification Required",
            "severity": "MEDIUM",
            "summary": "Transaction exhibits moderate deviation or contextual risk; routed to RBI-compliant Risk-Based Authentication.",
            "customer_actions": [
                "**Complete Dynamic 2FA / Biometric Check:** Verify transaction parameters (amount & payee) via app-bound biometric token.",
                "**Trusted Person Mechanism (TPM) Approval:** If enabled, secondary confirmation from your registered Trusted Contact is requested before release.",
                "**Verify Payee Details:** Confirm the recipient was not provided via an unsolicited call, SMS, or screen-sharing app."
            ]
        }

    # RED ZONE DIAGNOSES
    if drain_ratio >= 85.0 and tx_type in ["TRANSFER", "CASH_OUT"]:
        return {
            "typology": "Account Takeover (ATO) & Full-Balance Drain",
            "severity": "CRITICAL",
            "summary": f"Attempted liquidation of {drain_ratio:.1f}% of available funds in a single {tx_type} operation.",
            "customer_actions": [
                "**Activate Emergency Stop Switch:** Immediately halt all customer-initiated outbound debits (inward credits remain unaffected).",
                "**Revoke Active Sessions & Reset Password:** Force-logout all devices and reset net-banking/UPI credentials.",
                "**Check for SIM-Swap / Malware:** Verify cellular network connectivity and uninstall any third-party remote-access apps.",
                "**File Cyber-Fraud Dispute:** Report immediately to preserve zero-liability protection."
            ]
        }

    if tx_type == "TRANSFER" and is_new_recipient:
        return {
            "typology": "Money Mule / Funnel Account Routing",
            "severity": "HIGH",
            "summary": "Funds routed to a newly added or zero-balance destination account commonly used to layer illicit proceeds.",
            "customer_actions": [
                "**Independent Beneficiary Verification:** Call the recipient via a known, trusted phone number before re-initiating.",
                "**Avoid Task/Commission Scams:** Never receive or forward funds on behalf of unknown third parties.",
                "**Enforce Beneficiary Cooling-Off:** Apply a 24-hour cooling window for newly added payees."
            ]
        }

    if tx_type in ["CASH_OUT", "DEBIT"]:
        return {
            "typology": "Unauthorized Card / ATM Skimming Attack",
            "severity": "HIGH",
            "summary": f"Anomalous {tx_type} withdrawal of ${amount:,.2f} flagged by TabNet feature masks.",
            "customer_actions": [
                "**Hotlist Card Immediately:** Disable physical and virtual card channels via Customer Controls.",
                "**Regenerate PIN & Restrict Channels:** Reset ATM PIN and disable international/contactless usage.",
                "**Audit Mini-Statement:** Check for small unauthorized 'card-testing' debits."
            ]
        }

    return {
        "typology": "Authorized Push Payment (APP) / Social Engineering Scam",
        "severity": "HIGH",
        "summary": "High-risk payment anomaly matching impersonation, fake KYC update, or urgent payment coercion.",
        "customer_actions": [
            "**Pause & Verify Caller Claims:** Banks and police never threaten digital arrest or ask you to move money to a 'safe account'.",
            "**Never Approve Unknown Collect Requests:** Entering your PIN/OTP always *deducts* funds—never use it to claim refunds.",
            "**Call Official Bank Helpline:** Dial the toll-free number on the back of your debit card."
        ]
    }

# --- HEADER ---
st.markdown('<div class="main-header">🛡️ FinGuard Enterprise: AI Fraud Detection & Risk Ops Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">PyTorch TabNet XAI • Risk-Based Step-Up Auth • Customer Stop-Switch Controls | Shreyansh Giri (CSE • Roll No: 2408390100060)</div>', unsafe_allow_html=True)

if st.session_state.stop_switch_active:
    st.markdown(
        '<div class="stop-switch-banner">🛑 EMERGENCY STOP SWITCH ACTIVE: All customer-initiated outbound debits are LOCKED. Inward credits remain active.</div>',
        unsafe_allow_html=True
    )

# --- SIDEBAR: PRESETS, CUSTOMER CONTROLS & POLICY SLIDERS ---
with st.sidebar:
    st.header("⚡ Live Demo Scenarios")
    st.caption("One-click simulation presets:")
    
    if st.button("✅ Normal Merchant Payment", use_container_width=True):
        set_preset("PAYMENT", 250.0, 5000.0, False, False, False, False)

    if st.button("🟡 Medium-Risk Step-Up Case", use_container_width=True):
        set_preset("TRANSFER", 3200.0, 8000.0, False, True, False, False)
        
    if st.button("🚨 Account Takeover Drain", use_container_width=True):
        set_preset("TRANSFER", 9500.0, 9500.0, True, True, True, True)
        
    if st.button("🕵️ Mule Account Transfer", use_container_width=True):
        set_preset("TRANSFER", 4500.0, 12000.0, True, False, False, True)

    if st.button("💳 Suspicious ATM Cash-Out", use_container_width=True):
        set_preset("CASH_OUT", 15000.0, 15200.0, False, False, True, False)

    st.divider()
    st.header("🔒 Customer Self-Controls")
    st.session_state.stop_switch_active = st.toggle(
        "🛑 Emergency Outbound Stop Switch",
        value=st.session_state.stop_switch_active,
        help="Immediately blocks all outbound customer-initiated debits while keeping inward credits active."
    )
    st.session_state.customer_tx_limit = st.number_input(
        "Per-Transaction Self Limit ($)",
        min_value=100.0,
        max_value=100000.0,
        value=float(st.session_state.customer_tx_limit),
        step=500.0
    )
    st.session_state.trusted_person_enabled = st.checkbox(
        "Trusted Person Mechanism (TPM)",
        value=st.session_state.trusted_person_enabled,
        help="Requires secondary approval from a registered Trusted Contact for Amber-zone transfers."
    )

    st.divider()
    st.header("🎛️ Risk Policy Bands")
    amber_thresh = st.slider("Amber Step-Up Threshold (%)", 15, 60, 35, 5)
    red_thresh = st.slider("Red Auto-Block Threshold (%)", 50, 95, 70, 5)

# --- MAIN NAVIGATION TABS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Real-Time Risk & Decision Engine",
    "📂 Batch CSV Audit",
    "📋 Analyst Case Queue & HITL",
    "📊 Executive ROI & XAI Telemetry"
])

# ==========================================
# TAB 1: REAL-TIME RISK & DECISION ENGINE
# ==========================================
with tab1:
    col_input, col_result = st.columns([1.05, 1.15], gap="large")
    
    with col_input:
        st.subheader("1. Transaction & Session Telemetry")
        
        tx_type_options = ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"]
        tx_type_map = {"CASH_IN": 0, "CASH_OUT": 1, "DEBIT": 2, "PAYMENT": 3, "TRANSFER": 4}
        
        tx_type = st.selectbox(
            "Transaction Channel",
            tx_type_options,
            index=tx_type_options.index(st.session_state.tx_type)
        )
        
        c1, c2 = st.columns(2)
        with c1:
            amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=float(st.session_state.amount), step=100.0)
        with c2:
            old_orig = st.number_input("Sender Current Balance ($)", min_value=0.0, value=float(st.session_state.old_orig), step=500.0)

        is_new_recipient = False
        if tx_type == "TRANSFER":
            is_new_recipient = st.checkbox(
                "Recipient is a newly added / $0-balance account",
                value=st.session_state.is_new_recipient
            )

        # Contextual Behavioral & Device Signals
        st.markdown("#### 📱 Contextual Session & Device Signals")
        s1, s2, s3 = st.columns(3)
        new_device = s1.checkbox("New Device / SIM", value=st.session_state.new_device)
        geo_anomaly = s2.checkbox("Geo-Velocity Jump", value=st.session_state.geo_anomaly)
        high_velocity = s3.checkbox("Burst Velocity (4+/10m)", value=st.session_state.high_velocity)

        # Automatic Balance Ledger Calculation
        if tx_type == "CASH_IN":
            new_orig = old_orig + amount
            old_dest, new_dest = 0.0, 0.0
            insufficient_funds = False
        else:
            insufficient_funds = amount > old_orig
            new_orig = max(0.0, old_orig - amount)
            if tx_type == "TRANSFER":
                old_dest = 0.0 if is_new_recipient else 1500.0
                new_dest = old_dest + amount
            else:
                old_dest, new_dest = 0.0, 0.0

        # Auto-Calculated Ledger Preview
        st.markdown("#### 🧾 Auto-Calculated Ledger State")
        l1, l2 = st.columns(2)
        l1.metric(
            "Sender Post-Tx Balance",
            f"${new_orig:,.2f}",
            delta=f"{'+' if tx_type == 'CASH_IN' else '-'}${amount:,.2f}",
            delta_color="normal" if tx_type == "CASH_IN" else "off"
        )
        if tx_type == "TRANSFER":
            l2.metric("Recipient Post-Tx Balance", f"${new_dest:,.2f}", delta=f"+${amount:,.2f} (Old: ${old_dest:,.2f})")
        else:
            l2.metric("Recipient Ledger", "Merchant / External", delta="N/A for channel", delta_color="off")

        if insufficient_funds:
            st.warning("⚠️ Amount exceeds available balance. Post-transaction balance clamped to $0.00.")

        analyze_btn = st.button("🚀 Evaluate Transaction Risk", type="primary", use_container_width=True)

        # Interactive What-If Sensitivity Curve
        with st.expander("📈 Amount Risk Sensitivity Curve (Current Balance)"):
            test_amounts = np.linspace(0, max(old_orig * 1.1, 1000.0), 25)
            batch_sim = np.zeros((len(test_amounts), 9))
            for idx, a in enumerate(test_amounts):
                n_o = (old_orig + a) if tx_type == "CASH_IN" else max(0.0, old_orig - a)
                o_d = (0.0 if is_new_recipient else 1500.0) if tx_type == "TRANSFER" else 0.0
                n_d = (o_d + a) if tx_type == "TRANSFER" else 0.0
                batch_sim[idx] = [1.0, float(tx_type_map[tx_type]), a, 0.0, old_orig, n_o, 0.0, o_d, n_d]
            sim_probs = model.predict_proba(batch_sim)[:, 1] * 100
            sim_df = pd.DataFrame({"Amount ($)": test_amounts, "TabNet Risk (%)": sim_probs}).set_index("Amount ($)")
            st.line_chart(sim_df)

    with col_result:
        st.subheader("2. Hybrid Risk Decision & Advisory")
        
        if analyze_btn:
            # 1. Check Customer Stop Switch & Self-Limit Guardrails first
            if st.session_state.stop_switch_active and tx_type != "CASH_IN":
                st.error("### 🛑 BLOCKED BY CUSTOMER STOP SWITCH")
                st.write("Customer-initiated outbound debits are currently disabled via the Emergency Stop Switch. Deactivate the switch in the sidebar after completing identity verification to resume payments.")
            elif amount > st.session_state.customer_tx_limit and tx_type != "CASH_IN":
                st.warning(f"### ⚠️ BLOCKED BY CUSTOMER SELF-LIMIT (${st.session_state.customer_tx_limit:,.2f})")
                st.write(f"Transaction amount (${amount:,.2f}) exceeds your configured per-transaction limit.")
            else:
                # 2. TabNet Neural Inference
                input_data = np.zeros((1, 9))
                input_data[0] = [1.0, float(tx_type_map[tx_type]), amount, 0.0, old_orig, new_orig, 0.0, old_dest, new_dest]
                tabnet_prob = float(model.predict_proba(input_data)[0][1]) * 100

                # 3. Contextual Risk Uplift (Device + Geo + Velocity)
                context_uplift = 0.0
                active_flags = []
                if new_device:
                    context_uplift += 12.0
                    active_flags.append("New Device / SIM-Swap Risk (+12%)")
                if geo_anomaly:
                    context_uplift += 15.0
                    active_flags.append("Impossible Geo-Velocity (+15%)")
                if high_velocity:
                    context_uplift += 10.0
                    active_flags.append("High Burst Velocity (+10%)")

                composite_risk = min(99.99, tabnet_prob + context_uplift) if tx_type != "CASH_IN" else tabnet_prob

                # 4. Three-Zone Decision Routing
                if composite_risk >= red_thresh:
                    decision_zone = "RED"
                    decision_label = "AUTO-BLOCKED (RED)"
                elif composite_risk >= amber_thresh:
                    decision_zone = "AMBER"
                    decision_label = "STEP-UP AUTH (AMBER)"
                else:
                    decision_zone = "GREEN"
                    decision_label = "APPROVED (GREEN)"

                advisory = diagnose_fraud_and_advise(
                    tx_type, amount, old_orig, new_orig, is_new_recipient, active_flags, decision_zone
                )

                # Log to Session Case Queue
                st.session_state.audit_log.insert(0, {
                    "Case ID": f"FG-{1001 + len(st.session_state.audit_log)}",
                    "Time": datetime.now().strftime("%H:%M:%S"),
                    "Channel": tx_type,
                    "Amount ($)": round(amount, 2),
                    "TabNet (%)": round(tabnet_prob, 2),
                    "Composite (%)": round(composite_risk, 2),
                    "System Action": decision_label,
                    "Typology": advisory["typology"],
                    "Analyst Status": "Pending Review" if decision_zone != "GREEN" else "Auto-Cleared"
                })

                # Render Decision Banner
                if decision_zone == "RED":
                    st.error(f"### 🚨 {decision_label} — {advisory['typology']}")
                elif decision_zone == "AMBER":
                    st.warning(f"### 🟡 {decision_label} — {advisory['typology']}")
                else:
                    st.success(f"### ✅ {decision_label} — {advisory['typology']}")

                m1, m2, m3 = st.columns(3)
                m1.metric("TabNet Neural Score", f"{tabnet_prob:.2f}%")
                m2.metric("Contextual Uplift", f"+{context_uplift:.0f}%", delta=f"{len(active_flags)} signals" if active_flags else "Clean", delta_color="inverse" if active_flags else "off")
                m3.metric("Composite Risk", f"{composite_risk:.2f}%", delta=advisory["severity"], delta_color="inverse" if decision_zone != "GREEN" else "normal")

                st.progress(min(int(composite_risk), 100))

                # Step-Up Interactive Challenge for AMBER Zone
                if decision_zone == "AMBER":
                    with st.container(border=True):
                        st.markdown("#### 🔐 Step-Up Authentication Challenge")
                        st.write("Complete secondary verification to release this transaction:")
                        c_otp, c_tpm = st.columns(2)
                        otp_ok = c_otp.checkbox("Verify App Biometric / TOTP")
                        tpm_ok = c_tpm.checkbox(
                            "Trusted Person Approval",
                            disabled=not st.session_state.trusted_person_enabled,
                            value=not st.session_state.trusted_person_enabled
                        )
                        if otp_ok and tpm_ok:
                            st.success("Step-Up Verified! Transaction released for settlement.")

                # Emergency One-Click Stop Switch inside Red Alert
                if decision_zone == "RED":
                    if st.button("🛑 Activate Emergency Outbound Stop Switch Now", type="secondary", use_container_width=True):
                        st.session_state.stop_switch_active = True
                        st.rerun()

                # Customer Remediation & Protection Guidance
                st.markdown("#### 🛡️ Customer Protection & Actionable Suggestions")
                st.write(f"**Diagnosis:** {advisory['summary']}")
                for step in advisory["customer_actions"]:
                    st.markdown(f"- {step}")

                # XAI Attribution Breakdown
                with st.expander("🧠 View TabNet Explainable AI (XAI) Attribution", expanded=True):
                    drain_ratio = (amount / old_orig * 100) if (old_orig > 0 and tx_type != "CASH_IN") else 0.0
                    st.markdown(f"- **Balance Depletion (`newbalanceOrig` / `oldbalanceOrg`):** `{drain_ratio:.1f}%` moved (Remaining: `${new_orig:,.2f}`).")
                    st.markdown(f"- **Channel Attention Weight (`type`):** `{tx_type}` (ID `{tx_type_map[tx_type]}`).")
                    if active_flags:
                        st.markdown(f"- **Contextual Signals Triggered:** {', '.join(active_flags)}")
        else:
            st.info("Configure transaction details & session signals on the left, then click **Evaluate Transaction Risk**.")

# ==========================================
# TAB 2: BATCH CSV AUDIT
# ==========================================
with tab2:
    st.subheader("Bulk Transaction Screening")
    st.write("Upload a 9-column numerical CSV file to run high-throughput TabNet batch scoring.")
    
    sample_df = pd.DataFrame(
        [[1, 3, 250.0, 0, 5000.0, 4750.0, 0, 1200.0, 1450.0],
         [1, 4, 9500.0, 0, 9500.0, 0.0, 0, 0.0, 9500.0],
         [1, 1, 14000.0, 0, 14500.0, 500.0, 0, 0.0, 0.0]],
        columns=["step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest"]
    )
    st.download_button(
        "📥 Download Sample CSV Template",
        data=sample_df.to_csv(index=False).encode('utf-8'),
        file_name="finguard_batch_template.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader("Upload Transaction CSV", type=["csv"])
    if uploaded_file is not None:
        df_batch = pd.read_csv(uploaded_file)
        if df_batch.shape[1] == 9:
            probs = model.predict_proba(df_batch.values)[:, 1] * 100
            zones = ["RED (BLOCK)" if p >= red_thresh else ("AMBER (STEP-UP)" if p >= amber_thresh else "GREEN (APPROVE)") for p in probs]
            df_batch["Risk_Score_%"] = np.round(probs, 2)
            df_batch["Decision_Zone"] = zones
            
            blocked_amt = df_batch.loc[probs >= red_thresh, "amount"].sum() if "amount" in df_batch.columns else 0.0
            b1, b2, b3 = st.columns(3)
            b1.metric("Total Audited", len(df_batch))
            b2.metric("Red Alerts Blocked", int((probs >= red_thresh).sum()))
            b3.metric("Fraud Exposure Prevented", f"${blocked_amt:,.2f}")
            
            st.dataframe(df_batch, use_container_width=True)
        else:
            st.warning(f"Expected 9 columns, but uploaded CSV has {df_batch.shape[1]} columns.")

# ==========================================
# TAB 3: ANALYST CASE QUEUE & HUMAN-IN-THE-LOOP (HITL)
# ==========================================
with tab3:
    st.subheader("Fraud Operations Center — Human-in-the-Loop (HITL) Case Queue")
    st.caption("Review flagged transactions, override false positives, and export labeled feedback for incremental TabNet retraining.")
    
    if st.session_state.audit_log:
        log_df = pd.DataFrame(st.session_state.audit_log)
        st.dataframe(log_df, use_container_width=True)
        
        # Analyst Override Control
        case_ids = [row["Case ID"] for row in st.session_state.audit_log]
        c_sel, c_act, c_btn = st.columns([1.2, 1.5, 1])
        selected_case = c_sel.selectbox("Select Case ID", case_ids)
        new_status = c_act.selectbox("Analyst Disposition", ["Confirmed Fraud (True Positive)", "Override: False Positive (Whitelist)"])
        if c_btn.button("Update Case Status", use_container_width=True):
            for row in st.session_state.audit_log:
                if row["Case ID"] == selected_case:
                    row["Analyst Status"] = new_status
            st.rerun()

        st.divider()
        c_dl, c_clr = st.columns([1.5, 4])
        with c_dl:
            st.download_button(
                "📥 Export Retraining & Audit CSV",
                data=pd.DataFrame(st.session_state.audit_log).to_csv(index=False).encode('utf-8'),
                file_name="finguard_hitl_feedback.csv",
                mime="text/csv"
            )
        with c_clr:
            if st.button("🗑️ Reset Session Queue"):
                st.session_state.audit_log = []
                st.rerun()
    else:
        st.info("No transactions evaluated yet. Run scenarios in Tab 1 to populate the Fraud Operations Case Queue.")

# ==========================================
# TAB 4: EXECUTIVE ROI & XAI TELEMETRY
# ==========================================
with tab4:
    st.subheader("Live Financial Impact & Model Benchmarks")
    
    # Calculate Live Session Financial Metrics
    total_scanned = len(st.session_state.audit_log)
    blocked_rows = [r for r in st.session_state.audit_log if "RED" in r["System Action"]]
    stepup_rows = [r for r in st.session_state.audit_log if "AMBER" in r["System Action"]]
    prevented_loss = sum(r["Amount ($)"] for r in blocked_rows)
    
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Session Fraud Prevented", f"${prevented_loss:,.2f}", f"{len(blocked_rows)} Blocked")
    e2.metric("Step-Up Challenges", f"{len(stepup_rows)}", "Friction-Balanced")
    e3.metric("TabNet Validation AUC", "0.9990", "Epoch 8 Optimal")
    e4.metric("False Negative Rate", "0.85%", "99.15% Recall")

    st.divider()
    st.markdown("#### Global TabNet Sequential Attention Feature Importance")
    importance_data = pd.DataFrame({
        "Feature": ["newbalanceOrig", "amount", "oldbalanceOrg", "newbalanceDest", "type", "nameOrig", "step", "nameDest", "oldbalanceDest"],
        "Importance (%)": [30.26, 20.44, 15.04, 13.38, 12.18, 4.64, 2.11, 1.68, 0.28]
    }).set_index("Feature")
    
    st.bar_chart(importance_data)