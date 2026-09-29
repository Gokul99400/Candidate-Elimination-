import io
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Candidate Elimination Concept Learner",
    page_icon="🧠",
    layout="wide",
)


# --- Core Candidate Elimination Algorithm with Trace History ---
def run_candidate_elimination_trace(df, target_col):
    concepts = np.array(df.drop(columns=[target_col]))
    targets = np.array(df[target_col])
    num_attributes = concepts.shape[1]

    # Initialize S with first positive example
    specific_h = None
    first_pos_idx = -1
    for idx, label in enumerate(targets):
        if str(label).strip().lower() in ["yes", "1", "true", "positive"]:
            specific_h = concepts[idx].copy()
            first_pos_idx = idx
            break

    if specific_h is None:
        return None, "Error: Dataset must contain at least one positive example."

    # Initialize G with all wildcards
    general_h = [["?" for _ in range(num_attributes)]]

    # Store step-by-step history
    history = [
        {
            "step": 0,
            "instance": "Initial State",
            "type": "Init",
            "S": [specific_h.copy().tolist()],
            "G": [g.copy() for g in general_h],
            "description": "Initialized S to the first positive example and G to the most general hypothesis.",
        }
    ]

    for i, instance in enumerate(concepts):
        is_pos = (
            str(targets[i]).strip().lower() in ["yes", "1", "true", "positive"]
        )
        desc = ""

        if is_pos:
            # Positive instance: Generalize S
            for j in range(num_attributes):
                if instance[j] != specific_h[j]:
                    specific_h[j] = "?"

            # Prune G of any hypothesis inconsistent with this positive instance
            general_h = [
                g
                for g in general_h
                if all(
                    g[k] == "?" or g[k] == instance[k]
                    for k in range(num_attributes)
                )
            ]
            desc = (
                f"Instance {i+1} is Positive. Generalized S and pruned inconsistent hypotheses from G."
                if i != first_pos_idx
                else f"Instance {i+1} matches initial positive anchor."
            )

        else:
            # Negative instance: Specialize G
            new_g = []
            for g in general_h:
                # If g covers the negative instance, specialize it
                if all(
                    g[k] == "?" or g[k] == instance[k]
                    for k in range(num_attributes)
                ):
                    for j in range(num_attributes):
                        if (
                            instance[j] != specific_h[j]
                            and specific_h[j] != "?"
                        ):
                            spec_hyp = g.copy()
                            spec_hyp[j] = specific_h[j]
                            new_g.append(spec_hyp)
                else:
                    new_g.append(g)

            # Eliminate duplicates and hypotheses strictly more specific than others in G
            unique_g = []
            for g in new_g:
                if g not in unique_g:
                    unique_g.append(g)

            general_h = unique_g
            desc = f"Instance {i+1} is Negative. Specialized G to exclude the negative instance."

        history.append(
            {
                "step": i + 1,
                "instance": list(instance),
                "type": "Positive" if is_pos else "Negative",
                "S": [specific_h.copy().tolist()],
                "G": [g.copy() for g in general_h],
                "description": desc,
            }
        )

    return history, None


# --- Classification Function ---
def classify_instance(query_instance, S_boundary, G_boundary):
    num_attr = len(query_instance)

    def satisfies(hyp, inst):
        return all(h == "?" or h == inst[k] for k, h in enumerate(hyp))

    s_satisfied = all(satisfies(s, query_instance) for s in S_boundary)
    g_satisfied = all(satisfies(g, query_instance) for g in G_boundary)
    some_g_satisfied = any(satisfies(g, query_instance) for g in G_boundary)

    if s_satisfied and g_satisfied:
        return (
            "Positive (Yes)",
            "All hypotheses in both S and G satisfy this instance.",
            "success",
        )
    elif not s_satisfied and not some_g_satisfied:
        return (
            "Negative (No)",
            "No hypothesis in S or G satisfies this instance.",
            "error",
        )
    else:
        return (
            "Ambiguous / Uncertain",
            "Falls between boundaries: partially satisfied by the version space. More training samples needed.",
            "warning",
        )


# --- Default Datasets ---
DEFAULT_GFG = {
    "Sky": ["Sunny", "Sunny", "Rainy", "Sunny"],
    "AirTemp": ["Warm", "Warm", "Cold", "Warm"],
    "Humidity": ["Normal", "High", "High", "High"],
    "Wind": ["Strong", "Strong", "Strong", "Strong"],
    "Water": ["Warm", "Warm", "Warm", "Cool"],
    "Forecast": ["Same", "Same", "Change", "Change"],
    "EnjoySport": ["Yes", "Yes", "No", "Yes"],
}

EXPANDED_GFG = {
    "Sky": ["Sunny", "Sunny", "Rainy", "Sunny", "Rainy", "Sunny", "Sunny"],
    "AirTemp": ["Warm", "Warm", "Cold", "Warm", "Warm", "Cold", "Warm"],
    "Humidity": [
        "Normal",
        "High",
        "High",
        "High",
        "High",
        "Normal",
        "Normal",
    ],
    "Wind": [
        "Strong",
        "Strong",
        "Strong",
        "Strong",
        "Strong",
        "Strong",
        "Light",
    ],
    "Water": ["Warm", "Warm", "Warm", "Cool", "Cool", "Warm", "Warm"],
    "Forecast": [
        "Same",
        "Same",
        "Change",
        "Change",
        "Change",
        "Same",
        "Same",
    ],
    "EnjoySport": ["Yes", "Yes", "No", "Yes", "No", "No", "Yes"],
}

# --- Sidebar UI ---
st.sidebar.title("⚙️ Configuration")
data_source = st.sidebar.radio(
    "Select Dataset Source:",
    ("Classic GfG EnjoySport (4 rows)", "Expanded (7 rows)", "Upload Custom CSV"),
)

if data_source == "Classic GfG EnjoySport (4 rows)":
    df = pd.DataFrame(DEFAULT_GFG)
elif data_source == "Expanded (7 rows)":
    df = pd.DataFrame(EXPANDED_GFG)
else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload a CSV file", type=["csv"]
    )
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.info("Please upload a CSV file to proceed. Loading default dataset.")
        df = pd.DataFrame(DEFAULT_GFG)

target_col = st.sidebar.selectbox(
    "Target / Class Column:", df.columns, index=len(df.columns) - 1
)

# --- Header & Overview ---
st.title("🧠 Candidate Elimination Concept Learning System")
st.markdown(
    """
This application demonstrates the **Candidate Elimination Algorithm** (Mitchell, 1997) 
computing the version space bounded by the **Most Specific ($S$)** and **Most General ($G$)** hypotheses.
"""
)

# --- Dataset Preview ---
with st.expander("📊 Dataset View & Download", expanded=True):
    st.dataframe(df, use_container_width=True)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Current Dataset as CSV",
        data=csv_bytes,
        file_name="candidate_elimination_dataset.csv",
        mime="text/csv",
    )

# --- Run Algorithm ---
history, err = run_candidate_elimination_trace(df, target_col)

if err:
    st.error(err)
    st.stop()

st.divider()

# --- Execution Trace Tabs ---
tab1, tab2, tab3 = st.tabs(
    [
        "👣 Step-by-Step Stepper",
        "🏁 Final Version Space",
        "🔍 Classify New Instance",
    ]
)

feature_cols = [c for c in df.columns if c != target_col]

with tab1:
    st.subheader("Interactive Version Space Progression")
    max_steps = len(history) - 1
    selected_step = st.slider(
        "Step through training instances:", 0, max_steps, max_steps
    )

    current_state = history[selected_step]

    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.metric("Current Step", f"{selected_step} / {max_steps}")
        st.write(f"**Step Action:** {current_state['description']}")
        if current_state["type"] != "Init":
            badge_color = (
                "🟢" if current_state["type"] == "Positive" else "🔴"
            )
            st.write(
                f"**Instance Type:** {badge_color} {current_state['type']}"
            )
            st.code(
                f"Features: {dict(zip(feature_cols, current_state['instance']))}"
            )

    with col_b:
        st.markdown("#### Specific Boundary ($S$):")
        s_df = pd.DataFrame(current_state["S"], columns=feature_cols)
        st.dataframe(s_df, use_container_width=True)

        st.markdown("#### General Boundary ($G$):")
        g_df = pd.DataFrame(current_state["G"], columns=feature_cols)
        st.dataframe(g_df, use_container_width=True)

with tab2:
    st.subheader("Final Computed Version Space Boundaries")
    final_state = history[-1]

    col_s, col_g = st.columns(2)
    with col_s:
        st.success("🎯 **Final Specific Hypothesis ($S$)**")
        st.dataframe(
            pd.DataFrame(final_state["S"], columns=feature_cols),
            use_container_width=True,
        )
    with col_g:
        st.info("🌐 **Final General Hypotheses ($G$)**")
        st.dataframe(
            pd.DataFrame(final_state["G"], columns=feature_cols),
            use_container_width=True,
        )

    st.markdown("---")
    st.markdown("### Interpretation:")
    st.markdown(
        f"""
    - **Attributes bounded to fixed values** in $S$ represent essential conditions for a positive outcome.
    - **Wildcard values (`?`)** indicate attributes that can vary without altering classification consistency.
    - Total valid general bounds currently isolating the concept: **{len(final_state['G'])}**.
    """
    )

with tab3:
    st.subheader("Predict Unseen Instances with Learned Boundaries")
    st.write(
        "Select values for each attribute to evaluate where the new sample lands within the version space:"
    )

    query_inputs = []
    cols = st.columns(len(feature_cols))

    for idx, col_name in enumerate(feature_cols):
        unique_vals = list(df[col_name].unique())
        with cols[idx]:
            val = st.selectbox(f"{col_name}:", options=unique_vals, key=col_name)
            query_inputs.append(val)

    if st.button("Evaluate Instance", type="primary"):
        verdict, explanation, status_type = classify_instance(
            query_inputs, final_state["S"], final_state["G"]
        )

        if status_type == "success":
            st.success(f"### Classification: {verdict}")
        elif status_type == "error":
            st.error(f"### Classification: {verdict}")
        else:
            st.warning(f"### Classification: {verdict}")

        st.write(explanation)
        st.caption(f"Tested Vector: `{dict(zip(feature_cols, query_inputs))}`")