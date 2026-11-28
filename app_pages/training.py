# app_pages/training.py
# UI for starting and monitoring model training runs.
# Interacts with runner.py to manage background training processes.

import contextlib
import io
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

import config
from hiervision import data, runner, pipeline, plots

def show_setup() -> None:
    # Displays dataset checks and allows starting a new training run based on selected nodes.
    st.header("Training Setup")
    
    # Data check capture mirroring the Dataset page[cite: 1, 2]
    if st.button("Check data"):
        st.write("### Data Check Log")
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            try:
                corrupted = data.check_data()
                st.success(f"Check passed. Corrupted files found: {corrupted}")
            except ValueError as e:
                st.error(str(e))
        st.text(f.getvalue())
    
    nodes = data.find_nodes(config.DATA_DIR)
    display_nodes = {n: (n if n else "root") for n in nodes}
    
    selected_nodes = st.multiselect(
        "Select nodes to train", 
        options=nodes, 
        default=nodes,
        format_func=lambda x: display_nodes[x]
    )
    
    overview = data.dataset_overview()
    valid_nodes = []
    
    # Analyze each selected node to mark ones that would be skipped
    for node in selected_nodes:
        node_data = [row for row in overview if row["node"] == node]
        classes = [row["class"] for row in node_data]
        
        st.write(f"#### Node: {display_nodes[node]}")
        
        skip_reasons = []
        if len(classes) < 2:
            skip_reasons.append("fewer than 2 classes")
        for row in node_data:
            if row["train"] < 10:
                skip_reasons.append(f"class '{row['class']}' has < 10 train images")
        
        if skip_reasons:
            st.warning(f"Will be skipped: {', '.join(skip_reasons)}")
        else:
            valid_nodes.append(node)
            
        for row in node_data:
            st.write(f"- {row['class']}: {row['train']} train images")
            
    if st.button("Start training", type="primary"):
        # Enforce validation before triggering the background runner process[cite: 2, 5]
        try:
            data.check_data()
            runner.start_training(valid_nodes)
            st.rerun()
        except ValueError as e:
            st.error(f"Data check failed. Fix errors before training:\n{e}")


def process_events(events_list: list[dict]) -> dict:
    # Extracts relevant metrics and state from the raw event stream lines.
    state = {
        "nodes_total": 0, "nodes_done": 0, "current_node": None,
        "epoch": 0, "total_epochs": 0, "batch": 0, "total_batches": 0,
        "train_losses": [], "val_accs": [], "epochs": [],
        "latest_val_acc": 0.0, "best_val_acc": 0.0, "sec_per_epoch": 0.0,
        "latest_conf_matrix": None, "classes": [],
        "status": "running", "finished_nodes": []
    }

    for ev in events_list:
        t = ev.get("type")
        if t == "run_start":
            state["nodes_total"] = ev.get("total_nodes", 0)
        elif t == "node_start":
            state["current_node"] = ev.get("node")
            state["classes"] = ev.get("classes", [])
            state["train_losses"] = []
            state["val_accs"] = []
            state["epochs"] = []
            # Fallback: grab total_epochs early if the event provides it
            if "total_epochs" in ev:
                state["total_epochs"] = ev["total_epochs"]
        elif t == "batch_end":
            state["epoch"] = ev.get("epoch", state["epoch"])
            state["total_epochs"] = ev.get("total_epochs", state["total_epochs"])
            state["batch"] = ev.get("batch", 0)
            state["total_batches"] = ev.get("total_batches", 0)
            state["train_loss"] = ev.get("loss", 0.0)
        elif t == "epoch_end":
            # ADDED: Ensure epoch and total_epochs update even if batch_end is missing
            state["epoch"] = ev.get("epoch", state["epoch"])
            if "total_epochs" in ev:
                state["total_epochs"] = ev["total_epochs"]

            state["epochs"].append(ev.get("epoch", 0))
            state["train_losses"].append(ev.get("train_loss", 0.0))
            state["val_accs"].append(ev.get("val_acc", 0.0))
            state["latest_val_acc"] = ev.get("val_acc", 0.0)
            state["sec_per_epoch"] = ev.get("duration", 0.0)
            if state["latest_val_acc"] > state["best_val_acc"]:
                state["best_val_acc"] = state["latest_val_acc"]
            if "confusion_matrix" in ev:
                state["latest_conf_matrix"] = ev["confusion_matrix"]
        elif t == "node_end":
            state["nodes_done"] += 1
            state["finished_nodes"].append(ev)
            state["best_val_acc"] = 0.0
        elif t in ("run_end", "run_error"):
            state["status"] = "done" if t == "run_end" else "error"

    return state


def show_finished_node(ev: dict) -> None:
    # Displays the final test metrics and confusion matrix for a completed node.
    node_name = ev.get('node') or 'root'
    st.write(f"### Results for {node_name}")
    st.write(f"**Test Accuracy:** {ev.get('test_acc', 0.0):.4f}")
    
    report = ev.get("classification_report", {})
    if report:
        # Filter out string keys (like 'accuracy') to ensure the DataFrame only parses metrics dictionaries
        class_metrics = {k: v for k, v in report.items() if isinstance(v, dict)}
        if class_metrics:
            df = pd.DataFrame.from_dict(class_metrics, orient="index")
            st.dataframe(df, use_container_width=True)
            
    matrix = ev.get("test_confusion_matrix")
    classes = ev.get("classes", [])
    if matrix and classes:
        normalize = st.toggle(f"Normalize matrix for {node_name}", key=f"norm_{node_name}")
        fig = plots.confusion_figure(matrix, classes, "Test Confusion Matrix", normalize=normalize)
        st.pyplot(fig)
        plt.close(fig)


def show_current_node(state: dict) -> None:
    # Displays live progress, metrics, and line charts for the currently training node.
    node_name = state['current_node'] or 'root'
    st.write(f"#### Training {node_name}")
    
    frac = 0.0
    if state['total_epochs'] > 0:
        epoch_frac = (state['epoch'] - 1) / state['total_epochs']
        batch_frac = (state['batch'] / state['total_batches']) / state['total_epochs'] if state['total_batches'] > 0 else 0
        frac = min(1.0, max(0.0, epoch_frac + batch_frac))
        
    st.progress(frac, text=f"Epoch {state['epoch']}/{state['total_epochs']}, batch {state['batch']}/{state['total_batches']}")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Train Loss", f"{state.get('train_loss', 0.0):.4f}")
    c2.metric("Val Accuracy", f"{state['latest_val_acc']:.4f}")
    c3.metric("Best Val Acc", f"{state['best_val_acc']:.4f}")
    c4.metric("Sec / Epoch", f"{state['sec_per_epoch']:.1f}s")
    
    if state["epochs"]:
        chart_data = pd.DataFrame({
            "Epoch": state["epochs"],
            "Train Loss": state["train_losses"],
            "Val Accuracy": state["val_accs"]
        }).set_index("Epoch")
        st.line_chart(chart_data)
        
    if state["latest_conf_matrix"] and state["classes"]:
        normalize = st.toggle("Normalize validation matrix", key="norm_val")
        fig = plots.confusion_figure(state["latest_conf_matrix"], state["classes"], "Validation Confusion Matrix", normalize=normalize)
        st.pyplot(fig)
        plt.close(fig)


@st.fragment(run_every=1)
def show_live_view() -> None:
    # Polls the events file automatically to refresh the live dashboard without full page reloads.
    events_list = runner.get_events()
    if not events_list:
        st.info("Waiting for run events...")
        if st.button("Stop training", type="primary"):
            runner.stop()
            st.rerun()
        return
        
    state = process_events(events_list)
    
    total = state["nodes_total"]
    done = state["nodes_done"]
    if total > 0:
        node_name = state['current_node'] or 'root'
        current_node_idx = done + 1
        overall_frac = done / total
        st.progress(overall_frac, text=f"Node {current_node_idx} of {total}: {node_name}")
        
    if state["current_node"] is not None and state["status"] == "running":
        show_current_node(state)
        
    for finished in state["finished_nodes"]:
        show_finished_node(finished)
        
    if state["status"] == "running":
        if st.button("Stop training", type="primary"):
            runner.stop()
            st.rerun()
    else:
        st.success(f"Run ended with status: {state['status']}")
        
        curr_run = runner.current()
        run_id = curr_run["run_id"] if curr_run else "unknown"
        if state["status"] == "done":
            if st.session_state.get("last_reloaded_run") != run_id:
                pipeline.reload_models()
                st.session_state["last_reloaded_run"] = run_id
            st.info("Models were reloaded.")
    
    with st.expander("Log Tail"):
        st.code(runner.log_tail(), language="text")


def main() -> None:
    # Roots the application logic and routes between the setup form or the live run view.
    st.title("Training")
    
    if runner.is_running():
        show_live_view()
    else:
        show_setup()
        
        events_list = runner.get_events()
        if events_list:
            st.write("---")
            st.subheader("Last run")
            state = process_events(events_list)
            for finished in state["finished_nodes"]:
                show_finished_node(finished)

main()