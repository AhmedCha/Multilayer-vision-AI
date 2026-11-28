# app_pages/dataset.py
# Provides dataset analytics, Open Images catalogue browsing, and target management.
# Divided into tabs to keep the interface clean and focused.

import os
import socket
from urllib.parse import urlparse
import contextlib
import io
import streamlit as st

import config
from hiervision import data, openimages, targets, runner


def tab_current_dataset() -> None:
    # Shows statistics and integrity checks for the local data/ folder.
    @st.cache_data
    def get_cached_overview():
        return data.dataset_overview()

    overview = get_cached_overview()

    col1, col2, col3, col4 = st.columns(4)
    if col1.button("Refresh data"):
        get_cached_overview.clear()
        st.rerun()

    if col2.button("Run data check"):
        st.write("### Data Check Log")
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            try:
                corrupted = data.check_data()
                st.success(f"Check passed. Corrupted files found: {corrupted}")
            except ValueError as e:
                st.error(str(e))
        st.text(f.getvalue())

    if not overview:
        st.info("No data found.")
        return

    all_nodes = sorted(list(set(row["node"] for row in overview)))
    display_nodes = {n: (n if n else "root") for n in all_nodes}

    node_sel = st.selectbox("Select Node", options=all_nodes, format_func=lambda n: display_nodes[n])
    node_data = [row for row in overview if row["node"] == node_sel]

    total_images = sum(row["total"] for row in node_data)
    col1.metric("Nodes", len(all_nodes))
    col2.metric("Classes (this node)", len(node_data))
    col3.metric("Images (this node)", total_images)

    table_data = []
    warnings = []
    chart_data = {}

    for row in node_data:
        table_data.append({
            "node": display_nodes[row["node"]],
            "class": row["class"],
            "total": row["total"],
            "train": row["train"],
            "val": row["val"],
            "test": row["test"]
        })
        chart_data[row["class"]] = row["total"]

        if row["val"] < 30 or row["test"] < 30:
            warnings.append(f"Class '{row['class']}' has < 30 validation or test images ({row['val']} val, {row['test']} test).")
        if row["train"] < 10:
            warnings.append(f"Class '{row['class']}' has < 10 train images ({row['train']} train).")

    st.dataframe(table_data, use_container_width=True)

    st.write("### Class Distribution")
    st.bar_chart(chart_data)

    if warnings:
        st.write("### Warnings")
        for w in warnings:
            st.warning(w)

    st.write("### Source Breakdown")
    for row in node_data:
        with st.expander(f"{row['class']} sources"):
            for src, count in row["sources"].items():
                st.write(f"- **{src}**: {count}")


def tab_catalogue() -> None:
    # Browses and downloads the Open Images dataset annotation records.
    cat = openimages.load_catalogue()

    if cat is None:
        st.warning("Catalogue not found. Building it downloads about 90 MB of CSV files and takes a minute.")
        if st.button("Build Catalogue"):
            with st.spinner("Downloading and building catalogue..."):
                new_cat = openimages.build_catalogue()
                openimages.save_catalogue(new_cat)
            st.rerun()
        return

    st.write(f"**Generated on:** {cat['generated']}")
    if st.button("Rebuild Catalogue"):
        with st.spinner("Rebuilding..."):
            new_cat = openimages.build_catalogue()
            openimages.save_catalogue(new_cat)
        st.rerun()

    col1, col2 = st.columns(2)
    with col1:
        txt_filter = st.text_input("Filter classes")
    with col2:
        min_box = st.selectbox("Minimum box size", [0.10, 0.15, 0.25], index=2)

    try:
        current_targets = targets.load_targets()
    except Exception:
        current_targets = []

    used_in = {}
    for t in current_targets:
        used_in.setdefault(t.cls, []).append(t.folder)

    table_data = []
    for cls_name, stats in cat["classes"].items():
        if txt_filter and txt_filter.lower() not in cls_name.lower():
            continue

        avail = openimages.available(stats, min_box)
        folders = ", ".join(used_in.get(cls_name, []))

        table_data.append({
            "class": cls_name,
            "val": stats["val"],
            "test": stats["test"],
            "total": stats["total"],
            "available": avail,
            "used in": folders
        })

    st.dataframe(table_data, use_container_width=True)


@st.fragment(run_every=1)
def live_download_view(selected_folders: list[str]) -> None:
    # Polls the events file automatically to refresh the live download dashboard
    events_list = runner.get_events()
    if not events_list:
        st.info("Waiting for run events...")
        if st.button("Stop download", type="primary"):
            runner.stop()
            st.rerun()
        return

    status = "running"
    class_stats = {}
    for ev in events_list:
        t = ev.get("type")
        if t in ("class_start", "class_progress", "class_end"):
            cls = ev.get("cls")
            if cls:
                class_stats[cls] = {"added": ev.get("added", 0), "quota": ev.get("quota", 0), "short": ev.get("short", False)}
        elif t in ("run_end", "run_error"):
            status = ev.get("status", "done")

    for cls, stat in class_stats.items():
        quota = stat["quota"]
        added = stat["added"]
        frac = min(1.0, added / quota) if quota > 0 else 0.0
        st.progress(frac, text=f"{cls}: {added}/{quota}")
        if stat.get("short"):
            st.warning(f"Class {cls} fell short: {added}/{quota}")

    st.write("#### Folder counts")
    for folder in selected_folders:
        folder_path = config.DATA_DIR / folder
        count = sum(1 for item in folder_path.iterdir() if item.is_file() and not item.name.startswith(".")) if folder_path.exists() else 0
        st.write(f"- {folder}: {count}")

    if status == "running":
        if st.button("Stop download", type="primary"):
            runner.stop()
            st.rerun()
    else:
        st.success(f"Run ended with status: {status}")

    with st.expander("Log Tail"):
        st.code(runner.log_tail(), language="text")


def tab_targets() -> None:
    # Edits targets.json quotas and checks them against the catalogue's availability.
    cat = openimages.load_catalogue()
    cat_names = openimages.class_names(cat) if cat else []

    if "targets_data" not in st.session_state or st.button("Reload from file"):
        try:
            loaded = targets.load_targets()
            st.session_state["targets_data"] = [
                {"folder": t.folder, "class": t.cls, "quota": t.quota, "min_area": t.min_area, "note": t.note}
                for t in loaded
            ]
        except Exception:
            st.session_state["targets_data"] = []
            st.warning("Could not load targets.json, starting empty.")

    st.write("### Add classes")
    with st.form("add_targets_form"):
        leaves = data.find_leaf_folders()
        existing = [r["folder"] for r in st.session_state["targets_data"]]
        all_dests = sorted(list(set(leaves + existing)))

        col1, col2 = st.columns(2)
        with col1:
            dest_sel = st.selectbox("Destination folder (existing)", ["(Type new below)"] + all_dests)
            dest_new = st.text_input("Or new folder name")
        with col2:
            cls_sel = st.multiselect("Catalogue classes", options=cat_names)

        col3, col4 = st.columns(2)
        with col3:
            add_quota = st.number_input("Quota", min_value=1, value=100, step=10)
        with col4:
            add_min_area = st.number_input("Min Area", min_value=0.05, max_value=0.50, value=0.25, step=0.05)

        if st.form_submit_button("Add"):
            target_folder = dest_new.strip() if dest_new.strip() else dest_sel
            if target_folder == "(Type new below)":
                st.error("Please select a folder or type a new one.")
            elif not cls_sel:
                st.error("Please select at least one class.")
            else:
                for c in cls_sel:
                    st.session_state["targets_data"].append({
                        "folder": target_folder,
                        "class": c,
                        "quota": add_quota,
                        "min_area": add_min_area,
                        "note": ""
                    })
                st.success(f"Added {len(cls_sel)} targets.")

    st.write("### Edit Targets")
    col_config = {
        "folder": st.column_config.TextColumn("Folder", required=True),
        "quota": st.column_config.NumberColumn("Quota", min_value=1, required=True),
        "min_area": st.column_config.NumberColumn("Min Area", min_value=0.05, max_value=0.50, step=0.05, required=True),
        "note": st.column_config.TextColumn("Note")
    }

    if cat_names:
        col_config["class"] = st.column_config.SelectboxColumn("Class", options=cat_names, required=True)
    else:
        col_config["class"] = st.column_config.TextColumn("Class", required=True)

    edited_data = st.data_editor(
        st.session_state["targets_data"],
        num_rows="dynamic",
        column_config=col_config,
        key="targets_editor",
        use_container_width=True
    )
    st.session_state["targets_data"] = edited_data

    if st.button("Save Targets"):
        try:
            t_objs = [
                targets.Target(
                    folder=r["folder"],
                    cls=r["class"],
                    quota=int(r["quota"]),
                    min_area=float(r["min_area"]),
                    note=r.get("note", "")
                ) for r in edited_data
            ]
            probs = targets.find_problems(t_objs)
            if probs:
                for p in probs:
                    st.error(p)
                st.stop()

            if cat:
                for t in t_objs:
                    stats = cat["classes"].get(t.cls)
                    if stats:
                        avail = openimages.available(stats, t.min_area)
                        if t.quota > avail:
                            st.warning(f"Quota for '{t.cls}' ({t.quota}) exceeds availability ({avail} at size {t.min_area}).")

            targets.save_targets(t_objs)
            st.success("Targets saved successfully!")
            st.info("Removing a row does not delete images that were already downloaded.")
        except Exception as e:
            st.error(f"Error saving targets: {e}")

    st.write("### Check Table")
    check_rows = []
    for r in edited_data:
        cls_name = r.get("class", "")
        quota = r.get("quota", 0)
        min_area = r.get("min_area", 0.25)

        if not cat or cls_name not in cat["classes"]:
            status = "unknown class"
            avail = 0
        else:
            stats = cat["classes"][cls_name]
            avail = openimages.available(stats, min_area)
            if quota > avail:
                status = "quota above available"
            else:
                status = "ok"

        check_rows.append({
            "folder": r.get("folder", ""),
            "class": cls_name,
            "quota": quota,
            "available": avail,
            "status": status
        })

    st.dataframe(check_rows, use_container_width=True)

    st.write("### Download images")
    all_folders = sorted(list(set(r.get("folder", "") for r in st.session_state["targets_data"] if r.get("folder"))))
    selected_folders = st.multiselect("Destination folders", options=all_folders, default=all_folders)

    uri = os.environ.get("FIFTYONE_DATABASE_URI", "mongodb://localhost:27017")
    parsed_uri = urlparse(uri)
    host = parsed_uri.hostname or "localhost"
    port = parsed_uri.port or 27017
    
    mongo_up = False
    try:
        with socket.create_connection((host, port), timeout=1.0):
            mongo_up = True
    except OSError:
        st.warning(f"Cannot reach MongoDB at {uri}. Start it with:")
        st.code("docker start fiftyone-mongo")

    is_running = runner.is_running()
    curr = runner.current()

    if is_running and curr and curr.get("kind") != "download":
        st.info("A training run is currently active. Stop it to start a download.")
    elif is_running and curr and curr.get("kind") == "download":
        live_download_view(selected_folders)
    else:
        if st.button("Start Download", disabled=not mongo_up or not selected_folders):
            runner.start_download(selected_folders)
            st.rerun()


def main() -> None:
    st.title("Dataset & Targets")

    t1, t2, t3 = st.tabs(["Current dataset", "Open Images catalogue", "Targets"])

    with t1:
        tab_current_dataset()
    with t2:
        tab_catalogue()
    with t3:
        tab_targets()


main()