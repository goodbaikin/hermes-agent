import time

from hermes_state import SessionDB


def test_list_sessions_projection_preserves_tip_lineage_metadata(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("parent", source="webui", profile="webui-lite-eng", model="gpt-5.5")
    db.set_session_title("parent", "Compressed chat")
    db.append_message("parent", role="user", content="before compression")
    db.end_session("parent", "compression")
    time.sleep(0.01)
    db.create_session(
        "child",
        source="api_server",
        profile="webui-lite-eng",
        model="gpt-5.5",
        parent_session_id="parent",
    )
    db.set_session_title("child", "Compressed chat #2")

    items = db.list_sessions_rich(limit=10)

    assert len(items) == 1
    item = items[0]
    assert item["id"] == "child"
    assert item["title"] == "Compressed chat #2"
    assert item["source"] == "api_server"
    assert item["profile"] == "webui-lite-eng"
    assert item["parent_session_id"] == "parent"
    assert item["_lineage_root_id"] == "parent"
