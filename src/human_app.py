import streamlit as st
import json, os, time, csv, random
from datetime import datetime

ACTIONS = ["LTURN", "RTURN", "RUN", "WALK", "JUMP", "LOOK"]

def load_jsonl(path):
    return [json.loads(x) for x in open(path, "r", encoding="utf-8")]

def main():
    st.set_page_config(page_title="Human Baseline — Commands→Actions", layout="centered")
    st.title("Human Baseline: Commands → Actions")
    st.write("Click actions to produce the sequence. Use **Backspace** to undo last. Submit when done.")

    test_file = st.text_input("Path to test JSONL", value="data/generated/len_test.jsonl")
    user_id   = st.text_input("Your ID (initials)", value="")
    out_csv   = st.text_input("Output CSV", value="results/human.csv")

    # lazy init
    if "rows" not in st.session_state and os.path.exists(test_file):
        st.session_state.rows = load_jsonl(test_file)
        random.shuffle(st.session_state.rows)
        st.session_state.idx = 0
        st.session_state.actions = []
        st.session_state.started_at = None

    col1, _ = st.columns([1,1])
    if col1.button("Load / Reload examples"):
        if os.path.exists(test_file):
            st.session_state.rows = load_jsonl(test_file)
            random.shuffle(st.session_state.rows)
            st.session_state.idx = 0
            st.session_state.actions = []
            st.session_state.started_at = None

    if "rows" not in st.session_state:
        st.warning("Provide a valid test JSONL path and click 'Load / Reload examples'.")
        return

    if st.session_state.idx >= len(st.session_state.rows):
        st.success("All examples completed. Thanks!")
        return

    ex = st.session_state.rows[st.session_state.idx]
    tokens   = ex["tokens"]
    gold     = ex["actions"]
    template = ex.get("template", "?")

    if st.session_state.started_at is None:
        st.session_state.started_at = time.time()

    st.subheader("Command (tokens)")
    st.code(" ".join(tokens))

    st.subheader("Your actions")
    st.write(st.session_state.actions if st.session_state.actions else "—")

    # action buttons
    cols = st.columns(len(ACTIONS))
    for i, a in enumerate(ACTIONS):
        if cols[i].button(a):
            st.session_state.actions.append(a)

    c1, c2, c3 = st.columns(3)
    if c1.button("Backspace") and st.session_state.actions:
        st.session_state.actions.pop()

    if c2.button("Skip"):
        st.session_state.idx += 1
        st.session_state.actions = []
        st.session_state.started_at = None
        st.rerun()

    correct = st.session_state.actions == gold
    submitted = c3.button("Submit")

    st.caption(f"Gold length: {len(gold)} | Your length: {len(st.session_state.actions)}")
    if correct:
        st.success("Exact match ✓")

    if submitted:
        duration = time.time() - st.session_state.started_at
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        write_header = not os.path.exists(out_csv)
        with open(out_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow([
                    "timestamp","user_id","test_file","index","template",
                    "tokens","gold","human","correct","duration_sec"
                ])
            w.writerow([
                datetime.utcnow().isoformat(timespec="seconds"),
                user_id, test_file, st.session_state.idx, template,
                " ".join(tokens), " ".join(gold), " ".join(st.session_state.actions),
                int(correct), f"{duration:.2f}"
            ])
        st.session_state.idx += 1
        st.session_state.actions = []
        st.session_state.started_at = None
        st.rerun()

if __name__ == "__main__":
    main()

