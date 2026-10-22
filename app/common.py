"""
Shared pieces of the dashboard: path setup, the sidebar and cached loaders.

Pages read saved runs through ``farmrl.runs`` and draw with ``farmrl.viz.interactive``;
the only place the environment is run is ``farmrl.live`` (the "Simulate" page).
"""

import sys
from pathlib import Path

# Get the project's root directory.
# __file__ refers to the current Python file.
# parents[1] moves up two directory levels from this file.
ROOT = Path(__file__).resolve().parents[1]

# Add the project's src directory to Python's module search path.
# This allows Python to import project modules such as farmrl.
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

# Import Streamlit to build the dashboard UI.
import streamlit as st  # noqa: E402

# Import modules for running simulations and accessing saved experiment runs.
# live handles trained agents and live simulation functionality.
# runs handles saved experiment data.
from farmrl import live, runs  # noqa: E402

# Import styling helpers for displaying policy names.
from farmrl.viz import style  # noqa: E402


# Cache the loaded experiment run.
# Streamlit reuses the cached object when the same name and timestamp are provided.
# This avoids repeatedly loading the same run from disk.
@st.cache_resource(show_spinner=False)
def _load_run(name, stamp):
    return runs.get(name)


# Cache the trained agent associated with an experiment run.
# Streamlit displays a loading message while the agent is being loaded.
# The stamp argument ensures the cache can be refreshed when the run changes.
@st.cache_resource(show_spinner="Loading the trained agent...")
def _load_agent(name, stamp):
    # Retrieve the saved run and load its trained agent.
    return live.load_agent(runs.get(name))


def _stamp(name):
    """Return the modification timestamp of a run's metadata file.

    The timestamp changes when the run is rebuilt. Passing it to cached
    functions helps Streamlit detect when it should load fresh objects.
    """
    metadata_path = runs.RUNS_DIR / name / "metadata.json"

    # Return the last modification time of metadata.json.
    return metadata_path.stat().st_mtime


def agent_for(run):
    """Return the cached trained agent for the selected experiment run."""
    # Use the run name and its current timestamp as the cache key.
    return _load_agent(run.name, _stamp(run.name))


# Set up the page layout and sidebar.
# Return the selected experiment run and policies.
# If no saved runs exist, display instructions and stop rendering the page.
def setup(title):
    """Configure the dashboard and sidebar.

    Returns:
        tuple: (run, selected policies)

    Stops the page with instructions if no experiment run exists.
    """

    # Configure the browser tab title, page icon, and wide layout.
    st.set_page_config(
        page_title=f"FarmRL · {title}",
        page_icon="🌾",
        layout="wide"
    )

    # Retrieve the names of all available saved experiment runs.
    # The list is expected to be ordered with the newest runs first.
    names = runs.list_runs()

    # If no experiment runs exist, explain how to create one.
    if not names:
        st.title("FarmRL dashboard")

        # Display a warning in the dashboard.
        st.warning("No experiment run found in `results/runs/`.")

        # Tell the user how to generate a run using the training pipeline.
        st.markdown("Build one with the pipeline, then reload this page:")

        # Show example terminal commands.
        # The first command runs the full pipeline.
        # The second runs a faster smoke test.
        st.code(
            "conda run -n TogAI python scripts/run_all.py --run-name v1\n"
            "# or a fast smoke run\n"
            "conda run -n TogAI python scripts/run_all.py --run-name smoke --quick"
        )

        # Stop executing this page because there are no runs to display.
        st.stop()

    # Create the sidebar containing experiment settings.
    with st.sidebar:
        st.header("Run")

        # Let the user choose which saved experiment run to inspect.
        # The selection is stored in Streamlit's session state under "run_name".
        name = st.selectbox(
            "Experiment run",
            names,
            key="run_name",
            help="Every folder in results/runs/, newest first"
        )

        # Load the selected run, reusing the cache if possible.
        # A changed metadata timestamp produces a different cache key.
        run = _load_run(name, _stamp(name))

        # Extract the metadata dictionary for the selected run.
        meta = run.meta

        # Display the run's creation time and indicate whether it was a quick run.
        st.caption(
            f"created {meta['created'].replace('T', ' ')}"
            + (" · **quick run**" if meta.get("quick") else "")
        )

        # Create an expandable section for additional experiment details.
        with st.expander("Run metadata"):

            # Display important configuration and experiment information.
            st.write({
                "timesteps": meta["timesteps"],
                "training seeds": meta["train_seeds"],
                "rollout seeds per season": meta["rollout_seeds"],
                "weather site": meta["weather_site"],

                # Display the first and last training seasons.
                "train seasons": (
                    f"{meta['train_seasons'][0]}–"
                    f"{meta['train_seasons'][-1]}"
                ),

                # Display the seasons used for testing.
                "test seasons": meta["test_seasons"],

                # Display the first 10 characters of the Git commit hash.
                # If the hash is missing, display "n/a".
                "git": (meta.get("git_hash") or "n/a")[:10]
            })

        # Retrieve all policies available for this run, including auxiliary ones.
        available = run.policies(include_aux=True)

        # Create a unique session-state key for the policy selection of this run.
        # Different experiment runs can therefore have separate selections.
        key = f"policies_{name}"

        # Allow the user to select multiple policies for comparison.
        # By default, select the standard policies returned by run.policies().
        # format_func=style.label controls how policy names appear in the UI.
        chosen = st.multiselect(
            "Policies",
            available,
            default=run.policies(),
            key=key,
            format_func=style.label
        )

    # If the user deselects every policy, ask them to select at least one.
    if not chosen:
        st.info("Select at least one policy in the sidebar.")

        # Stop rendering the page until a policy is selected.
        st.stop()

    # Return the selected experiment run and policy names to the calling page.
    return run, chosen


# Helper function to display Plotly charts across the full available page width.
# Other dashboard pages can call chart(fig) instead of repeating this code.
def chart(fig):
    # Render the Plotly figure in Streamlit.
    # use_container_width=True makes the chart fill its available container.
    st.plotly_chart(fig, use_container_width=True)