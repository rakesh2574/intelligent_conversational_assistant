"""
TIMING ANALYSIS MODULE
Add this to your existing code WITHOUT changing any logic
Just tracks timing at each step of query processing (Flow B)

Usage:
1. Add this file as timing_analysis.py
2. Import at top of streamlit_app_prompt_assist.py
3. Add timing points (shown below)
4. Display table after query completes
"""

import time
from datetime import datetime
from contextlib import contextmanager
import streamlit as st
import pandas as pd

# Global timing storage for current query
QUERY_TIMING = {}
QUERY_START_TIME = None


def reset_timing():
    """Reset timing for new query"""
    global QUERY_TIMING, QUERY_START_TIME
    QUERY_TIMING = {}
    QUERY_START_TIME = time.time()


@contextmanager
def time_step(step_name, description=""):
    """
    Context manager to time a step without changing code logic

    Usage:
        with time_step("Step Name", "Description"):
            # Your existing code here
            result = some_function()
    """
    step_start = time.time()

    try:
        yield
    finally:
        step_end = time.time()
        elapsed = step_end - step_start

        QUERY_TIMING[step_name] = {
            'duration': elapsed,
            'description': description,
            'timestamp': datetime.now().strftime('%H:%M:%S.%f')[:-3]
        }


def display_timing_table(total_label="Total Query Time"):
    """
    Display comprehensive timing table
    Shows each step with time, percentage, and visual bar
    """
    if not QUERY_TIMING:
        return

    # Calculate total time
    total_time = time.time() - QUERY_START_TIME if QUERY_START_TIME else sum(
        step['duration'] for step in QUERY_TIMING.values()
    )

    # Prepare data for table
    table_data = []

    for step_name, step_info in QUERY_TIMING.items():
        duration = step_info['duration']
        percentage = (duration / total_time) * 100 if total_time > 0 else 0

        # Create visual bar
        bar_length = int(percentage / 2)  # Scale to 50 chars max
        bar = '█' * bar_length

        # Determine color emoji based on time
        if percentage > 40:
            emoji = '🔴'  # Red - very slow
        elif percentage > 20:
            emoji = '🟡'  # Yellow - moderate
        elif percentage > 10:
            emoji = '🟠'  # Orange - noticeable
        else:
            emoji = '🟢'  # Green - fast

        table_data.append({
            'Step': step_name,
            'Description': step_info['description'],
            'Time': f"{duration:.3f}s",
            'Percentage': f"{percentage:.1f}%",
            'Visual': f"{emoji} {bar}",
            'Timestamp': step_info['timestamp'],
            '_duration': duration,  # For sorting
            '_percentage': percentage
        })

    # Sort by duration (longest first)
    table_data.sort(key=lambda x: x['_duration'], reverse=True)

    # Create DataFrame
    df = pd.DataFrame(table_data)

    # Display timing table
    st.markdown("---")
    st.subheader("⏱️ Query Performance Analysis")

    # Summary metrics at top
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Time", f"{total_time:.2f}s")

    with col2:
        slowest = max(table_data, key=lambda x: x['_duration'])
        st.metric("Slowest Step", slowest['Step'][:20], f"{slowest['_duration']:.2f}s")

    with col3:
        fastest = min(table_data, key=lambda x: x['_duration'])
        st.metric("Fastest Step", fastest['Step'][:20], f"{fastest['_duration']:.3f}s")

    with col4:
        st.metric("Steps Tracked", len(table_data))

    # Display detailed table
    display_df = df[['Step', 'Description', 'Time', 'Percentage', 'Visual', 'Timestamp']]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    # Show bottleneck analysis
    bottlenecks = [item for item in table_data if item['_percentage'] > 20]

    if bottlenecks:
        st.warning(f"⚠️ **Performance Bottlenecks Detected:** {len(bottlenecks)} step(s) taking >20% of total time")

        for bottleneck in bottlenecks:
            with st.expander(f"🔴 {bottleneck['Step']} - {bottleneck['Time']} ({bottleneck['Percentage']})"):
                st.write(f"**Description:** {bottleneck['Description']}")
                st.write(f"**Time:** {bottleneck['Time']}")
                st.write(f"**Impact:** {bottleneck['Percentage']} of total query time")

                # Suggestions based on step name
                if 'enhance' in bottleneck['Step'].lower():
                    st.info("💡 **Suggestion:** Consider skipping response enhancement or making it optional")
                elif 'llm' in bottleneck['Step'].lower() or 'response' in bottleneck['Step'].lower():
                    st.info("💡 **Suggestion:** Use streaming to improve perceived speed, or reduce retrieved chunks")
                elif 'retrieve' in bottleneck['Step'].lower():
                    st.info("💡 **Suggestion:** Reduce k value (number of chunks retrieved) from 8 to 5")
                elif 'load' in bottleneck['Step'].lower():
                    st.info("💡 **Suggestion:** Vectorstore should stay in memory after first load")
    else:
        st.success("✅ No major bottlenecks detected! All steps are reasonably fast.")

    # Export option
    csv = display_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Timing Report (CSV)",
        data=csv,
        file_name=f"timing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )


def get_timing_summary():
    """Get timing summary as dict (for logging/debugging)"""
    if not QUERY_TIMING:
        return {}

    total_time = time.time() - QUERY_START_TIME if QUERY_START_TIME else sum(
        step['duration'] for step in QUERY_TIMING.values()
    )

    return {
        'total_time': total_time,
        'steps': len(QUERY_TIMING),
        'breakdown': {name: info['duration'] for name, info in QUERY_TIMING.items()}
    }


# ============================================
# HOW TO INTEGRATE INTO YOUR EXISTING CODE
# ============================================

"""
STEP-BY-STEP INTEGRATION:

1. Save this file as timing_analysis.py in your project folder

2. Add import at top of streamlit_app_prompt_assist.py:

   from timing_analysis import reset_timing, time_step, display_timing_table

3. Find your query handling code (around line 1800-1900 in main()):

   # BEFORE:
   if ask_button and question.strip():
       if not openai_api_key:
           st.error("Please enter your OpenAI API key")
       elif not st.session_state.conversation_chain:
           st.error("Knowledge base not ready")
       else:
           with st.spinner("Generating answer..."):
               response = st.session_state.conversation_chain.invoke({'question': question})
               # ... rest of code

4. Wrap it with timing (see example below)
"""