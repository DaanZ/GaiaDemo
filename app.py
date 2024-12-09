import streamlit as st

from templates.meeting_data import extract_product_recommendation, extract_article_suggestions, extract_case_studies, \
    extract_meeting_takeaways, summarize_meeting

# Initialize session state for company data
if "company_data" not in st.session_state:
    st.session_state["company_data"] = {}

# Title
st.title("Gaia Hubspot Demo: Meeting Transcript Analyzer")

# Multiline input for meeting transcript
meeting_transcript = st.text_area("Enter the meeting transcript here:", height=200)

# Button to trigger analysis
if st.button("Analyze Transcript"):
    if not meeting_transcript.strip():
        st.warning("Please enter a valid meeting transcript before analyzing.")
    else:
        with st.spinner("Analyzing the meeting transcript..."):
            # Step 1: Summarize the meeting
            meeting_summary = summarize_meeting(meeting_transcript)

            # Step 2: Extract meeting takeaways
            st.session_state["company_data"] = extract_meeting_takeaways(st.session_state["company_data"],
                                                                         meeting_summary)

            # Step 3: Extract case studies
            st.session_state["company_data"] = extract_case_studies(st.session_state["company_data"], meeting_summary)

            # Step 4: Extract article suggestions
            st.session_state["company_data"] = extract_article_suggestions(st.session_state["company_data"],
                                                                           meeting_summary)

            # Step 5: Extract product recommendations
            st.session_state["company_data"] = extract_product_recommendation(st.session_state["company_data"],
                                                                              meeting_summary)

        st.success("Analysis complete!")

# Display extracted information
st.subheader("Extracted Company Data")
if st.session_state["company_data"]:
    for key, value in st.session_state["company_data"].items():
        st.text_input(key, value)
else:
    st.info("No data to display yet. Please analyze a transcript.")
