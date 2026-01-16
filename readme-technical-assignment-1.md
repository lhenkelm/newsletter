# 📩 Assignment: Automated AI Newsletter Generator

During the technical interview round, we will ask you to work on an assignment. During the interview, you will get time to work on it. This document serves as a general introduction to the problem statement. Feel free to read this as preparation for the interview. We don't expect you to deliver anything prior to the interview. This document is intended to help you properly prepare for the assignment, and perhaps you can already prepare questions about any unclear points. During the interview, we will provide the dataset that you will be using for the assignment, as well as the deliverables we expect from you.

## Context
Odido is actively exploring and applying **AI/LLM technologies** across our products and internal workflows.  
To keep up with this fast-moving field, we want to create a **weekly AI newsletter** for Odido employees.  

A newsletter helps us:  
- **Stay ahead of industry trends** by surfacing the latest AI developments  
- **Discover new use cases** that could inspire or accelerate our own projects  
- **Reduce information overload** by filtering and summarizing only the most relevant updates  
- **Share knowledge internally** in a concise, digestible, and engaging format  

Your task is to design and implement a pipeline that automatically generates this newsletter.

We will provide you with a dataset during the interview.

---

## Your Task
Build an **end-to-end pipeline** that:

1. **Ingests the dataset**  
   - Use the dataset which we will provide during the interview.

2. **Processes & Analyzes**  
   - Identify the most relevant or trending items for the current week.  
   - Categorize items into meaningful sections (e.g., *Research Highlights*, *Industry News*, *Cool Use Cases*).  
   - Summarize use cases into clear, concise, newsletter-friendly text.  

3. **Generates a Newsletter**  
   - Produce the newsletter in **Markdown format**.  
   - Include:  
     - A short introduction  
     - Categorized sections with summaries and links  
     - A short closing section  

4. **Automate the Workflow**  
   - Ensure the pipeline can be run on a weekly schedule.  
   - Handle updates gracefully (avoid duplicates; include only fresh items).  

---

## Requirements
- Use an **LLM** (e.g., OpenAI API or an open-source model) for summarization/writing.  
- Write **modular, clean, and well-documented code**.  
- Provide a **sample newsletter** generated from the dataset.  
- Provide clear instructions on how to run your solution locally.  

**Bonus points if you:**  
- Deploy the pipeline with automation (e.g., GitHub Actions or another scheduler).  
- Add personalization (e.g., selecting sections based on reader profile: technical or non-technical).  
- Evaluate or improve newsletter quality.  

---

## Deliverables
We will provide you with the deliverables during the interview.
---

## Evaluation Criteria
We’ll evaluate based on:  
- **Data Handling** – How well you ingest, filter, and process the Hugging Face dataset  
- **LLM Usage** – Effective prompt design and output quality  
- **Code Quality** – Structure, clarity, documentation, reproducibility  
- **Automation** – Ability to set up a repeatable, scheduled workflow  
- **Creativity** – Newsletter structure, style, polish  
- **Bonus** – Deployment, personalization, or evaluation methods  

---

## Time Expectation
We estimate this assignment will take **3–4 hours**. Please focus on **building a working prototype**, not on perfection.  

---

✅ **What you’ll get from us**  
- The dataset

✅ **What we expect from you**  
- A working pipeline that generates a weekly newsletter.  
- Sample newsletter outputs in **Markdown**.  
- Your code and documentation.  

Feel free to reach out to us if you have any problems or questions: zhengru.shen@odido.nl, jeroen.esseveld@odido.nl
