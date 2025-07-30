## AI Researcher Interview

Goal: Evaluate and fine-tune language models for coding tasks. You will fine-tune

##Qwen-7B-Instruct on an open-source dataset of multi-domain (e.g., math, coding)
reasoning traces and evaluate performance on a Codeforces-style benchmark. We are
assessing your ability to (a) design evaluations, (b) perform supervised fine-tuning
(SFT), and (c) make and justify training/data decisions under time and budget
constraints that are consistent with the goals and scope.
Part 1. Benchmark coding performance
● Evaluate a Qwen-7B-Instruct model on Codeforces
● Report prompting format & any data-processing/cleaning decisions.
Part 2. Fine-tune on a open-corpus dataset of 350K reasoning traces
● Report training decisions you make, including, but not limited to the following
○ Document any decisions regarding data cleaning, processing or filtering
○ Training hyper-parameter choices, e.g., learning-rate, batch-size
● What metrics do you measure to assess training performance and downstream
performance on the codeforces benchmark
Part 3. Re-evaluate benchmark performance
Part 4. [Bonus] Open up an inference endpoint that we can query
● You can use Hugging Face or vLLM