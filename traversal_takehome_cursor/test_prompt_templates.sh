#!/bin/bash

# Script to test different prompt templates for run_codeforces.py
# Tests 4 different prompts with random samples of the dataset

set -e  # Exit on error

# Configuration
SCRIPT_NAME="run_codeforces.py"
ORIGINAL_SCRIPT="run_codeforces.py"
NUM_WORKERS=2
TIMEOUT=120
WANDB_PROJECT="codeforces-prompt-comparison"
WANDB_ENTITY=""  # Set this if you have a specific wandb entity
BASE_OUTPUT_DIR="prompt_comparison_results"
SEED_BASE=42

# Create output directory
mkdir -p "$BASE_OUTPUT_DIR"

echo "=========================================="
echo "Codeforces Prompt Template Comparison Test"
echo "=========================================="
echo "Testing 4 different prompt templates with random dataset samples"
echo "Base output directory: $BASE_OUTPUT_DIR"
echo "Workers per test: $NUM_WORKERS"
echo "Timeout per problem: ${TIMEOUT}s"
echo ""

# Function to create a script variant with a specific prompt template
create_script_variant() {
    local variant_name="$1"
    local prompt_template="$2"
    local output_file="run_codeforces_${variant_name}.py"
    
    echo "Creating script variant: $output_file"
    
    # Copy the original script
    cp "$ORIGINAL_SCRIPT" "$output_file"
    
    # Replace the PROMPT_TEMPLATE with our new one
    # Using a here document to handle multi-line replacement
    python3 << EOF
import re

# Read the original file
with open('$output_file', 'r') as f:
    content = f.read()

# Define the new prompt template
new_prompt = '''$prompt_template'''

# Replace the PROMPT_TEMPLATE variable
# Find the pattern: PROMPT_TEMPLATE = """..."""
pattern = r'PROMPT_TEMPLATE = """.*?"""'
replacement = f'PROMPT_TEMPLATE = """{new_prompt}"""'

# Perform the replacement
new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

# Write back to file
with open('$output_file', 'w') as f:
    f.write(new_content)

print(f"Updated prompt template in $output_file")
EOF
}

# Function to run a test with a specific script variant
run_test() {
    local variant_name="$1"
    local description="$2"
    local script_file="run_codeforces_${variant_name}.py"
    local seed=$((SEED_BASE + $3))  # Different seed for each test
    local output_dir="${BASE_OUTPUT_DIR}/${variant_name}"
    
    echo ""
    echo "----------------------------------------"
    echo "Running test: $variant_name"
    echo "Description: $description"
    echo "Script: $script_file"
    echo "Seed: $seed"
    echo "Output dir: $output_dir"
    echo "----------------------------------------"
    
    # Run the test
    if [ -n "$WANDB_ENTITY" ]; then
        python3 "$script_file" \
            --wandb_project "$WANDB_PROJECT" \
            --wandb_entity "$WANDB_ENTITY" \
            --wandb_group "prompt-comparison-$(date +%Y%m%d-%H%M%S)" \
            --num_workers "$NUM_WORKERS" \
            --seed "$seed" \
            --timeout "$TIMEOUT" \
            --output_dir "$output_dir" \
            --subset "verifiable" \
            --split "test"
    else
        python3 "$script_file" \
            --wandb_project "$WANDB_PROJECT" \
            --wandb_group "prompt-comparison-$(date +%Y%m%d-%H%M%S)" \
            --num_workers "$NUM_WORKERS" \
            --seed "$seed" \
            --timeout "$TIMEOUT" \
            --output_dir "$output_dir" \
            --subset "verifiable" \
            --split "test"
    fi
    
    echo "Test $variant_name completed!"
}

# Define the 4 different prompt templates

# Template 1: Detailed Analysis (Original style)
PROMPT_1="You will be given a competitive programming problem.
Analyze the maximum input constraints and identify the optimal algorithmic approach and data structures needed to process the largest possible test cases within the time and memory limits, then explain why your chosen implementation strategy is the most efficient solution. Please reason step by step about your solution approach, then provide a complete implementation in Python 3 that is thoroughly optimized for both speed and memory usage.

Implement only the solve function, which takes one argument, the input string, and returns the output string.

Put your final solution within a single code block:
\`\`\`python
def solve(inp: str) -> str:
    <your code here>
\`\`\`
# Problem statement
{description}
# Input
{input_format}
# Output
{output_format}"

# Template 2: Concise Direct (Similar to fast version)
PROMPT_2="You are solving a Codeforces problem. Implement **only**

the function: 
\`\`\`python
def solve(inp: str) -> str:
    # Return the answer for this instance.
\`\`\`

Use only the Python standard library. Be concise and don't elaborate too much just output the solution.

---
# Problem statement
{description}
# Input
{input_format}
# Output
{output_format}"

# Template 3: Step-by-step approach
PROMPT_3="Solve this Codeforces problem step by step:

1. Read and understand the problem constraints
2. Identify the key algorithmic pattern
3. Choose appropriate data structures  
4. Implement an efficient solution

Provide your solution as a single function:
\`\`\`python
def solve(inp: str) -> str:
    # Your solution here
\`\`\`

Use only Python standard library. Focus on correctness and efficiency.

# Problem statement
{description}
# Input
{input_format}
# Output
{output_format}"

# Template 4: Competitive programming focused
PROMPT_4="COMPETITIVE PROGRAMMING PROBLEM

Constraints and efficiency are critical. Analyze time/space complexity before coding.

Required output format:
\`\`\`python  
def solve(inp: str) -> str:
    # Efficient solution
\`\`\`

Key considerations:
- Handle edge cases
- Optimize for large inputs
- Use appropriate algorithms/data structures
- Python standard library only

PROBLEM:
{description}

INPUT FORMAT:
{input_format}

OUTPUT FORMAT:
{output_format}"

echo "Creating script variants with different prompt templates..."

# Create the 4 script variants
create_script_variant "detailed" "$PROMPT_1"
create_script_variant "concise" "$PROMPT_2" 
create_script_variant "stepwise" "$PROMPT_3"
create_script_variant "competitive" "$PROMPT_4"

echo ""
echo "All script variants created successfully!"
echo ""

# Run tests with each variant
echo "Starting prompt template comparison tests..."

run_test "detailed" "Detailed analysis approach (original)" 0
run_test "concise" "Concise direct approach" 1  
run_test "stepwise" "Step-by-step structured approach" 2
run_test "competitive" "Competitive programming focused" 3

echo ""
echo "=========================================="
echo "All tests completed!"
echo "=========================================="
echo "Results are saved in: $BASE_OUTPUT_DIR/"
echo "Check your wandb project '$WANDB_PROJECT' for metrics comparison"
echo ""

# Clean up temporary script files
echo "Cleaning up temporary script files..."
rm -f run_codeforces_detailed.py
rm -f run_codeforces_concise.py  
rm -f run_codeforces_stepwise.py
rm -f run_codeforces_competitive.py

echo "Cleanup completed!"
echo ""
echo "Summary of test configurations:"
echo "- detailed: Original detailed analysis approach"
echo "- concise: Short, direct instructions"  
echo "- stepwise: Structured step-by-step approach"
echo "- competitive: Competition-focused with constraints emphasis"
echo ""
echo "Each test used a different random seed for dataset sampling:"
echo "- detailed: seed 42"
echo "- concise: seed 43" 
echo "- stepwise: seed 44"
echo "- competitive: seed 45"