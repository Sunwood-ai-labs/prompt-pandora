import streamlit as st
import os
from litellm import completion

# AWS Bedrock credentials
os.environ["AWS_ACCESS_KEY_ID"] = st.secrets["AWS_ACCESS_KEY_ID"]
os.environ["AWS_SECRET_ACCESS_KEY"] = st.secrets["AWS_SECRET_ACCESS_KEY"]
os.environ["AWS_REGION_NAME"] = st.secrets["AWS_REGION_NAME"]

def generate_prompt(task_or_prompt: str):
    response = completion(
        model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
        messages=[
            {
                "role": "system",
                "content": """
Given a task description or existing prompt, produce a detailed system prompt to guide a language model in completing the task effectively.

# Guidelines

- Understand the Task: Grasp the main objective, goals, requirements, constraints, and expected output.
- Minimal Changes: If an existing prompt is provided, improve it only if it's simple. For complex prompts, enhance clarity and add missing elements without altering the original structure.
- Reasoning Before Conclusions: Encourage reasoning steps before any conclusions are reached.
- Examples: Include high-quality examples if helpful, using placeholders [in brackets] for complex elements.
- Clarity and Conciseness: Use clear, specific language. Avoid unnecessary instructions or bland statements.
- Formatting: Use markdown features for readability.
- Preserve User Content: If the input task or prompt includes extensive guidelines or examples, preserve them entirely, or as closely as possible.
- Output Format: Explicitly specify the most appropriate output format, in detail.

The final prompt you output should adhere to the following structure:

[Concise instruction describing the task - this should be the first line in the prompt, no section header]

[Additional details as needed.]

[Optional sections with headings or bullet points for detailed steps.]

# Steps [optional]

[optional: a detailed breakdown of the steps necessary to accomplish the task]

# Output Format

[Specifically call out how the output should be formatted, be it response length, structure e.g. JSON, markdown, etc]

# Examples [optional]

[Optional: 1-3 well-defined examples with placeholders if necessary.]

# Notes [optional]

[optional: edge cases, details, and an area to call or repeat out specific important considerations]
"""
            },
            {
                "role": "user",
                "content": f"Task, Goal, or Current Prompt:\n{task_or_prompt}",
            },
        ]
    )
    return response.choices[0].message.content

def main():
    st.title("Prompt Pandora - プロンプト生成アプリ")

    task_or_prompt = st.text_area("タスクの説明または既存のプロンプトを入力してください：")

    if st.button("プロンプトを生成"):
        if task_or_prompt:
            with st.spinner("プロンプトを生成中..."):
                generated_prompt = generate_prompt(task_or_prompt)
                st.subheader("生成されたプロンプト：")
                st.markdown(generated_prompt)
        else:
            st.warning("タスクの説明または既存のプロンプトを入力してください。")

if __name__ == "__main__":
    main()
