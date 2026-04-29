from openai import OpenAI
import os
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd

from prompts import prompt_extra_18
from prompts import prompt_orig_15
from prompts import prompt_gen_14


PROMPTS = {
    "extra18": prompt_extra_18,
    "orig15": prompt_orig_15,
    "gen14": prompt_gen_14,
}


def parse_arguments():
    parser = ArgumentParser()

    parser.add_argument(
        "--api_key",
        type=str,
        default=os.getenv("OPENAI_API_KEY"),
        help="OpenAI API key. You can also set it as OPENAI_API_KEY."
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Folder containing .txt fairy-tale files."
    )

    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Excel file to create, for example results/answers.xlsx."
    )

    parser.add_argument(
        "--prompt",
        type=str,
        choices=PROMPTS.keys(),
        default="gen14",
        help="Prompt to use: extra18, orig15, or gen14."
    )

    parser.add_argument(
        "--motif_count",
        type=int,
        default=15,
        help="Number of motif columns to save."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    if not args.api_key:
        raise ValueError(
            "No API key provided. Use --api_key or set OPENAI_API_KEY."
        )

    client = OpenAI(api_key=args.api_key)

    prompt = PROMPTS[args.prompt]

    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    data = []

    for tale_path in input_dir.glob("*.txt"):
        with open(tale_path, "r", encoding="utf-8") as f_story:
            fairy_tale = f_story.read()

        completion = client.chat.completions.create(
            model="gpt-4.5-preview",
            temperature=0.3,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant, skilled in finding motifs in fairy-tales. "
                        "Follow the instructions carefully."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt + fairy_tale,
                },
            ],
        )

        response = completion.choices[0].message.content.strip()
        answers = response.split("\n")

        formatted_answers = []

        for answer in answers:
            parts = answer.split(".", 1)

            if len(parts) > 1:
                formatted_answers.append(parts[1].strip())
            else:
                formatted_answers.append(answer.strip())

        data.append([tale_path.name] + formatted_answers[:args.motif_count])

        print(f"Processed: {tale_path.name}")
        print(formatted_answers)

    columns = ["Title"] + [f"{i}." for i in range(1, args.motif_count + 1)]

    df = pd.DataFrame(data, columns=columns)
    df.to_excel(output_file, index=False)

    print(f"Excel file '{output_file}' generated successfully.")