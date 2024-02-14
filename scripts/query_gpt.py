import openai
from argparse import ArgumentParser
from pathlib import Path
import base64
import random
from tqdm import tqdm


class Message:
    def __init__(self, role="user", content=None):
        self.role = role
        if content is None:
            self.content = []

    def to_dict(self):
        return {"role": self.role, "content": self.content}

    def __repr__(self):
        return f"Message(role={self.role}, content={self.content})"

    def add_text(self, text):
        self.content.append({"type": "text", "text": text})
        return self

    def add_image(self, image_path):
        with open(image_path, "rb") as image_file:
            image = base64.b64encode(image_file.read()).decode("utf-8")
        self.content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image}",
                    "detail": "low",
                },
            }
        )
        return self


def query_gpt(args):
    client = openai.OpenAI()
    accuracies = []
    cost_dollars = estimate_cost_dollars(args.num_choices, args.num_tasks)
    print(f"Estimated cost: ${cost_dollars:.2f}")
    for num_choices in range(2, args.num_choices):
        print(f"Testing with {num_choices} choices.")
        scores = []
        for task in tqdm(range(args.num_tasks)):
            task_dir = args.benchmark_path / f"task_{task}"
            actual, predicted = run_task(task_dir, client, num_choices)
            scores.append(actual == predicted)
        accuracy = sum(scores) / len(scores)
        accuracies.append(accuracy)
        print(f"Accuracy: {accuracy}")

    with open(args.out_path / "results.csv", "w") as f:
        f.write("num_choices,accuracy\n")
        for num_choices, accuracy in zip(range(2, args.num_choices), accuracies):
            f.write(f"{num_choices},{accuracy}\n")


def estimate_cost_dollars(num_choices, num_tasks):
    images_per_task = 1 + sum(range(2, num_choices))
    tokens_per_task = 65 * images_per_task + 1
    return num_tasks * tokens_per_task / 1000 * 0.01


def run_task(task_dir, client, num_choices):
    messages, actual_answer = load_task(task_dir, num_choices)
    try:
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=messages,
        )
    except openai.BadRequestError as e:
        print(e.message)
        predicted_answer = -1
    predicted_answer = int(response.choices[0].message.content.strip())
    return actual_answer, predicted_answer


def load_task(task_dir, num_choices):
    messages = []
    prompt = f"""
    You will be shown a query image containing a black and white shape on a gray
    background. You will then be shown {num_choices} images, one of which is the same
    shape as the query image, but rotated, translated, and scaled. Please select the
    image that matches the query image. Please only select one image. Please only
    output a single number between 0 and {num_choices - 1} (inclusive) indicating
    your choice.
    """
    messages.append(Message().add_text(prompt).to_dict())
    messages.append(Message().add_image(task_dir / "query.png").to_dict())

    image_paths = [
        task_dir / "correct.png",
        *(task_dir / f"incorrect_{i}.png" for i in range(num_choices - 1)),
    ]
    random.shuffle(image_paths)
    actual_answer = image_paths.index(task_dir / "correct.png")

    message = Message()
    for img_path in image_paths:
        message.add_image(img_path)
    messages.append(message.to_dict())
    return messages, actual_answer


if __name__ == "__main__":
    root = Path(__file__).parent.parent
    parser = ArgumentParser()
    parser.add_argument(
        "--benchmark_path",
        type=int,
        default=root / "img/gpt_benchmark",
        help="Path to the benchmark.",
    )
    parser.add_argument(
        "--out_path",
        type=int,
        default=root / "results",
        help="Path to save the results.",
    )
    parser.add_argument(
        "--num_tasks", type=int, default=10, help="Number of tasks to run."
    )
    parser.add_argument(
        "--num_choices", type=int, default=4, help="Maximum number of choices to test."
    )
    args = parser.parse_args()
    query_gpt(args)
