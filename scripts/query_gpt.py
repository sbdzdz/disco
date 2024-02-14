from openai import OpenAI
from argparse import ArgumentParser
from pathlib import Path
import base64
import random


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
    client = OpenAI()
    scores = []
    for task_dir in args.path.glob("task_*"):
        actual, predicted = run_task(task_dir, args, client)
        scores.append(actual == predicted)
        print(f"Actual: {actual}, Predicted: {predicted}")
    print(f"Accuracy: {sum(scores) / len(scores)}")


def run_task(task_dir, args, client):
    messages, actual_answer = load_task(task_dir, args)
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=messages,
        max_tokens=args.max_tokens,
    )
    predicted_answer = parse_response(response)
    return actual_answer, predicted_answer


def parse_response(response):
    return int(response.choices[0].message.content.strip())


def load_task(task_dir, args):
    messages = []
    prompt = f"""
    You will be shown a query image containing a black and white shape on a gray
    background. You will then be shown {args.num_answers} images, one of which is
    the same shape as the query image, but rotated, translated, or scaled. Please
    select the image that matches the query image.  Please only select one image.
    Please only output a single number between 0 and {args.num_answers - 1}
    (inclusive) indicating your choice.
    """
    messages.append(Message().add_text(prompt).to_dict())
    messages.append(Message().add_image(task_dir / "query.png").to_dict())

    image_paths = [
        task_dir / "correct.png",
        *(task_dir / f"incorrect_{i}.png" for i in range(args.num_answers - 1)),
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
        "--path",
        type=int,
        default=root / "img/gpt_benchmark",
        help="Path to the benchmark.",
    )
    parser.add_argument("--num_answers", type=int, default=4, help="Number of answers.")
    parser.add_argument("--max_tokens", type=int, default=10, help="Max output tokens.")
    args = parser.parse_args()
    query_gpt(args)
