import os
import time

import requests
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# File containing the API key
API_KEY_FILE = "./lambda_cloud/lambda-key.txt"

# Lambda Cloud API URL
API_URL = "https://cloud.lambdalabs.com/api/v1/instance-types"

COMPATIBLE_GPUS = ["RTX 6000", "A6000"]

# Slack configuration
SLACK_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_USER_ID = os.environ.get("SLACK_USER_ID")  # Your Slack user ID


def read_api_key():
    try:
        with open(API_KEY_FILE, "r") as file:
            return file.read().strip()
    except FileNotFoundError:
        print(f"Error: API key file '{API_KEY_FILE}' not found.")
        return None
    except IOError:
        print(f"Error: Unable to read the API key file '{API_KEY_FILE}'.")
        return None


def check_availability(api_key):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        response = requests.get(API_URL, headers=headers)
        response.raise_for_status()  # Raises an HTTPError for bad responses

        data = response.json()
        available_instances = []

        for instance_type, info in data["data"].items():
            gpu_name = info["instance_type"].get("gpu_description", "").upper()
            if any(gpu in gpu_name for gpu in COMPATIBLE_GPUS):
                regions = info["regions_with_capacity_available"]
                if regions:
                    available_instances.append(
                        {"type": instance_type, "gpu": gpu_name, "regions": [region["name"] for region in regions]}
                    )

        return available_instances
    except requests.exceptions.RequestException as e:
        print(f"Error accessing the API: {e}")
        return None


def send_slack_dm(message):
    if not SLACK_TOKEN:
        print("Error: SLACK_BOT_TOKEN environment variable not set.")
        return
    if not SLACK_USER_ID:
        print("Error: SLACK_USER_ID environment variable not set.")
        return

    client = WebClient(token=SLACK_TOKEN)

    try:
        # Open a direct message channel
        response = client.conversations_open(users=SLACK_USER_ID)
        channel_id = response["channel"]["id"]

        # Send the message
        response = client.chat_postMessage(channel=channel_id, text=message)
        print(f"Slack DM sent: {response['ts']}")
    except SlackApiError as e:
        print(f"Error sending Slack DM: {e}")


def main():
    check_interval = 600  # seconds

    api_key = read_api_key()
    if not api_key:
        print("Exiting due to missing API key.")
        return

    print(f"API key loaded successfully from {API_KEY_FILE}")

    while True:
        print(f"\nChecking availability... {time.strftime('%Y-%m-%d %H:%M:%S')}")
        available_instances = check_availability(api_key)

        if available_instances is not None:
            if available_instances:
                print("GPU Instances AVAILABLE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                message = "GPU Instances Available:\n"
                for instance in available_instances:
                    instance_info = f"Type: {instance['type']}\nGPU: {instance['gpu']}\nRegions: {', '.join(instance['regions'])}\n\n"
                    print(instance_info)
                    message += instance_info
                send_slack_dm(message)
            else:
                print("No RTX6000 or A6000 instances available at this time.")

        print(f"Next check in {check_interval} seconds.")
        print("-" * 40)
        time.sleep(check_interval)


if __name__ == "__main__":
    main()
