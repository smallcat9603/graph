import slackweb
import os
import socket


def notify_slack(
    title=None,
    result=None,
    with_mention=False,
):
    slack = slackweb.Slack(
        url=os.environ.get("SLACK_INCOMING_WEBHOOK_URL")
    )
    hostname = socket.gethostname()
    text = "*Experiment Finished on " + hostname + "*"
    if with_mention:
        text = "<@UTELMRVU0> " + text
    if title:
        text += "\nTitle: " + title
    if result:
        text += "\nResult: " + result
    slack.notify(text=text)
