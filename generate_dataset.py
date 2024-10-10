import json

def generate_chat_dataset():
    chat_list = []
    with open("./datasets/human_chat.txt", "r") as fl:
        message_list = fl.readlines()
    for msg in message_list:
        try:
            msg_content = msg.split(':')[1]
            chat_list.append(msg_content.replace("\n", ""))
        except Exception as e:
            pass
    with open("./datasets/chat4.json", "w") as fl:
        json.dump(chat_list, fl)



generate_chat_dataset()